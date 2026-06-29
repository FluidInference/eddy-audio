// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0
//
// Cache-aware streaming inference for NVIDIA Nemotron FastConformer-RNNT ASR.
// Serves both the 3.5-ASR-Streaming-Multilingual 0.6B model (per-chunk prompt_id
// language conditioning) and the English speech-streaming 0.6B model (no prompt,
// auto-detected from the encoder's inputs).
//
// Pipeline per chunk: native C++ mel featurizer -> cache-aware encoder
// (+ prompt_id when present) -> greedy RNNT decode, carrying the encoder caches
// and decoder LSTM state across chunks.

#include "eddy/models/nemotron/nemotron.hpp"

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/models/nemotron/nemotron_featurizer.hpp"

#include <memory>

#include <openvino/openvino.hpp>
#include <openvino/op/bitwise_not.hpp>
#include <openvino/op/logical_not.hpp>
#include <openvino/core/graph_util.hpp>
#include <nlohmann/json.hpp>

#include <string_view>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

namespace eddy::nemotron {

namespace {

// Mel feature buffer in [bins, frames] row-major (bin-major, matching the
// [1, bins, T] tensor layout the encoder expects).
struct MelBuf {
  std::vector<float> data;  // size = bins * frames
  size_t bins = 0;
  size_t frames = 0;
};

// The OpenVINO NPU plugin miscompiles BitwiseNot on a boolean tensor: it does
// an integer bitwise complement, so ~0 = -1 and ~1 = -2 are *both* nonzero
// ("true"). The FastConformer attention mask is built with a `~` over a bool,
// so on NPU the mask becomes all-true -> every key is masked -> uniform softmax
// -> the encoder output collapses to ~0 and every transcript is empty. Replace
// BitwiseNot(bool) with the semantically identical LogicalNot, which the NPU
// compiles correctly; it is a no-op on CPU/GPU. Returns the (possibly rewritten)
// model ready to compile.
std::shared_ptr<ov::Model> load_npu_safe(ov::Core& core, const std::string& xml) {
  auto model = core.read_model(xml);
  bool changed = false;
  for (const auto& node : model->get_ordered_ops()) {
    // Only a BitwiseNot over a boolean is equivalent to LogicalNot. Guard on the
    // input element type so a future IR with an integer BitwiseNot isn't silently
    // miscompiled (LogicalNot would change both semantics and output dtype).
    if (ov::as_type_ptr<ov::op::v13::BitwiseNot>(node) &&
        node->get_input_element_type(0) == ov::element::boolean) {
      auto repl = std::make_shared<ov::op::v1::LogicalNot>(node->input_value(0));
      repl->set_friendly_name(node->get_friendly_name());
      ov::copy_runtime_info(node, repl);
      ov::replace_node(node, repl);
      changed = true;
    }
  }
  if (changed) model->validate_nodes_and_infer_types();
  return model;
}

ov::Tensor make_i32(int value) {
  ov::Tensor t(ov::element::i32, ov::Shape{1});
  t.data<int32_t>()[0] = value;
  return t;
}

// SentencePiece word boundary marker (U+2581 "▁"). Unlike Parakeet,
// Nemotron's multilingual tokenizer emits standalone ▁ tokens, so the
// faithful decode is "concatenate pieces, then replace ▁ with space"
// (matches the validated Python reference), not per-piece prefix logic.
constexpr std::string_view kWordBoundary = "\xE2\x96\x81";

std::string finalize_text(std::string s) {
  // Replace every ▁ with a space.
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    if (s.compare(i, kWordBoundary.size(), kWordBoundary) == 0) {
      out.push_back(' ');
      i += kWordBoundary.size();
    } else {
      out.push_back(s[i]);
      ++i;
    }
  }
  // Trim leading/trailing whitespace.
  const auto b = out.find_first_not_of(" \t\n\r");
  const auto e = out.find_last_not_of(" \t\n\r");
  if (b == std::string::npos) return "";
  return out.substr(b, e - b + 1);
}

}  // namespace

struct OpenVINONemotron::Impl {
  std::shared_ptr<eddy::OpenVINOBackend> backend;
  ModelPaths paths;
  Config config;

  std::vector<std::string> vocab;  // id -> piece (raw, ▁-marked)

  ov::CompiledModel encoder, decoder, joint;
  ov::InferRequest encoder_req, decoder_req, joint_req;

  // Native C++ log-mel featurizer replacing the nemotron_preprocessor.xml IR.
  std::unique_ptr<MelFeaturizer> featurizer;

  // Metadata
  int sample_rate = 16000;
  int mel_features = 128;
  int chunk_mel_frames = 112;
  int pre_encode_cache = 9;
  int total_mel_frames = 121;
  int blank_idx = 13087;
  int vocab_size = 13087;
  int decoder_hidden = 640;
  int decoder_layers = 2;
  int default_prompt_id = 101;
  ov::Shape cache_channel_shape;
  ov::Shape cache_time_shape;
  std::map<std::string, int> prompt_dictionary;
  std::set<int> lang_tag_token_ids;

  // Whether the encoder takes a `prompt_id` input. True for the multilingual
  // model (per-chunk language conditioning); false for the monolingual English
  // speech-streaming model. Auto-detected from the encoder's input ports so one
  // backend serves both variants.
  bool has_prompt = false;

  ov::element::Type token_et = ov::element::i32;

  std::once_flag compile_once;
  std::mutex infer_guard;

  size_t chunk_samples() const {
    // mel hop is 10 ms => frames * sample_rate / 100. Pure integer math avoids
    // the rounding hazard of multiplying by the non-representable 0.01.
    return static_cast<size_t>(chunk_mel_frames) * static_cast<size_t>(sample_rate) / 100;
  }
};

OpenVINONemotron::OpenVINONemotron(std::shared_ptr<eddy::OpenVINOBackend> backend,
                                   ModelPaths paths, Config config)
    : impl_(std::make_unique<Impl>()) {
  if (!backend) {
    throw std::invalid_argument("OpenVINO backend is null");
  }
  impl_->backend = std::move(backend);
  impl_->paths = std::move(paths);
  impl_->config = std::move(config);
}

OpenVINONemotron::~OpenVINONemotron() = default;

void OpenVINONemotron::warmup() { ensure_compiled(); }

int OpenVINONemotron::resolve_prompt_id(const std::string& language) const {
  // prompt_dictionary is populated by ensure_compiled(); make this safe to call
  // standalone (before transcribe()/warmup()). ensure_compiled() is const (it only
  // mutates *impl_, reachable through the unique_ptr in a const method) and
  // std::call_once guarded, so this is a cheap no-op once compiled.
  ensure_compiled();
  const auto& dict = impl_->prompt_dictionary;
  auto it = dict.find(language);
  if (it != dict.end()) {
    return it->second;
  }
  if (language.size() == 2) {
    const std::string prefix = language + "-";
    for (const auto& [k, v] : dict) {
      if (k.size() >= prefix.size() &&
          std::equal(prefix.begin(), prefix.end(), k.begin(),
                     [](char a, char b) { return std::tolower(a) == std::tolower(b); })) {
        return v;
      }
    }
  }
  return impl_->default_prompt_id;
}

void OpenVINONemotron::ensure_compiled() const {
  std::call_once(impl_->compile_once, [this]() {
    auto& core = impl_->backend->core();
    const std::string device = impl_->config.device.empty() ? "AUTO" : impl_->config.device;

    // --- Load metadata.json ---
    {
      std::ifstream f(impl_->paths.metadata_json);
      if (!f.good()) {
        throw std::runtime_error("Failed to open Nemotron metadata: " + impl_->paths.metadata_json);
      }
      nlohmann::json m;
      f >> m;
      impl_->sample_rate = m.value("sample_rate", 16000);
      impl_->mel_features = m.value("mel_features", 128);
      impl_->chunk_mel_frames = m.value("chunk_mel_frames", 112);
      impl_->pre_encode_cache = m.value("pre_encode_cache", 9);
      impl_->total_mel_frames = m.value("total_mel_frames", 121);
      impl_->blank_idx = m.value("blank_idx", 13087);
      impl_->vocab_size = m.value("vocab_size", 13087);
      impl_->decoder_hidden = m.value("decoder_hidden", 640);
      impl_->decoder_layers = m.value("decoder_layers", 2);
      impl_->default_prompt_id = m.value("default_prompt_id", 101);

      auto to_shape = [](const nlohmann::json& arr) {
        ov::Shape s;
        for (const auto& d : arr) s.push_back(d.get<size_t>());
        return s;
      };
      // These two keys have no sensible default (they size the encoder caches),
      // so require them explicitly with a path-aware message rather than letting
      // nlohmann's bare "key not found" propagate from a truncated metadata.json.
      if (!m.contains("cache_channel_shape") || !m.contains("cache_time_shape")) {
        throw std::runtime_error(
            "Nemotron metadata missing required cache shape keys "
            "(cache_channel_shape / cache_time_shape): " + impl_->paths.metadata_json);
      }
      impl_->cache_channel_shape = to_shape(m.at("cache_channel_shape"));
      impl_->cache_time_shape = to_shape(m.at("cache_time_shape"));

      if (m.contains("prompt_dictionary")) {
        for (auto& [k, v] : m["prompt_dictionary"].items()) {
          impl_->prompt_dictionary[k] = v.get<int>();
        }
      }
      if (m.contains("lang_tag_token_ids")) {
        for (const auto& id : m["lang_tag_token_ids"]) {
          impl_->lang_tag_token_ids.insert(id.get<int>());
        }
      }
    }

    // --- Mel featurizer (native C++, replaces nemotron_preprocessor.xml) ---
    // The IR preprocessor is dynamic-shaped (NPU-incompatible) and adds an
    // OV inference per chunk; the native featurizer reproduces it exactly
    // (validated to fp16-storage precision). paths.preprocessor is unused.
    impl_->featurizer = std::make_unique<MelFeaturizer>(impl_->sample_rate, impl_->mel_features);
    // Guard: the featurizer's framing must match the model's expected geometry.
    // A full chunk (chunk_mel_frames * sample_rate/100 samples) must yield
    // chunk_mel_frames + 1 mel frames; otherwise metadata (sample_rate /
    // chunk_mel_frames) disagrees with the hardcoded 10 ms hop and the mel would
    // silently misalign with the encoder.
    {
      const size_t cs = impl_->chunk_samples();
      std::vector<float> probe(cs, 0.0f);
      std::vector<float> mel_probe;
      size_t probe_frames = 0;
      impl_->featurizer->compute(probe.data(), cs, static_cast<int>(cs), mel_probe, probe_frames);
      const size_t expected = static_cast<size_t>(impl_->chunk_mel_frames) + 1;
      if (probe_frames != expected) {
        throw std::runtime_error(
            "Nemotron C++ featurizer geometry mismatch: produced " +
            std::to_string(probe_frames) + " frames for a chunk, expected " +
            std::to_string(expected) + " (chunk_mel_frames=" +
            std::to_string(impl_->chunk_mel_frames) + ", sample_rate=" +
            std::to_string(impl_->sample_rate) + "). The model's featurizer "
            "config differs from the hardcoded 25 ms/10 ms framing.");
      }
    }

    // --- Compile models on the chosen device ---
    // The encoder carries the attention-mask BitwiseNot that the NPU plugin
    // miscompiles, so load it through the rewrite (no-op on CPU/GPU).
    impl_->encoder = core.compile_model(load_npu_safe(core, impl_->paths.encoder), device);
    impl_->decoder = core.compile_model(impl_->paths.decoder, device);
    impl_->joint = core.compile_model(impl_->paths.joint, device);

    impl_->encoder_req = impl_->encoder.create_infer_request();
    impl_->decoder_req = impl_->decoder.create_infer_request();
    impl_->joint_req = impl_->joint.create_infer_request();

    // Detect prompt conditioning from the encoder's input ports: the
    // multilingual encoder has a "prompt_id" input; the English speech-streaming
    // encoder does not. Drives whether transcribe() feeds a prompt_id tensor.
    impl_->has_prompt = false;
    for (const auto& in : impl_->encoder.inputs()) {
      if (in.get_names().count("prompt_id")) { impl_->has_prompt = true; break; }
    }

    impl_->token_et = impl_->decoder.input("token").get_element_type();

    // --- Vocab (id -> piece). Flat {"0":"piece", ...} format. ---
    {
      std::ifstream f(impl_->paths.vocab_json);
      if (!f.good()) {
        throw std::runtime_error("Failed to open Nemotron vocab: " + impl_->paths.vocab_json);
      }
      nlohmann::json v;
      f >> v;
      size_t max_id = 0;
      for (auto& [k, _] : v.items()) {
        max_id = std::max(max_id, static_cast<size_t>(std::stoul(k)));
      }
      impl_->vocab.assign(max_id + 1, std::string{});
      for (auto& [k, val] : v.items()) {
        impl_->vocab[std::stoul(k)] = val.get<std::string>();
      }
    }
  });
}

TranscriptionResult OpenVINONemotron::transcribe(const std::vector<float>& pcm) {
  ensure_compiled();
  std::lock_guard<std::mutex> lock(impl_->infer_guard);

  const auto t_start = std::chrono::steady_clock::now();

  auto& I = *impl_;
  const size_t bins = static_cast<size_t>(I.mel_features);
  const size_t total = static_cast<size_t>(I.total_mel_frames);
  const size_t pre_cache = static_cast<size_t>(I.pre_encode_cache);
  const size_t chunk_samples = I.chunk_samples();

  // prompt_id only applies to the multilingual (prompt-conditioned) encoder.
  const int prompt_id = I.has_prompt ? resolve_prompt_id(I.config.language) : 0;

  // Persistent encoder caches (carried across chunks).
  ov::Tensor cache_channel(ov::element::f32, I.cache_channel_shape);
  ov::Tensor cache_time(ov::element::f32, I.cache_time_shape);
  std::memset(cache_channel.data<float>(), 0, cache_channel.get_byte_size());
  std::memset(cache_time.data<float>(), 0, cache_time.get_byte_size());
  ov::Tensor cache_len = make_i32(0);

  // Persistent LSTM state.
  const ov::Shape lstm_shape{static_cast<size_t>(I.decoder_layers), 1,
                             static_cast<size_t>(I.decoder_hidden)};
  ov::Tensor h(ov::element::f32, lstm_shape);
  ov::Tensor c(ov::element::f32, lstm_shape);
  std::memset(h.data<float>(), 0, h.get_byte_size());
  std::memset(c.data<float>(), 0, c.get_byte_size());

  int last_token = I.blank_idx;
  std::vector<int> all_tokens;

  MelBuf mel_cache;  // last pre_encode_cache frames of previous chunk's mel
  std::vector<float> mel_scratch;  // per-chunk featurizer output [bins * t_mel]

  const ov::Tensor prompt_tensor = make_i32(prompt_id);  // unused when !has_prompt

  // Hot-path tensors whose shapes/values are fixed for the whole call — allocate
  // once and reuse across chunks and the inner RNNT loop instead of re-allocating
  // every iteration (the inner `token`/`token_length` allocs dominate otherwise).
  ov::Tensor audio(ov::element::f32, ov::Shape{1, chunk_samples});
  ov::Tensor mel_in(ov::element::f32, ov::Shape{1, bins, total});
  ov::Tensor token(I.token_et, ov::Shape{1, 1});
  const ov::Tensor token_length = make_i32(1);
  const ov::Tensor mel_length = make_i32(static_cast<int>(total));

  size_t off = 0;
  while (off < pcm.size()) {
    const size_t end = std::min(off + chunk_samples, pcm.size());

    // Raw audio chunk, padded to chunk_samples (reusing the hoisted tensor).
    float* adst = audio.data<float>();
    std::memset(adst, 0, audio.get_byte_size());
    std::copy(pcm.begin() + static_cast<long>(off), pcm.begin() + static_cast<long>(end), adst);

    // Preprocessor: audio -> mel [bins, T_mel] (native C++ featurizer).
    // audio_length is intentionally the FULL padded chunk size on every chunk
    // (mirrors the WER-validated reference: the chunk is zero-padded to
    // chunk_samples and that full length is passed), so the length mask zeros
    // only the trailing frame, which the assembly below trims anyway.
    size_t t_mel = 0;
    I.featurizer->compute(adst, chunk_samples, static_cast<int>(chunk_samples),
                          mel_scratch, t_mel);
    const float* mel_src = mel_scratch.data();  // bin-major [bins * t_mel]

    // Build encoder mel input [1, bins, total]: prepend cache (or zero
    // pre_encode_cache on first chunk), then pad/trim to total. (reusing the
    // hoisted tensor; fully overwritten via memset + fills below.)
    float* mdst = mel_in.data<float>();
    std::memset(mdst, 0, mel_in.get_byte_size());

    const size_t cache_frames = mel_cache.frames;  // 0 on first chunk
    const size_t lead = (cache_frames > 0) ? cache_frames : pre_cache;  // zero-pad lead on first chunk
    for (size_t bin = 0; bin < bins; ++bin) {
      float* row = mdst + bin * total;
      size_t col = 0;
      // leading cache frames
      for (size_t t = 0; t < lead && col < total; ++t, ++col) {
        if (cache_frames > 0) {
          row[col] = mel_cache.data[bin * cache_frames + t];
        }  // else zero (already memset)
      }
      // current chunk mel frames
      for (size_t t = 0; t < t_mel && col < total; ++t, ++col) {
        row[col] = mel_src[bin * t_mel + t];
      }
    }

    // Update mel_cache = last pre_encode_cache frames of current chunk mel.
    const size_t keep = std::min(pre_cache, t_mel);
    mel_cache.bins = bins;
    mel_cache.frames = keep;
    // resize, not assign: every element is unconditionally overwritten by the
    // loop below (bin*keep + t is a bijection over [0, bins*keep)), so the
    // zero-fill assign() would do is pure waste on this per-chunk hot buffer.
    mel_cache.data.resize(bins * keep);
    for (size_t bin = 0; bin < bins; ++bin) {
      for (size_t t = 0; t < keep; ++t) {
        mel_cache.data[bin * keep + t] = mel_src[bin * t_mel + (t_mel - keep + t)];
      }
    }

    // Encoder: mel + caches + prompt_id -> encoded + caches
    I.encoder_req.set_tensor("mel", mel_in);
    I.encoder_req.set_tensor("mel_length", mel_length);
    I.encoder_req.set_tensor("cache_channel", cache_channel);
    I.encoder_req.set_tensor("cache_time", cache_time);
    I.encoder_req.set_tensor("cache_len", cache_len);
    if (I.has_prompt) I.encoder_req.set_tensor("prompt_id", prompt_tensor);
    I.encoder_req.infer();

    const ov::Tensor encoded = I.encoder_req.get_tensor("encoded");  // [1, D, T_enc]
    // Persist updated caches (copy out before next infer overwrites them).
    {
      const ov::Tensor cc = I.encoder_req.get_tensor("cache_channel_out");
      const ov::Tensor ctt = I.encoder_req.get_tensor("cache_time_out");
      const ov::Tensor cl = I.encoder_req.get_tensor("cache_len_out");
      // Cache-aware streaming: the *_out caches are the same fixed shape as the
      // input caches (the ring buffer is re-filled in place), so we copy back
      // into the pre-allocated input tensors. Check the byte sizes agree so a
      // mismatched IR export throws here instead of silently over-/under-reading.
      // Runtime check (not assert): Release builds define NDEBUG.
      if (cc.get_byte_size() != cache_channel.get_byte_size() ||
          ctt.get_byte_size() != cache_time.get_byte_size() ||
          cl.get_element_type() != ov::element::i32) {
        throw std::runtime_error(
            "Nemotron encoder cache_*_out shape/type differs from the pre-allocated "
            "input cache; the model IR does not match metadata.json cache shapes.");
      }
      std::memcpy(cache_channel.data<float>(), cc.data<float>(), cache_channel.get_byte_size());
      std::memcpy(cache_time.data<float>(), ctt.data<float>(), cache_time.get_byte_size());
      cache_len.data<int32_t>()[0] = cl.data<int32_t>()[0];
    }

    const ov::Shape enc_shape = encoded.get_shape();  // [1, D, T_enc]
    const size_t enc_d = enc_shape[1];
    const size_t t_enc = enc_shape[2];
    const float* enc_data = encoded.data<float>();

    // Greedy RNNT decode over encoder frames.
    ov::Tensor enc_step(ov::element::f32, ov::Shape{1, enc_d, 1});
    for (size_t t = 0; t < t_enc; ++t) {
      float* es = enc_step.data<float>();
      for (size_t ch = 0; ch < enc_d; ++ch) {
        es[ch] = enc_data[ch * t_enc + t];
      }

      for (size_t sym = 0; sym < I.config.max_symbols_per_frame; ++sym) {
        // Decoder (token/token_length tensors hoisted above the loops; just
        // overwrite the scalar token value each iteration).
        if (I.token_et == ov::element::i64) {
          token.data<int64_t>()[0] = last_token;
        } else {
          token.data<int32_t>()[0] = last_token;
        }
        I.decoder_req.set_tensor("token", token);
        I.decoder_req.set_tensor("token_length", token_length);
        I.decoder_req.set_tensor("h_in", h);
        I.decoder_req.set_tensor("c_in", c);
        I.decoder_req.infer();
        const ov::Tensor dec_out = I.decoder_req.get_tensor("decoder_out");  // [1, H, 1]

        // Joint
        I.joint_req.set_tensor("encoder", enc_step);
        I.joint_req.set_tensor("decoder", dec_out);
        I.joint_req.infer();
        const ov::Tensor logits = I.joint_req.get_tensor("logits");  // [1,1,1,V]
        const float* lg = logits.data<float>();
        const size_t vsz = logits.get_size();

        int best = 0;
        float best_score = lg[0];
        for (size_t i = 1; i < vsz; ++i) {
          if (lg[i] > best_score) {
            best_score = lg[i];
            best = static_cast<int>(i);
          }
        }

        if (best == I.blank_idx) {
          break;
        }
        all_tokens.push_back(best);
        last_token = best;
        // Advance LSTM state on emission. h_out/c_out are the same fixed shape as
        // h_in/c_in by construction; guard so a mismatched decoder IR throws
        // instead of corrupting the fixed-size state tensors.
        const ov::Tensor h_out = I.decoder_req.get_tensor("h_out");
        const ov::Tensor c_out = I.decoder_req.get_tensor("c_out");
        if (h_out.get_byte_size() != h.get_byte_size() ||
            c_out.get_byte_size() != c.get_byte_size()) {
          throw std::runtime_error(
              "Nemotron decoder h_out/c_out byte size differs from the LSTM state "
              "tensors; the decoder IR does not match the expected layer/hidden dims.");
        }
        std::memcpy(h.data<float>(), h_out.data<float>(), h.get_byte_size());
        std::memcpy(c.data<float>(), c_out.data<float>(), c.get_byte_size());
      }
    }

    off += chunk_samples;
  }

  // Strip blank / out-of-range / language-tag tokens; concatenate pieces.
  auto piece = [&](int tok) -> const std::string& {
    static const std::string empty;
    return (tok >= 0 && tok < static_cast<int>(I.vocab.size())) ? I.vocab[tok] : empty;
  };

  TranscriptionResult result;
  result.prompt_id_used = prompt_id;
  result.token_ids = all_tokens;
  std::string body;
  for (int tok : all_tokens) {
    // blank intentionally sits at index vocab_size (== blank_idx), so the
    // explicit blank check is redundant with `tok >= vocab_size`; kept for
    // clarity since the two are configured independently from metadata.json.
    if (tok == I.blank_idx || tok >= I.vocab_size) continue;
    if (I.lang_tag_token_ids.count(tok)) {
      if (result.detected_language.empty()) {
        result.detected_language = finalize_text(piece(tok));
      }
      continue;
    }
    body += piece(tok);
  }
  result.text = finalize_text(body);

  const auto t_end = std::chrono::steady_clock::now();
  result.latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  return result;
}

}  // namespace eddy::nemotron
