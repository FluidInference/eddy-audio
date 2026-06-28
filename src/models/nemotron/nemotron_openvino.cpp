// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0
//
// Cache-aware streaming inference for NVIDIA Nemotron-3.5-ASR-Streaming
// Multilingual 0.6B. Port of the validated Python reference
// (nemotron-ov-export/transcribe_ov.py), which mirrors mobius's CoreML
// streaming loop: chunk raw audio -> preprocessor -> cache-aware encoder
// (+ prompt_id) -> greedy RNNT decode, carrying encoder caches and LSTM
// state across chunks.

#include "eddy/models/nemotron/nemotron.hpp"

#include "eddy/backends/openvino_backend.hpp"

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>

#include <string_view>

#include <algorithm>
#include <chrono>
#include <cmath>
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

  ov::CompiledModel preproc, encoder, decoder, joint;
  ov::InferRequest preproc_req, encoder_req, decoder_req, joint_req;

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

  ov::element::Type token_et = ov::element::i32;

  std::once_flag compile_once;
  std::mutex infer_guard;

  size_t chunk_samples() const {
    return static_cast<size_t>(static_cast<double>(chunk_mel_frames) * 0.01 * sample_rate);
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

void OpenVINONemotron::ensure_compiled() {
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

    // --- Compile models (preprocessor on CPU; rest on chosen device) ---
    std::string preproc_device = "CPU";
    if (const char* env = std::getenv("EDDY_PREPROC_DEVICE")) {
      if (*env) preproc_device = env;
    }
    impl_->preproc = core.compile_model(impl_->paths.preprocessor, preproc_device);
    impl_->encoder = core.compile_model(impl_->paths.encoder, device);
    impl_->decoder = core.compile_model(impl_->paths.decoder, device);
    impl_->joint = core.compile_model(impl_->paths.joint, device);

    impl_->preproc_req = impl_->preproc.create_infer_request();
    impl_->encoder_req = impl_->encoder.create_infer_request();
    impl_->decoder_req = impl_->decoder.create_infer_request();
    impl_->joint_req = impl_->joint.create_infer_request();

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

  const int prompt_id = resolve_prompt_id(I.config.language);

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

  const ov::Tensor prompt_tensor = make_i32(prompt_id);

  size_t off = 0;
  while (off < pcm.size()) {
    const size_t end = std::min(off + chunk_samples, pcm.size());

    // Raw audio chunk, padded to chunk_samples.
    ov::Tensor audio(ov::element::f32, ov::Shape{1, chunk_samples});
    float* adst = audio.data<float>();
    std::memset(adst, 0, audio.get_byte_size());
    std::copy(pcm.begin() + static_cast<long>(off), pcm.begin() + static_cast<long>(end), adst);

    // Preprocessor: audio -> mel [1, bins, T_mel]
    I.preproc_req.set_tensor("audio", audio);
    I.preproc_req.set_tensor("audio_length", make_i32(static_cast<int>(chunk_samples)));
    I.preproc_req.infer();
    const ov::Tensor mel_out = I.preproc_req.get_tensor("mel");
    const ov::Shape mel_shape = mel_out.get_shape();  // [1, bins, T_mel]
    const size_t t_mel = mel_shape[2];
    const float* mel_src = mel_out.data<float>();

    // Build encoder mel input [1, bins, total]: prepend cache (or zero
    // pre_encode_cache on first chunk), then pad/trim to total.
    ov::Tensor mel_in(ov::element::f32, ov::Shape{1, bins, total});
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
    mel_cache.data.assign(bins * keep, 0.0f);
    for (size_t bin = 0; bin < bins; ++bin) {
      for (size_t t = 0; t < keep; ++t) {
        mel_cache.data[bin * keep + t] = mel_src[bin * t_mel + (t_mel - keep + t)];
      }
    }

    // Encoder: mel + caches + prompt_id -> encoded + caches
    I.encoder_req.set_tensor("mel", mel_in);
    I.encoder_req.set_tensor("mel_length", make_i32(static_cast<int>(total)));
    I.encoder_req.set_tensor("cache_channel", cache_channel);
    I.encoder_req.set_tensor("cache_time", cache_time);
    I.encoder_req.set_tensor("cache_len", cache_len);
    I.encoder_req.set_tensor("prompt_id", prompt_tensor);
    I.encoder_req.infer();

    const ov::Tensor encoded = I.encoder_req.get_tensor("encoded");  // [1, D, T_enc]
    // Persist updated caches (copy out before next infer overwrites them).
    {
      const ov::Tensor cc = I.encoder_req.get_tensor("cache_channel_out");
      const ov::Tensor ctt = I.encoder_req.get_tensor("cache_time_out");
      const ov::Tensor cl = I.encoder_req.get_tensor("cache_len_out");
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
        // Decoder
        ov::Tensor token(I.token_et, ov::Shape{1, 1});
        if (I.token_et == ov::element::i64) {
          token.data<int64_t>()[0] = last_token;
        } else {
          token.data<int32_t>()[0] = last_token;
        }
        I.decoder_req.set_tensor("token", token);
        I.decoder_req.set_tensor("token_length", make_i32(1));
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
        // Advance LSTM state on emission.
        std::memcpy(h.data<float>(), I.decoder_req.get_tensor("h_out").data<float>(), h.get_byte_size());
        std::memcpy(c.data<float>(), I.decoder_req.get_tensor("c_out").data<float>(), c.get_byte_size());
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
