#include "eddy/models/parakeet/parakeet_openvino.hpp"

#include <openvino/openvino.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <optional>

namespace eddy::parakeet {

namespace {

constexpr std::string_view kWordBoundary = "\xE2\x96\x81";  // SentencePiece space marker "▁"

class Tokenizer {
public:
  void load(const std::string& path, int blank_id) {
    std::ifstream stream(path);
    if (!stream.good()) {
      throw std::runtime_error("Failed to open Parakeet vocabulary: " + path);
    }

    nlohmann::json json;
    stream >> json;

    size_t max_id = 0;
    for (const auto& item : json.items()) {
      const size_t id = static_cast<size_t>(std::stoul(item.key()));
      max_id = std::max(max_id, id);
    }

    vocab_.assign(max_id + 1, std::string{});
    for (const auto& item : json.items()) {
      const size_t id = static_cast<size_t>(std::stoul(item.key()));
      vocab_[id] = item.value().get<std::string>();
    }

    blank_id_ = blank_id;
    if (blank_id_ >= static_cast<int>(vocab_.size())) {
      vocab_.resize(static_cast<size_t>(blank_id_) + 1U);
    }
  }

  [[nodiscard]] std::string decode(const std::vector<int>& token_ids) const {
    std::string result;
    bool first_piece = true;
    for (int token_id : token_ids) {
      if (token_id == blank_id_ || token_id < 0) {
        continue;
      }
      const auto idx = static_cast<size_t>(token_id);
      if (idx >= vocab_.size()) {
        continue;
      }
      std::string_view piece{vocab_[idx]};
      if (piece.empty()) {
        continue;
      }

      bool prepend_space = piece.starts_with(kWordBoundary);
      if (prepend_space) {
        piece.remove_prefix(kWordBoundary.size());
      }

      if (!piece.empty()) {
        if (!first_piece && prepend_space && !result.empty()) {
          result.push_back(' ');
        }
        result.append(piece);
        first_piece = false;
      }
    }
    return result;
  }

  [[nodiscard]] size_t vocab_size() const { return vocab_.size(); }
  [[nodiscard]] int blank_id() const { return blank_id_; }

  [[nodiscard]] bool is_punctuation(int token_id) const {
    if (token_id < 0) return false;
    const auto idx = static_cast<size_t>(token_id);
    if (idx >= vocab_.size()) return false;
    std::string_view piece{vocab_[idx]};
    if (piece.empty()) return false;
    // Strip SentencePiece word boundary marker if present
    if (piece.starts_with(kWordBoundary)) {
      piece.remove_prefix(kWordBoundary.size());
    }
    return (piece == "." || piece == "?" || piece == "!");
  }

private:
  std::vector<std::string> vocab_;
  int blank_id_ = 1024;
};

ov::AnyMap make_compile_cfg_from_env() {
  ov::AnyMap cfg;
  if (const char* perf = std::getenv("EDDY_OV_PERF")) {
    std::string v(perf);
    for (auto& c : v) c = static_cast<char>(::toupper(c));
    if (v == "LATENCY") cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    else if (v == "THROUGHPUT") cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
  }
  if (const char* nr = std::getenv("EDDY_OV_NUM_REQUESTS")) {
    try { int n = std::max(1, std::stoi(nr)); cfg[ov::hint::num_requests.name()] = n; } catch (...) {}
  }
  if (const char* th = std::getenv("EDDY_OV_THREADS")) {
    try { int n = std::max(1, std::stoi(th)); cfg[ov::inference_num_threads.name()] = n; } catch (...) {}
  }
  if (const char* ip = std::getenv("EDDY_OV_PRECISION")) {
    std::string v(ip);
    for (auto& c : v) c = static_cast<char>(::toupper(c));
    if (v == "FP16" || v == "F16") cfg[ov::hint::inference_precision.name()] = ov::element::f16;
    else if (v == "FP32" || v == "F32") cfg[ov::hint::inference_precision.name()] = ov::element::f32;
    else if (v == "INT8" || v == "I8") cfg[ov::hint::inference_precision.name()] = ov::element::i8;
  }
  return cfg;
}

ov::CompiledModel compile_component(ov::Core& core, const ModelFile& file, const std::string& device) {
  if (file.path.empty()) {
    throw std::invalid_argument("Parakeet component path is empty");
  }

  auto cfg = make_compile_cfg_from_env();

  if (file.compiled) {
    std::ifstream blob_stream(file.path, std::ios::binary);
    if (!blob_stream.good()) {
      throw std::runtime_error("Failed to open compiled blob: " + file.path);
    }
    if (!cfg.empty()) return core.import_model(blob_stream, device, cfg);
    return core.import_model(blob_stream, device);
  }

  if (!cfg.empty()) return core.compile_model(file.path, device, cfg);
  return core.compile_model(file.path, device);
}

// Try to get an input/output port by any of the provided names; if none found,
// return an empty optional.
std::optional<ov::Output<const ov::Node>> try_input_by_names(const ov::CompiledModel& model,
                                                             const std::vector<std::string>& names) {
  for (const auto& name : names) {
    try {
      return model.input(name);
    } catch (...) {
      // ignore and continue
    }
  }
  return std::nullopt;
}

// Helper to resolve encoder ports in a name/shape-robust way.
struct EncoderPorts {
  std::optional<ov::Output<const ov::Node>> mel_in;   // [1, 128, T]
  std::optional<ov::Output<const ov::Node>> len_in;   // [1]
  std::optional<ov::Output<const ov::Node>> enc_out;  // [1, hidden, time]
};

EncoderPorts select_encoder_ports(const ov::CompiledModel& model) {
  EncoderPorts ports;

  // Prefer common friendly names first
  ports.mel_in = try_input_by_names(model, {"melspectogram", "melspectrogram", "mel", "melspec"});
  ports.len_in = try_input_by_names(model, {"melspectogram_length", "melspectrogram_length", "mel_length", "length"});

  // Fallback: scan inputs by shape
  if (!ports.mel_in.has_value() || !ports.len_in.has_value()) {
    for (const auto& p : model.inputs()) {
      try {
        const auto pshape = p.get_partial_shape();
        if (!ports.mel_in.has_value()) {
          if (pshape.rank().is_static() && pshape.rank().get_length() == 3) {
            const auto d1 = pshape[1];
            if (d1.is_static() && d1.get_length() == 128) {
              ports.mel_in = p;
              continue;
            }
          }
        }
        if (!ports.len_in.has_value()) {
          if (pshape.rank().is_static() && pshape.rank().get_length() == 1) {
            ports.len_in = p;
            continue;
          }
        }
      } catch (...) {
        // ignore this port
      }
    }
  }

  // Resolve encoder output: name first, then by shape
  try {
    ports.enc_out = model.output("encoder_output");
  } catch (...) {
    for (const auto& p : model.outputs()) {
      try {
        const auto shape = p.get_partial_shape();
        if (shape.rank().is_static() && shape.rank().get_length() == 3) {
          ports.enc_out = p;
          break;
        }
      } catch (...) {
        // keep searching
      }
    }
  }

  return ports;
}

struct MelFeatures {
  // Native tensors from preprocessor (for zero-copy encoder handoff when shapes match)
  ov::Tensor mel_tensor;       // [1, 128, T]
  ov::Tensor length_tensor;    // [1]
  size_t time_steps = 0;       // T

  // Convenience buffer for chunking path (time-major [mel_bins][time])
  std::vector<float> data;
  size_t frames = 0;           // valid frames (<= time_steps)
};

struct EncoderActivations {
  ov::Tensor tensor;      // [1, hidden, time]
  size_t hidden_size = 0;
  size_t time_steps = 0;
  size_t valid_frames = 0;
};

size_t required_length(const ov::Output<const ov::Node>& port) {
  const auto shape = port.get_partial_shape();
  if (shape.rank().is_static() && shape.rank().get_length() >= 3) {
    const auto len = shape[shape.size() - 1];
    if (len.is_static()) {
      return static_cast<size_t>(len.get_length());
    }
  }
  const auto static_shape = port.get_shape();
  return static_shape.empty() ? 0U : static_shape.back();
}

// Collapse immediate repeated tail subsequences in-place.
// If the last K tokens equal the K tokens immediately before them, drop the
// last K (and corresponding timings). This helps remove seam-induced echoes
// like "... X Y Z X Y Z" at chunk joins.
// (Tail dedup removed by request; rely on overlap dedup only.)

// Read a single scalar length value from a tensor that may be i32 or i64.
// Returns value as int64_t for consistency.
int64_t read_length_scalar(const ov::Tensor& t) {
  const auto et = t.get_element_type();
  if (et == ov::element::i64) {
    return t.data<int64_t>()[0];
  } else if (et == ov::element::i32) {
    return static_cast<int64_t>(t.data<int32_t>()[0]);
  } else {
    // Fallback: try to interpret as i64
    return t.data<int64_t>()[0];
  }
}

}  // namespace

std::shared_ptr<OpenVINOParakeet> make_openvino_parakeet(std::shared_ptr<eddy::OpenVINOBackend> backend,
                                                        ModelPaths model_paths,
                                                        RuntimeConfig runtime_cfg) {
  return std::make_shared<OpenVINOParakeet>(std::move(backend), std::move(model_paths), std::move(runtime_cfg));
}

struct OpenVINOParakeet::Impl {
  std::shared_ptr<eddy::OpenVINOBackend> backend;
  ModelPaths model_paths;
  RuntimeConfig runtime_cfg;

  Tokenizer tokenizer;

  ov::CompiledModel preproc_model;
  ov::InferRequest preproc_request;

  ov::CompiledModel encoder_model;
  ov::InferRequest encoder_request;

  ov::CompiledModel decoder_model;
  ov::InferRequest decoder_request;

  ov::CompiledModel joint_model;
  ov::InferRequest joint_request;

  size_t encoder_expected_frames = 0;
  size_t encoder_hidden_size = 0;
  size_t decoder_hidden_size = 0;
  size_t joint_output_size = 0;

  std::once_flag compile_once;
  std::mutex request_guard;

  // Resolved encoder ports
  EncoderPorts encoder_ports;

  // Output indices for encoder outputs (robust retrieval)
  size_t encoder_output_index = 0;   // [1, hidden, time]
  size_t encoder_length_index = 1;   // [1]
};

OpenVINOParakeet::OpenVINOParakeet(std::shared_ptr<eddy::OpenVINOBackend> backend,
                                   ModelPaths model_paths,
                                   RuntimeConfig runtime_cfg)
    : impl_(std::make_unique<Impl>()) {
  if (!backend) {
    throw std::invalid_argument("OpenVINO backend is null");
  }

  if (model_paths.preprocessor.path.empty() || model_paths.encoder.path.empty() ||
      model_paths.decoder.path.empty() || model_paths.joint.path.empty()) {
    throw std::invalid_argument("Parakeet model paths must include preprocessor, encoder, decoder, and joint graphs");
  }
  if (model_paths.tokenizer_json.empty()) {
    throw std::invalid_argument("Parakeet tokenizer path is empty");
  }
  if (runtime_cfg.duration_bins.empty()) {
    throw std::invalid_argument("Parakeet runtime config must provide at least one duration bin");
  }

  impl_->backend = std::move(backend);
  impl_->model_paths = std::move(model_paths);
  impl_->runtime_cfg = std::move(runtime_cfg);
}

OpenVINOParakeet::~OpenVINOParakeet() = default;

std::string OpenVINOParakeet::decode_tokens(const std::vector<int>& token_ids) {
  ensure_compiled_model();
  return impl_->tokenizer.decode(token_ids);
}

void OpenVINOParakeet::ensure_compiled_model() {
  std::call_once(impl_->compile_once, [this]() {
    // Allow environment overrides for critical runtime knobs without recompiling callers
    // EDDY_BLANK_ID: integer
    // EDDY_DURATION_BINS: comma-separated integers, e.g. "0,1,2,3,4,6,8,12,16,24"
    if (const char* env_blank = std::getenv("EDDY_BLANK_ID")) {
      try {
        int v = std::stoi(env_blank);
        if (v >= 0) {
          impl_->runtime_cfg.blank_token_id = v;
          std::cerr << "[CFG] Overriding blank_token_id from env: " << v << "\n";
        }
      } catch (...) {
        std::cerr << "[WARN] Failed to parse EDDY_BLANK_ID='" << env_blank << "'\n";
      }
    }
    if (const char* env_bins = std::getenv("EDDY_DURATION_BINS")) {
      std::vector<int> bins;
      std::string s(env_bins);
      size_t pos = 0;
      while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string token = s.substr(pos, comma == std::string::npos ? std::string::npos : (comma - pos));
        try {
          if (!token.empty()) bins.push_back(std::stoi(token));
        } catch (...) {}
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
      if (!bins.empty()) {
        impl_->runtime_cfg.duration_bins = std::move(bins);
        std::cerr << "[CFG] Overriding duration_bins from env (" << impl_->runtime_cfg.duration_bins.size() << " values)\n";
      } else {
        std::cerr << "[WARN] EDDY_DURATION_BINS provided but parsed empty\n";
      }
    }
    auto& core = impl_->backend->core();
    const std::string device = impl_->runtime_cfg.device.empty() ? "AUTO" : impl_->runtime_cfg.device;
    const bool target_npu = (device == "NPU");

    // Preprocessor device selection (override via EDDY_PREPROC_DEVICE)
    std::string preproc_device = target_npu ? std::string("CPU") : device;
    if (const char* env_pre = std::getenv("EDDY_PREPROC_DEVICE")) {
      if (*env_pre) preproc_device = env_pre;
    }
    impl_->preproc_model = compile_component(core, impl_->model_paths.preprocessor, preproc_device);
    impl_->preproc_request = impl_->preproc_model.create_infer_request();

    auto compile_with_fallback = [&](const ModelFile& file, const char* name) -> ov::CompiledModel {
      if (!target_npu) {
        return compile_component(core, file, device);
      }
      try {
        return compile_component(core, file, "NPU");
      } catch (const std::exception& e) {
        std::cerr << "[WARN] NPU compile failed for " << name << ": " << e.what() << "\n";
        std::cerr << "[WARN] Falling back to CPU for " << name << "\n";
        return compile_component(core, file, "CPU");
      }
    };

    impl_->encoder_model = compile_with_fallback(impl_->model_paths.encoder, "encoder");
    impl_->encoder_request = impl_->encoder_model.create_infer_request();

    // Decoder: try NPU with CPU fallback like others
    impl_->decoder_model = compile_with_fallback(impl_->model_paths.decoder, "decoder");
    impl_->decoder_request = impl_->decoder_model.create_infer_request();

    impl_->joint_model = compile_with_fallback(impl_->model_paths.joint, "joint");
    impl_->joint_request = impl_->joint_model.create_infer_request();

    impl_->tokenizer.load(impl_->model_paths.tokenizer_json, impl_->runtime_cfg.blank_token_id);

    // Resolve encoder ports robustly (supports 10s/15s variants and slight renames)
    impl_->encoder_ports = select_encoder_ports(impl_->encoder_model);
    if (!impl_->encoder_ports.mel_in.has_value() || !impl_->encoder_ports.len_in.has_value()) {
      throw std::runtime_error("Failed to resolve encoder input ports (mel/length)");
    }
    if (!impl_->encoder_ports.enc_out.has_value()) {
      throw std::runtime_error("Failed to resolve encoder output port");
    }

    impl_->encoder_expected_frames = required_length(impl_->encoder_ports.mel_in.value());

    // Determine encoder outputs and hidden size by scanning outputs
    const auto outs = impl_->encoder_model.outputs();
    bool found_output = false;
    bool found_len = false;
    for (size_t i = 0; i < outs.size(); ++i) {
      const auto& p = outs[i];
      try {
        const auto shape = p.get_shape();
        if (shape.size() == 3 && !found_output) {
          impl_->encoder_output_index = i;
          if (shape[1] == 0) {
            throw std::runtime_error("Encoder hidden size is zero");
          }
          impl_->encoder_hidden_size = shape[1];
          found_output = true;
        } else if (shape.size() == 1 && !found_len) {
          impl_->encoder_length_index = i;
          found_len = true;
        }
      } catch (...) {
        // ignore
      }
    }
    if (!found_output) {
      throw std::runtime_error("Failed to locate encoder main output [1, hidden, time]");
    }

    const auto decoder_state_shape = impl_->decoder_model.input("h_in").get_shape();
    if (decoder_state_shape.size() != 3) {
      throw std::runtime_error("Unexpected decoder hidden state shape");
    }
    impl_->decoder_hidden_size = decoder_state_shape[2];

    const auto joint_shape = impl_->joint_model.output("logits").get_shape();
    if (joint_shape.empty()) {
      throw std::runtime_error("Joint model logits tensor has no dimensions");
    }
    impl_->joint_output_size = joint_shape.back();

    // Use blank_id + 1 as the effective vocab size for validation
    // (vocab JSON may have extra entries beyond blank that aren't used)
    const auto effective_vocab_size = static_cast<size_t>(impl_->runtime_cfg.blank_token_id) + 1;
    const auto total_heads = effective_vocab_size + impl_->runtime_cfg.duration_bins.size();
    if (total_heads > impl_->joint_output_size) {
      throw std::runtime_error("Joint model output smaller than token+duration heads");
    }

    // Informative log to help diagnose mismatches between model head and configured bins
    std::cerr << "[CFG] Joint output size: " << impl_->joint_output_size
              << ", token head: " << effective_vocab_size
              << ", duration bins: " << impl_->runtime_cfg.duration_bins.size() << "\n";
  });
}

namespace {

MelFeatures run_preprocessor(OpenVINOParakeet::Impl& impl, const AudioSegment& segment) {
  if (segment.pcm.empty()) {
    throw std::invalid_argument("Audio segment contains no PCM samples");
  }
  if (segment.sample_rate != 16000) {
    throw std::invalid_argument("Parakeet OpenVINO pipeline expects 16 kHz audio samples");
  }

  // Detect static preprocessor input length (window size). If static, we will
  // process the audio in windows and concatenate mel outputs to avoid
  // truncating long utterances. If dynamic, we pass the full audio once.
  size_t window_samples = 0;  // 0 means dynamic length supported
  try {
    const auto pshape = impl.preproc_model.input(0).get_partial_shape();
    if (pshape.rank().is_static() && pshape.rank().get_length() >= 2) {
      const auto len_dim = pshape[1];
      if (len_dim.is_static()) {
        const auto static_len = static_cast<size_t>(len_dim.get_length());
        if (static_len > 0) {
          window_samples = static_len;  // fixed-size preprocessor
        }
      }
    }
  } catch (...) {
    // If shape query fails, assume dynamic
    window_samples = 0;
  }

  MelFeatures features;

  auto run_window = [&](const float* pcm_ptr, size_t pcm_count) -> std::pair<ov::Tensor, ov::Tensor> {
    const size_t req = window_samples > 0 ? window_samples : pcm_count;
    ov::Tensor audio_signal(ov::element::f32, {1, req});
    std::fill(audio_signal.data<float>(), audio_signal.data<float>() + audio_signal.get_size(), 0.0F);
    if (pcm_count) {
      std::copy(pcm_ptr, pcm_ptr + std::min(req, pcm_count), audio_signal.data<float>());
    }
    // Match preprocessor length element type to model port to avoid i32/i64 mismatches
    ov::Tensor audio_length;
    try {
      const auto len_port = impl.preproc_model.input(1);
      const auto len_et = len_port.get_element_type();
      if (len_et == ov::element::i64) {
        audio_length = ov::Tensor(ov::element::i64, {1});
        audio_length.data<int64_t>()[0] = static_cast<int64_t>(std::min(req, pcm_count));
      } else {
        audio_length = ov::Tensor(ov::element::i32, {1});
        audio_length.data<int32_t>()[0] = static_cast<int32_t>(std::min(req, pcm_count));
      }
    } catch (...) {
      // Fallback to i64
      audio_length = ov::Tensor(ov::element::i64, {1});
      audio_length.data<int64_t>()[0] = static_cast<int64_t>(std::min(req, pcm_count));
    }

    impl.preproc_request.set_input_tensor(0, audio_signal);  // audio_signal
    impl.preproc_request.set_input_tensor(1, audio_length);  // audio_length
    impl.preproc_request.infer();

    return {
      impl.preproc_request.get_output_tensor(0),  // melspectrogram
      impl.preproc_request.get_output_tensor(1)   // melspectrogram_length
    };
  };

  if (window_samples == 0 || segment.pcm.size() <= window_samples) {
    // Single-shot path: dynamic model or short audio
    auto [mel_tensor, length_tensor] = run_window(segment.pcm.data(), segment.pcm.size());

    const int64_t valid_frames = read_length_scalar(length_tensor);
    if (valid_frames <= 0) {
      throw std::runtime_error("Preprocessor returned zero mel frames");
    }

    const auto mel_shape = mel_tensor.get_shape();
    if (mel_shape.size() != 3 || mel_shape[1] != 128) {
      throw std::runtime_error("Unexpected mel tensor shape from preprocessor");
    }

    const size_t mel_bins = mel_shape[1];
    const size_t time_steps = mel_shape[2];
    const size_t elements = mel_bins * time_steps;

    // Retain tensors for potential zero-copy into encoder when shapes match
    features.mel_tensor = mel_tensor;
    features.length_tensor = length_tensor;
    features.time_steps = time_steps;
    features.frames = static_cast<size_t>(valid_frames);

    // Also keep a host copy for chunking operations (time-major layout)
    features.data.resize(elements);
    std::copy(mel_tensor.data<float>(), mel_tensor.data<float>() + elements, features.data.begin());
  } else {
    // Windowed path: fixed-size preprocessor and long audio.
    // Process in contiguous windows and append mel frames without overwriting
    // previous data. We collect per-bin sequences, then flatten once at end.
    const size_t total_samples = segment.pcm.size();
    size_t offset = 0;
    size_t total_frames = 0;
    constexpr size_t kMelBins = 128;

    // Accumulate per-bin frame sequences to avoid re-striding issues when growing
    std::vector<std::vector<float>> mel_bins(kMelBins);

    while (offset < total_samples) {
      const size_t remaining = total_samples - offset;
      const size_t this_count = std::min(window_samples, remaining);

      auto [mel_tensor, length_tensor] = run_window(segment.pcm.data() + offset, this_count);
      const int64_t vframes = read_length_scalar(length_tensor);
      if (vframes <= 0) {
        offset += this_count;
        continue;  // skip empty window (should not happen)
      }

      const auto mel_shape = mel_tensor.get_shape();
      if (mel_shape.size() != 3 || mel_shape[1] != kMelBins) {
        throw std::runtime_error("Unexpected mel tensor shape from preprocessor (windowed)");
      }
      const size_t time_steps = mel_shape[2];
      const size_t frames_to_append = static_cast<size_t>(std::min<int64_t>(vframes, static_cast<int64_t>(time_steps)));

      // Append frames for each mel bin; source layout is [bin][time]
      const float* src_base = mel_tensor.data<float>();
      for (size_t bin = 0; bin < kMelBins; ++bin) {
        const float* src = src_base + bin * time_steps;
        mel_bins[bin].insert(mel_bins[bin].end(), src, src + frames_to_append);
      }

      total_frames += frames_to_append;
      offset += this_count;
    }

    if (total_frames == 0) {
      throw std::runtime_error("Preprocessor produced no frames for long audio");
    }

    // Flatten into time-major buffer [bin][time] with a consistent stride of total_frames
    std::vector<float> mel_concat(kMelBins * total_frames);
    for (size_t bin = 0; bin < kMelBins; ++bin) {
      if (mel_bins[bin].size() != total_frames) {
        throw std::runtime_error("Internal error: inconsistent frame counts while concatenating mel frames");
      }
      std::copy(mel_bins[bin].begin(), mel_bins[bin].end(), mel_concat.data() + bin * total_frames);
    }

    features.frames = total_frames;
    features.time_steps = total_frames;
    features.data = std::move(mel_concat);
    // In windowed mode we don't retain a single mel_tensor/length_tensor; encoder path will use host copy
  }

  return features;
}

EncoderActivations run_encoder(OpenVINOParakeet::Impl& impl, const MelFeatures& mel) {
  if (impl.encoder_expected_frames == 0) {
    throw std::runtime_error("Encoder expected frame count is zero");
  }

  // Determine encoder length input element type to avoid device-specific mismatches (e.g., NPU strict typing)
  const auto len_port = impl.encoder_ports.len_in.value();
  const auto len_et = len_port.get_element_type();

  // Fast path: if preprocessor produced exactly the expected time axis, bind tensors directly
  if (mel.time_steps == impl.encoder_expected_frames && mel.mel_tensor) {
    // Bind mel using resolved port
    impl.encoder_request.set_tensor(impl.encoder_ports.mel_in.value(), mel.mel_tensor);
    // Create a length tensor with the encoder's required element type
    const size_t frames_for_len = std::min(mel.frames, mel.time_steps);
    ov::Tensor enc_len_tensor;
    if (len_et == ov::element::i64) {
      enc_len_tensor = ov::Tensor(ov::element::i64, {1});
      enc_len_tensor.data<int64_t>()[0] = static_cast<int64_t>(frames_for_len);
    } else {  // default to i32
      enc_len_tensor = ov::Tensor(ov::element::i32, {1});
      enc_len_tensor.data<int32_t>()[0] = static_cast<int32_t>(frames_for_len);
    }
    impl.encoder_request.set_tensor(len_port, enc_len_tensor);
  } else {
    // Fallback: copy into a padded/trimmed tensor matching encoder's expected frames
    ov::Tensor mel_tensor(ov::element::f32, {1, 128, impl.encoder_expected_frames});
    std::fill(mel_tensor.data<float>(), mel_tensor.data<float>() + mel_tensor.get_size(), 0.0F);

    const size_t frames_to_copy = std::min(impl.encoder_expected_frames, mel.frames);
    const size_t src_stride = mel.frames;
    const size_t dst_stride = impl.encoder_expected_frames;
    for (size_t bin = 0; bin < 128; ++bin) {
      const float* src = mel.data.data() + bin * src_stride;
      float* dst = mel_tensor.data<float>() + bin * dst_stride;
      std::copy(src, src + frames_to_copy, dst);
    }

    // Length tensor with correct element type
    ov::Tensor mel_length;
    if (len_et == ov::element::i64) {
      mel_length = ov::Tensor(ov::element::i64, {1});
      mel_length.data<int64_t>()[0] = static_cast<int64_t>(frames_to_copy);
    } else {
      mel_length = ov::Tensor(ov::element::i32, {1});
      mel_length.data<int32_t>()[0] = static_cast<int32_t>(frames_to_copy);
    }

    impl.encoder_request.set_tensor(impl.encoder_ports.mel_in.value(), mel_tensor);
    impl.encoder_request.set_tensor(impl.encoder_ports.len_in.value(), mel_length);
  }
  impl.encoder_request.infer();

  const auto encoder_tensor = impl.encoder_request.get_output_tensor(impl.encoder_output_index);  // encoder_output
  const auto encoder_length_tensor = impl.encoder_request.get_output_tensor(impl.encoder_length_index);  // encoder_output_length

  EncoderActivations activations;
  activations.hidden_size = impl.encoder_hidden_size;
  const auto shape = encoder_tensor.get_shape();
  if (shape.size() != 3 || shape[1] != impl.encoder_hidden_size) {
    throw std::runtime_error("Unexpected encoder output tensor shape");
  }
  activations.time_steps = shape[2];
  activations.valid_frames = static_cast<size_t>(std::min<int64_t>(read_length_scalar(encoder_length_tensor), static_cast<int64_t>(activations.time_steps)));

  // Zero-copy: retain tensor and read directly downstream
  activations.tensor = encoder_tensor;
  return activations;
}

struct DecoderResult {
  std::vector<int> tokens;
  std::vector<TokenTiming> timings;
  double t_decoder_ms = 0.0;
  double t_joint_ms = 0.0;
  size_t decoder_runs = 0;
  size_t joint_calls = 0;
};

DecoderResult run_greedy_decoder(OpenVINOParakeet::Impl& impl,
                                 const EncoderActivations& encoder,
                                 const SegmentOptions& options,
                                 DecoderState& state,
                                 bool is_last_chunk) {
  const size_t valid_frames = std::min(encoder.valid_frames, encoder.time_steps);
  if (valid_frames == 0) {
    return {{}, {}};
  }

  // Use blank_id + 1 as the effective vocab size (tokens 0 through blank_id)
  // The vocab JSON may have extra entries beyond blank, but they're not used
  const size_t vocab_size = static_cast<size_t>(impl.runtime_cfg.blank_token_id) + 1;
  const size_t duration_head = impl.runtime_cfg.duration_bins.size();
  if (vocab_size == 0 || vocab_size + duration_head > impl.joint_output_size) {
    throw std::runtime_error("Invalid joint head configuration");
  }

  // Head layout: by default tokens-first then durations. Allow overriding via env.
  bool tokens_first = true;
  if (const char* env_df = std::getenv("EDDY_DURATION_FIRST")) {
    tokens_first = !(std::string(env_df) == "1");
  }
  if (const char* env_layout = std::getenv("EDDY_HEAD_LAYOUT")) {
    std::string v(env_layout);
    for (auto& c : v) c = static_cast<char>(::tolower(c));
    if (v == "durations_first" || v == "duration_first") tokens_first = false;
    else if (v == "tokens_first" || v == "token_first") tokens_first = true;
  }
  const size_t tokens_offset = tokens_first ? 0 : duration_head;
  const size_t durations_offset = tokens_first ? vocab_size : 0;
  if (std::getenv("EDDY_DEBUG")) {
    std::cerr << "[CFG] Head layout: " << (tokens_first ? "tokens-first" : "durations-first")
              << ", tokens_offset=" << tokens_offset
              << ", durations_offset=" << durations_offset << "\n";
  }

  ov::Tensor encoder_step(ov::element::f32, {1, 1, impl.encoder_hidden_size});
  ov::Tensor decoder_step(ov::element::f32, {1, 1, impl.decoder_hidden_size});

  // Initialize or reuse LSTM state
  ov::Tensor hidden_state(ov::element::f32, {2, 1, impl.decoder_hidden_size});
  ov::Tensor cell_state(ov::element::f32, {2, 1, impl.decoder_hidden_size});

  if (state.has_lstm_state) {
    // Continue from previous chunk's LSTM state
    if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Continuing with preserved LSTM state from previous chunk\n";
    std::memcpy(hidden_state.data<float>(), state.hidden_state.data<float>(), hidden_state.get_byte_size());
    std::memcpy(cell_state.data<float>(), state.cell_state.data<float>(), cell_state.get_byte_size());
  } else {
    // First chunk: zero-initialize LSTM state
    if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Starting with fresh LSTM state (first chunk)\n";
    std::fill(hidden_state.data<float>(), hidden_state.data<float>() + hidden_state.get_size(), 0.0F);
    std::fill(cell_state.data<float>(), cell_state.data<float>() + cell_state.get_size(), 0.0F);
  }

  // Use last token from previous chunk if available, otherwise blank
  int starting_token = state.last_token.value_or(impl.runtime_cfg.blank_token_id);
  if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Starting decoder with token: " << starting_token << "\n";

  // Prepare token input tensor with element type matching decoder 'targets' port
  auto targets_port = impl.decoder_model.input("targets");
  const auto targets_et = targets_port.get_element_type();
  ov::Tensor token_input;
  if (targets_et == ov::element::i64) {
    token_input = ov::Tensor(ov::element::i64, {1, 1});
    token_input.data<int64_t>()[0] = static_cast<int64_t>(starting_token);
  } else {  // default to i32
    token_input = ov::Tensor(ov::element::i32, {1, 1});
    token_input.data<int32_t>()[0] = static_cast<int32_t>(starting_token);
  }

  std::vector<int> tokens;
  std::vector<TokenTiming> timings;
  tokens.reserve(options.max_tokens);
  timings.reserve(options.max_tokens);

  size_t frame_index = 0;
  int last_token = starting_token;

  // TDT statistics
  size_t total_joint_calls = 0;
  size_t decoder_runs = 0;
  size_t cache_hits = 0;
  size_t blank_tokens = 0;
  size_t non_blank_tokens = 0;

  // Pre-bind joint inputs once; mutate tensor memory between steps
  auto joint_enc_port = impl.joint_model.input("encoder_outputs");
  auto joint_dec_port = impl.joint_model.input("decoder_outputs");
  ov::Tensor joint_enc_in = impl.joint_request.get_input_tensor(0);
  ov::Tensor joint_dec_in = impl.joint_request.get_input_tensor(1);

  // Confidence computation toggle via env (default off)
  const bool track_confidence = (std::getenv("EDDY_TRACK_CONFIDENCE") && std::string(std::getenv("EDDY_TRACK_CONFIDENCE")) == "1");

  // TDT timings
  double t_decoder_ms = 0.0;
  double t_joint_ms = 0.0;

  // Outer loop: runs decoder, then enters inner loop for blank processing
  while (frame_index < valid_frames && tokens.size() < options.max_tokens) {
    // Check if we can use cached decoder output
    ov::Tensor decoder_output;
    ov::Tensor next_hidden;
    ov::Tensor next_cell;

    if (state.has_cached_output && last_token == state.last_token.value_or(-1)) {
      // CACHE HIT: Reuse cached decoder output - significant speedup!
      cache_hits++;
      std::memcpy(joint_dec_in.data<float>(), state.cached_decoder_output.data<float>(), decoder_step.get_byte_size());
      // Keep existing LSTM state tensors
      next_hidden = hidden_state;
      next_cell = cell_state;
    } else {
      // CACHE MISS: Need to run decoder LSTM
      decoder_runs++;
      if (targets_et == ov::element::i64) {
        token_input.data<int64_t>()[0] = static_cast<int64_t>(last_token);
      } else {
        token_input.data<int32_t>()[0] = static_cast<int32_t>(last_token);
      }

      impl.decoder_request.set_tensor(impl.decoder_model.input("targets"), token_input);
      impl.decoder_request.set_tensor(impl.decoder_model.input("h_in"), hidden_state);
      impl.decoder_request.set_tensor(impl.decoder_model.input("c_in"), cell_state);
      auto td0 = std::chrono::steady_clock::now();
      impl.decoder_request.infer();
      auto td1 = std::chrono::steady_clock::now();
      t_decoder_ms += std::chrono::duration<double, std::milli>(td1 - td0).count();

      decoder_output = impl.decoder_request.get_output_tensor(0);  // decoder_output
      next_hidden = impl.decoder_request.get_output_tensor(1);  // h_out
      next_cell = impl.decoder_request.get_output_tensor(2);  // c_out

      // Store decoder output for potential reuse
      std::memcpy(joint_dec_in.data<float>(), decoder_output.data<float>(), decoder_output.get_byte_size());

      // Cache this decoder output for next iteration
      state.cached_decoder_output = ov::Tensor(ov::element::f32, {1, 1, impl.decoder_hidden_size});
      std::memcpy(state.cached_decoder_output.data<float>(), decoder_output.data<float>(), decoder_output.get_byte_size());
      state.has_cached_output = true;
    }

    // TDT Inner Loop: Process consecutive blank tokens without re-running decoder
    // This is the key optimization - decoder output is intentionally reused
    // because blank tokens (silence) shouldn't change language model context
    bool advance_mask = true;
    while (advance_mask && frame_index < valid_frames && tokens.size() < options.max_tokens) {
      total_joint_calls++;

      // Extract encoder frame for current timestep into contiguous buffer
      {
        const float* enc = encoder.tensor.data<float>();
        float* dst = joint_enc_in.data<float>();
        for (size_t channel = 0; channel < impl.encoder_hidden_size; ++channel) {
          const size_t offset = channel * encoder.time_steps + frame_index;
          dst[channel] = enc[offset];
        }
      }

      // Run joint network with encoder frame + REUSED decoder output
      auto tj0 = std::chrono::steady_clock::now();
      impl.joint_request.infer();
      auto tj1 = std::chrono::steady_clock::now();
      t_joint_ms += std::chrono::duration<double, std::milli>(tj1 - tj0).count();

      const auto logits_tensor = impl.joint_request.get_output_tensor(0);  // logits
      const float* logits = logits_tensor.data<float>();

      // Find best token within token head region
      size_t best_token = 0;
      float best_token_score = logits[tokens_offset + 0];
      for (size_t i = 1; i < vocab_size; ++i) {
        const float score = logits[tokens_offset + i];
        if (score > best_token_score) {
          best_token_score = score;
          best_token = i;
        }
      }

      // Calculate confidence only if requested
      float token_confidence = 0.0F;
      if (track_confidence) {
        float token_sum_exp = 0.0F;
        for (size_t i = 0; i < vocab_size; ++i) {
          token_sum_exp += std::exp(logits[tokens_offset + i]);
        }
        token_confidence = std::exp(best_token_score) / token_sum_exp;
      }

      // Find best duration
      size_t best_duration_idx = 0;
      float best_duration_score = logits[durations_offset + 0];
      for (size_t i = 1; i < duration_head; ++i) {
        const float score = logits[durations_offset + i];
        if (score > best_duration_score) {
          best_duration_score = score;
          best_duration_idx = i;
        }
      }

      int duration = impl.runtime_cfg.duration_bins[best_duration_idx];
      if (duration <= 0) {
        duration = 1;  // Always advance at least 1 frame
      }

      const bool is_blank = static_cast<int>(best_token) == impl.runtime_cfg.blank_token_id;

      if (!is_blank) {
        // Non-blank token: emit it and exit inner loop
        non_blank_tokens++;
        const int token_id = static_cast<int>(best_token);
        tokens.push_back(token_id);

        // Record timing information
        timings.push_back({
          .token_id = token_id,
          .frame_index = frame_index,
          .confidence = token_confidence
        });

        last_token = token_id;
        // Update LSTM state with new token's context
        std::memcpy(hidden_state.data<float>(), next_hidden.data<float>(), next_hidden.get_byte_size());
        std::memcpy(cell_state.data<float>(), next_cell.data<float>(), next_cell.get_byte_size());

        // IMPORTANT: Invalidate cache since token changed - force decoder run next iteration
        state.has_cached_output = false;

        advance_mask = false;  // Exit inner loop - will run decoder again
      } else {
        // Blank token: continue inner loop without updating decoder
        blank_tokens++;
        advance_mask = true;  // Continue inner loop
      }

      // Advance frame index by predicted duration
      frame_index = std::min(frame_index + static_cast<size_t>(duration), valid_frames);
    }
  }

  // Last-chunk finalization: for the final audio chunk only, continue at
  // the last encoder frame until a small blank threshold or max steps.
  // This flushes trailing tokens that need extra predictor iterations.
  if (is_last_chunk) {
    const size_t last_frame = valid_frames > 0 ? (valid_frames - 1) : 0;
    size_t additional_steps = 0;
    size_t consecutive_blanks = 0;
    size_t max_additional_steps = 8;   // reduced for speed (overridable)
    size_t max_consecutive_blanks = 1; // reduced for speed (overridable)
    if (const char* env_steps = std::getenv("EDDY_MAX_ADDITIONAL_STEPS")) {
      try { int v = std::stoi(env_steps); if (v >= 0) max_additional_steps = static_cast<size_t>(v); } catch (...) {}
    }
    if (const char* env_blanks = std::getenv("EDDY_MAX_CONSEC_BLANKS")) {
      try { int v = std::stoi(env_blanks); if (v >= 1) max_consecutive_blanks = static_cast<size_t>(v); } catch (...) {}
    }

    while (additional_steps < max_additional_steps &&
           consecutive_blanks < max_consecutive_blanks &&
           tokens.size() < options.max_tokens) {
      // Prepare decoder output for current last_token
      ov::Tensor decoder_output;
      ov::Tensor next_hidden;
      ov::Tensor next_cell;

      if (state.has_cached_output && last_token == state.last_token.value_or(-1)) {
        // Cache hit across chunk boundary
        cache_hits++;
        std::memcpy(joint_dec_in.data<float>(), state.cached_decoder_output.data<float>(), decoder_step.get_byte_size());
        next_hidden = hidden_state;
        next_cell = cell_state;
      } else {
        // Need to run decoder for the current last_token
        decoder_runs++;
        if (targets_et == ov::element::i64) {
          token_input.data<int64_t>()[0] = static_cast<int64_t>(last_token);
        } else {
          token_input.data<int32_t>()[0] = static_cast<int32_t>(last_token);
        }

        // Use named ports to avoid index/type mismatches
        impl.decoder_request.set_tensor(impl.decoder_model.input("targets"), token_input);
        impl.decoder_request.set_tensor(impl.decoder_model.input("h_in"), hidden_state);
        impl.decoder_request.set_tensor(impl.decoder_model.input("c_in"), cell_state);
        auto td0b = std::chrono::steady_clock::now();
        impl.decoder_request.infer();
        auto td1b = std::chrono::steady_clock::now();
        t_decoder_ms += std::chrono::duration<double, std::milli>(td1b - td0b).count();

        decoder_output = impl.decoder_request.get_output_tensor(0);  // decoder_output
        next_hidden = impl.decoder_request.get_output_tensor(1);      // h_out
        next_cell = impl.decoder_request.get_output_tensor(2);        // c_out

        // Store decoder output for potential reuse
        std::memcpy(joint_dec_in.data<float>(), decoder_output.data<float>(), decoder_output.get_byte_size());
        state.cached_decoder_output = ov::Tensor(ov::element::f32, {1, 1, impl.decoder_hidden_size});
        std::memcpy(state.cached_decoder_output.data<float>(), decoder_output.data<float>(), decoder_output.get_byte_size());
        state.has_cached_output = true;
      }

      // Run joint with the LAST encoder frame and current decoder output
      total_joint_calls++;
      {
        const float* enc = encoder.tensor.data<float>();
        float* dst = joint_enc_in.data<float>();
        for (size_t channel = 0; channel < impl.encoder_hidden_size; ++channel) {
          const size_t off = channel * encoder.time_steps + last_frame;
          dst[channel] = enc[off];
        }
      }
      auto tj0b = std::chrono::steady_clock::now();
      impl.joint_request.infer();
      auto tj1b = std::chrono::steady_clock::now();
      t_joint_ms += std::chrono::duration<double, std::milli>(tj1b - tj0b).count();

      const auto logits_tensor = impl.joint_request.get_output_tensor(0);
      const float* logits = logits_tensor.data<float>();

      // Best token in token head region (finalization loop)
      size_t best_token = 0;
      float best_token_score = logits[tokens_offset + 0];
      for (size_t i = 1; i < vocab_size; ++i) {
        const float score = logits[tokens_offset + i];
        if (score > best_token_score) {
          best_token_score = score;
          best_token = i;
        }
      }

      // Confidence over token head
      float token_sum_exp = 0.0F;
      for (size_t i = 0; i < vocab_size; ++i) token_sum_exp += std::exp(logits[tokens_offset + i]);
      float token_confidence = std::exp(best_token_score) / token_sum_exp;

      const bool is_blank = static_cast<int>(best_token) == impl.runtime_cfg.blank_token_id;
      if (!is_blank) {
        // Emit token at last frame index
        non_blank_tokens++;
        const int token_id = static_cast<int>(best_token);
        tokens.push_back(token_id);
        timings.push_back({
            .token_id = token_id,
            .frame_index = last_frame,
            .confidence = token_confidence});

        // Update LSTM with this token's context for next step
        std::memcpy(hidden_state.data<float>(), next_hidden.data<float>(), next_hidden.get_byte_size());
        std::memcpy(cell_state.data<float>(), next_cell.data<float>(), next_cell.get_byte_size());

        last_token = token_id;
        state.has_cached_output = false;  // Force decoder run for new token on next step
        consecutive_blanks = 0;
      } else {
        // Count consecutive blanks; do not update LSTM
        blank_tokens++;
        consecutive_blanks++;
      }

      additional_steps++;
    }
  }

  // Per-file TDT stats removed (keep benchmark end summary only)

  // Save final LSTM state and last token for next chunk
  // Always allocate fresh tensors for state (OpenVINO tensors can't be default-constructed safely)
  state.hidden_state = ov::Tensor(ov::element::f32, {2, 1, impl.decoder_hidden_size});
  state.cell_state = ov::Tensor(ov::element::f32, {2, 1, impl.decoder_hidden_size});

  std::memcpy(state.hidden_state.data<float>(), hidden_state.data<float>(), hidden_state.get_byte_size());
  std::memcpy(state.cell_state.data<float>(), cell_state.data<float>(), cell_state.get_byte_size());
  state.has_lstm_state = true;

  if (!tokens.empty()) {
    state.last_token = tokens.back();
    if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Saved final LSTM state and last token: " << *state.last_token << "\n";

    // Clear cache after punctuation tokens to prevent duplicates at chunk boundaries
    if (impl.tokenizer.is_punctuation(*state.last_token)) {
      state.has_cached_output = false;
      if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Cleared decoder cache after punctuation token\n";
    }
  } else {
    state.last_token = starting_token;
    if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] No tokens emitted, keeping starting token: " << starting_token << "\n";
  }

  DecoderResult out;
  out.tokens = std::move(tokens);
  out.timings = std::move(timings);
  out.t_decoder_ms = t_decoder_ms;
  out.t_joint_ms = t_joint_ms;
  out.decoder_runs = decoder_runs;
  out.joint_calls = total_joint_calls;
  return out;
}

}  // namespace

InferenceResult OpenVINOParakeet::infer(const AudioSegment& segment, const SegmentOptions& options) {
  ensure_compiled_model();

  const auto start = std::chrono::steady_clock::now();
  double t_preproc_ms = 0.0;
  double t_encoder_ms = 0.0;
  double t_decoder_ms = 0.0;
  double t_joint_ms = 0.0;

  std::vector<int> token_ids;
  std::vector<TokenTiming> all_timings;
  std::vector<size_t> chunk_sizes_frames;  // for result metadata
  std::vector<InferenceResult::ChunkInfo> chunk_logs;
  {
    std::lock_guard<std::mutex> lock(impl_->request_guard);
    auto t0 = std::chrono::steady_clock::now();
    const auto mel = run_preprocessor(*impl_, segment);
    auto t1 = std::chrono::steady_clock::now();
    t_preproc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Check if audio is longer than encoder can handle
    const size_t max_frames = impl_->encoder_expected_frames;

    if (mel.frames <= max_frames) {
      // Short audio - process normally
      DecoderState state;  // Fresh state for single-chunk audio
      t0 = std::chrono::steady_clock::now();
      const auto encoder = run_encoder(*impl_, mel);
      t1 = std::chrono::steady_clock::now();
      const double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      t_encoder_ms += enc_ms;
      t0 = std::chrono::steady_clock::now();
  auto result = run_greedy_decoder(*impl_, encoder, options, state, /*is_last_chunk=*/true);
      t1 = std::chrono::steady_clock::now();
      t_decoder_ms += result.t_decoder_ms;
      t_joint_ms += result.t_joint_ms;
      token_ids = std::move(result.tokens);
      all_timings = std::move(result.timings);
      // Chunk metadata: single chunk of full valid frames
      chunk_sizes_frames.push_back(mel.frames);
      // Single chunk: we omit per-chunk log to keep JSON compact (only include when >1)
    } else {
      // Long audio - process in overlapping chunks
      if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Audio too long (" << mel.frames << " frames), processing in chunks\n";

      // Use overlap to avoid losing audio at boundaries
      // FluidAudio uses ~20 frames (~1.6s) when max window ~180 frames (~14.4s)
      // Choose a proportional overlap (~1/9 of max_frames) with sane bounds
      const size_t overlap_frames = std::min<size_t>(
          max_frames > 1 ? max_frames - 1 : 1,
          std::max<size_t>(4, max_frames / 9));
      const size_t stride = max_frames - overlap_frames;

      // Persistent decoder state across chunks for full state continuity
      DecoderState decoder_state;

      size_t offset = 0;
      size_t chunk_idx = 0;
      // Track the last emitted token's global encoder frame index to enforce monotonic time
      size_t last_emitted_global_frame = 0;
      bool have_last_emitted_frame = false;

      while (offset < mel.frames) {
        // Extract chunk with overlap
        const size_t chunk_size = std::min(max_frames, mel.frames - offset);
        const bool is_last_chunk = (offset + chunk_size >= mel.frames);
        // Track size for result metadata
        chunk_sizes_frames.push_back(chunk_size);

        MelFeatures chunk;
        chunk.frames = chunk_size;
        chunk.data.resize(128 * chunk_size);

        // Copy mel data for this chunk (mel is stored as [mel_bins][time])
        for (size_t bin = 0; bin < 128; ++bin) {
          const float* src = mel.data.data() + bin * mel.frames + offset;
          float* dst = chunk.data.data() + bin * chunk_size;
          std::copy(src, src + chunk_size, dst);
        }

        if (std::getenv("EDDY_DEBUG")) {
          std::cerr << "[INFO] Processing chunk " << chunk_idx
                    << " at offset " << offset
                    << " (size: " << chunk_size << " frames)\n";
        }

        t0 = std::chrono::steady_clock::now();
        const auto encoder = run_encoder(*impl_, chunk);
        t1 = std::chrono::steady_clock::now();
        const double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        t_encoder_ms += enc_ms;
        // Pass persistent decoder_state - it will be updated with final state after decoding
        t0 = std::chrono::steady_clock::now();
        auto decoder_result = run_greedy_decoder(*impl_, encoder, options, decoder_state, is_last_chunk);
        t1 = std::chrono::steady_clock::now();
        t_decoder_ms += decoder_result.t_decoder_ms;
        t_joint_ms += decoder_result.t_joint_ms;
        auto& chunk_tokens = decoder_result.tokens;
        auto& chunk_timings = decoder_result.timings;

        InferenceResult::ChunkInfo ci;
        ci.index = chunk_idx;
        ci.offset_frames = offset;
        ci.size_frames = chunk_size;
        ci.is_last = is_last_chunk;
        ci.tokens_predicted = chunk_tokens.size();

        if (chunk_idx == 0) {
          // First chunk: keep all tokens and timings
          token_ids = std::move(chunk_tokens);
          // Convert timings to global frame indices (offset = 0 for first chunk)
          for (auto& t : chunk_timings) {
            t.frame_index += offset;
          }
          all_timings = std::move(chunk_timings);
          if (!all_timings.empty()) {
            last_emitted_global_frame = all_timings.back().frame_index;
            have_last_emitted_frame = true;
          }
          ci.tokens_appended = token_ids.size();
          ci.skip_prefix = 0;
          ci.holdback = 0;
          // Appended text is the entire first chunk
          ci.appended_text = impl_->tokenizer.decode(token_ids);
        } else {
          // Subsequent chunks: use deduplication as safety net (FluidAudio approach)
          // Even with LSTM state continuity, decoder state reset at chunk boundaries
          // can cause some duplicates, so we apply deduplication as a safety net
          size_t skip_count = 0;

          // Convert timings to global frame indices by adding the current chunk offset
          for (auto& t : chunk_timings) {
            t.frame_index += offset;
          }

          // Global time-gate: ensure strictly increasing global frame indices
          if (have_last_emitted_frame && !chunk_timings.empty()) {
            size_t time_gate_idx = 0;
            while (time_gate_idx < chunk_timings.size() &&
                   chunk_timings[time_gate_idx].frame_index <= last_emitted_global_frame) {
              ++time_gate_idx;
            }
            if (time_gate_idx > 0) {
              if (std::getenv("EDDY_DEBUG")) {
                std::cerr << "[INFO] Time-gate skipped " << time_gate_idx
                          << " tokens to enforce monotonic global timing\n";
              }
              skip_count = std::max(skip_count, time_gate_idx);
            }
          }

          if (std::getenv("EDDY_DEBUG")) {
            std::cerr << "[DEBUG] Last 10 tokens of prev chunk: ";
            for (size_t i = std::max(size_t(0), token_ids.size() - 10); i < token_ids.size(); ++i) {
              std::cerr << token_ids[i] << " ";
            }
            std::cerr << "\n[DEBUG] First 10 tokens of curr chunk: ";
            for (size_t i = 0; i < std::min(size_t(10), chunk_tokens.size()); ++i) {
              std::cerr << chunk_tokens[i] << " ";
            }
            std::cerr << "\n";
          }

          // FluidAudio-style deduplication:
          // 1) Punctuation guard
          // 2) Exact suffix-prefix overlap (longest-first)
          // 3) Boundary-limited partial overlap search within the beginning of the current chunk

          // 1) Punctuation guard: if previous tail token equals current head and is punctuation, drop the head
          size_t punctuation_removed = 0;
          if (!token_ids.empty() && !chunk_tokens.empty()) {
            int last_prev = token_ids.back();
            int first_curr = chunk_tokens.front();
            bool is_punc = false;
            try {
              is_punc = impl_->tokenizer.is_punctuation(first_curr);
            } catch (...) {
              is_punc = false;
            }
            if (last_prev == first_curr && is_punc) {
              punctuation_removed = 1;
              if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Dropping duplicate leading punctuation at chunk start\n";
            }
          }

          // Parameters (tunable via env):
          size_t boundary_search_frames = 20;  // ~1.6s at 12.5 fps
          if (const char* env_b = std::getenv("EDDY_BOUNDARY_SEARCH_FRAMES")) {
            try { int v = std::stoi(env_b); if (v > 0) boundary_search_frames = static_cast<size_t>(v); } catch (...) {}
          }
          size_t max_overlap_tokens = 30;      // how many tokens to consider for overlap
          if (const char* env_o = std::getenv("EDDY_MAX_OVERLAP_TOKENS")) {
            try { int v = std::stoi(env_o); if (v > 0) max_overlap_tokens = static_cast<size_t>(v); } catch (...) {}
          }

          // 2) Exact suffix-prefix match (longest-first)
          size_t prev_tail_window = std::min<size_t>(15, token_ids.size());
          size_t working_curr_count = chunk_tokens.size() >= punctuation_removed ? (chunk_tokens.size() - punctuation_removed) : 0;
          size_t max_match_len = std::min({prev_tail_window, max_overlap_tokens, working_curr_count});

          size_t best_exact_overlap = 0;
          for (size_t overlap_len = max_match_len; overlap_len >= 2 && overlap_len <= max_match_len; --overlap_len) {
            bool match = true;
            const size_t prev_start = token_ids.size() - overlap_len;
            for (size_t i = 0; i < overlap_len; ++i) {
              if (token_ids[prev_start + i] != chunk_tokens[punctuation_removed + i]) {
                match = false;
                break;
              }
            }
            if (match) {
              best_exact_overlap = overlap_len;
              if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Exact suffix-prefix overlap length " << overlap_len << "\n";
              break;
            }
            if (overlap_len == 2) break;  // prevent size_t underflow
          }

          size_t dedup_skip = punctuation_removed;
          if (best_exact_overlap > 0) {
            dedup_skip += best_exact_overlap;
          } else {
            // 3) Boundary-limited partial overlap search
            // Limit currentStart by frames within the right-context boundary window
            size_t search_limit_tokens = 0;
            if (!chunk_timings.empty()) {
              for (size_t idx = 0; idx < chunk_timings.size(); ++idx) {
                const size_t local_frame = chunk_timings[idx].frame_index - offset;  // convert to local
                if (local_frame >= boundary_search_frames) break;
                // Only count tokens after punctuation_removed
                if (idx >= punctuation_removed) search_limit_tokens = (idx - punctuation_removed) + 1;
              }
            } else {
              // Fallback: use token count if timings absent
              search_limit_tokens = std::min(working_curr_count, boundary_search_frames);
            }

            // Search for any prev subsequence against early part of current
            size_t effective_prev_tail = std::min<size_t>(15, token_ids.size());
            for (size_t overlap_len = std::min({effective_prev_tail, max_overlap_tokens, working_curr_count});
                 overlap_len >= 2; --overlap_len) {
              const size_t prev_start_min = token_ids.size() > effective_prev_tail ? (token_ids.size() - effective_prev_tail) : 0;
              const size_t prev_end = token_ids.size() >= overlap_len ? (token_ids.size() - overlap_len + 1) : 0;
              if (prev_end <= prev_start_min) continue;

              for (size_t prev_start = prev_start_min; prev_start < prev_end; ++prev_start) {
                // Iterate currentStart within boundary window
                const size_t curr_end_limit = (working_curr_count >= overlap_len) ? (working_curr_count - overlap_len + 1) : 0;
                const size_t search_limit = std::min(search_limit_tokens, curr_end_limit);
                for (size_t curr_off = 0; curr_off < search_limit; ++curr_off) {
                  bool eq = true;
                  for (size_t k = 0; k < overlap_len; ++k) {
                    if (token_ids[prev_start + k] != chunk_tokens[punctuation_removed + curr_off + k]) { eq = false; break; }
                  }
                  if (eq) {
                    dedup_skip = std::max(dedup_skip, punctuation_removed + curr_off + overlap_len);
                    if (std::getenv("EDDY_DEBUG")) {
                      std::cerr << "[INFO] Boundary duplicate seq len=" << overlap_len
                                << ", curr_off=" << curr_off << ", prev_start=" << prev_start << "\n";
                    }
                    // Break out of loops in order: curr_off, prev_start, overlap_len
                    curr_off = search_limit;  // force exit
                    prev_start = prev_end;    // force exit
                    overlap_len = 2;          // minimal to exit outer loop
                    break;
                  }
                }
              }
              if (overlap_len == 2) break;  // prevent underflow
            }
          }

          if (dedup_skip > 0) {
            skip_count = std::max(skip_count, dedup_skip);
          }

          // Right-context holdback: for non-final chunks, optionally hold back tokens
          // near the end so that the next chunk (with more right context) can decide.
          size_t emit_end = chunk_tokens.size();
          if (!is_last_chunk && !chunk_timings.empty()) {
            bool disable_holdback = false;
            if (const char* env_hb = std::getenv("EDDY_DISABLE_HOLDBACK")) {
              disable_holdback = std::string(env_hb) == "1";
            }
            size_t right_context_frames = std::min(overlap_frames, chunk_size);
            if (const char* env_hbf = std::getenv("EDDY_HOLDBACK_FRAMES")) {
              try {
                int v = std::stoi(env_hbf);
                if (v <= 0) {
                  disable_holdback = true;
                } else {
                  right_context_frames = std::min<size_t>(static_cast<size_t>(v), chunk_size);
                }
              } catch (...) {
                // ignore and use default
              }
            }
            if (!disable_holdback && right_context_frames > 0) {
            size_t holdback_start = chunk_tokens.size();
            for (size_t idx = 0; idx < chunk_timings.size(); ++idx) {
              const size_t local_frame = chunk_timings[idx].frame_index - offset;  // convert back to local
              if (local_frame >= (chunk_size > right_context_frames ? (chunk_size - right_context_frames) : 0)) {
                holdback_start = idx;
                break;
              }
            }
            if (holdback_start < chunk_tokens.size()) {
              emit_end = std::min(emit_end, holdback_start);
              if (std::getenv("EDDY_DEBUG")) {
                std::cerr << "[INFO] Holding back " << (chunk_tokens.size() - holdback_start)
                          << " tokens for right-context lookahead\n";
              }
            }
            }
          }

          // Append tokens and timings, skipping duplicates and any filtered prefix/suffix
          if (skip_count >= emit_end) {
            if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Entire chunk consists of overlapped region; appending nothing\n";
          } else {
            token_ids.insert(token_ids.end(), chunk_tokens.begin() + skip_count, chunk_tokens.begin() + emit_end);
            all_timings.insert(all_timings.end(), chunk_timings.begin() + skip_count, chunk_timings.begin() + emit_end);

            if (!all_timings.empty()) {
              last_emitted_global_frame = all_timings.back().frame_index;
              have_last_emitted_frame = true;
            }
          }

          // Record dedup/holdback stats
          const size_t total_curr = chunk_tokens.size();
          const size_t appended = (skip_count >= emit_end) ? 0 : (emit_end - skip_count);
          const size_t held_back = (emit_end <= total_curr) ? (total_curr - emit_end) : 0;
          ci.tokens_appended = appended;
          ci.skip_prefix = skip_count;
          ci.holdback = held_back;
          // Decode the appended slice as the chunk's individual output
          if (appended > 0) {
            std::vector<int> appended_tokens(chunk_tokens.begin() + skip_count,
                                            chunk_tokens.begin() + emit_end);
            ci.appended_text = impl_->tokenizer.decode(appended_tokens);
          } else {
            ci.appended_text.clear();
          }
        }

        chunk_logs.push_back(ci);
        chunk_idx++;

        // Check if we're done
        if (offset + chunk_size >= mel.frames) {
          break;  // Processed all frames
        }

        // Move forward by stride (not full chunk size) to create overlap
        offset += stride;
      }
    }
  }

  const auto end = std::chrono::steady_clock::now();
  const auto latency_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  const double known_ms = t_preproc_ms + t_encoder_ms + t_decoder_ms + t_joint_ms;
  const double other_ms = latency_ms > known_ms ? (latency_ms - known_ms) : 0.0;

  InferenceResult result;
  result.token_ids = std::move(token_ids);
  result.text = impl_->tokenizer.decode(result.token_ids);
  result.latency_ms = latency_ms;
  result.token_timings = std::move(all_timings);
  result.chunk_sizes_frames = std::move(chunk_sizes_frames);
  result.chunks = std::move(chunk_logs);

  // Calculate overall confidence (average of token confidences)
  if (!result.token_timings.empty()) {
    float sum = 0.0F;
    for (const auto& timing : result.token_timings) {
      sum += timing.confidence;
    }
    result.overall_confidence = sum / static_cast<float>(result.token_timings.size());
  } else {
    result.overall_confidence = 0.1F;  // FluidAudio default for empty transcription
  }

  // Per-file profile removed (keep benchmark end summary only)

  return result;
}

void OpenVINOParakeet::warmup() {
  ensure_compiled_model();
}

}  // namespace eddy::parakeet
