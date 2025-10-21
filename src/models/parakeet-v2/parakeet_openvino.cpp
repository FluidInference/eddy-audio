#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "parakeet_openvino_impl.hpp"
#include "eddy/models/parakeet-v2/parakeet_preprocessor.hpp"
#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"
#include "eddy/models/parakeet-v2/parakeet_decoder.hpp"
#include "eddy/models/parakeet-v2/parakeet_chunking.hpp"

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
#include "eddy/utils/tokenizer.hpp"

namespace eddy::parakeet {

namespace {

ov::AnyMap make_compile_cfg_from_env() {
  ov::AnyMap cfg;
  if (const char* perf = std::getenv("EDDY_OV_PERF")) {
    std::string v(perf);
    for (auto& c : v) c = static_cast<char>(::toupper(c));
    if (v == "LATENCY") cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    else if (v == "THROUGHPUT") cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
  }
  if (const char* nr = std::getenv("EDDY_OV_NUM_REQUESTS")) {
    try {
      int n = std::max(1, std::stoi(nr));
      cfg[ov::hint::num_requests.name()] = n;
    } catch (const std::exception& e) {
      std::cerr << "[WARN] Invalid EDDY_OV_NUM_REQUESTS value '" << nr << "', using default\n";
    }
  }
  if (const char* th = std::getenv("EDDY_OV_THREADS")) {
    try {
      int n = std::max(1, std::stoi(th));
      cfg[ov::inference_num_threads.name()] = n;
    } catch (const std::exception& e) {
      std::cerr << "[WARN] Invalid EDDY_OV_THREADS value '" << th << "', using default\n";
    }
  }
  // Removed precision hint: rely on device defaults and model precision.
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

struct ChunkPipelineResult {
  std::vector<int> tokens;
  std::vector<TokenTiming> timings;
  double encoder_ms;
  double decoder_ms;
  double joint_ms;
};

// Helper function to run encoder + decoder pipeline on a chunk
// Note: decoder_state is passed by reference and will be updated with LSTM state
// for continuity across chunks
ChunkPipelineResult run_chunk_pipeline(
    OpenVINOParakeet::Impl& impl,
    const MelFeatures& mel,
    const SegmentOptions& options,
    DecoderState& decoder_state,
    bool is_last_chunk) {

  auto t0 = std::chrono::steady_clock::now();
  const auto encoder = run_encoder(impl, mel);
  auto t1 = std::chrono::steady_clock::now();
  const double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  t0 = std::chrono::steady_clock::now();
  auto decoder_result = run_decoder(impl, encoder, options, decoder_state, is_last_chunk);
  t1 = std::chrono::steady_clock::now();

  ChunkPipelineResult result;
  result.tokens = std::move(decoder_result.tokens);
  result.timings = std::move(decoder_result.timings);
  result.encoder_ms = enc_ms;
  result.decoder_ms = decoder_result.t_decoder_ms;
  result.joint_ms = decoder_result.t_joint_ms;

  return result;
}

// Extract a chunk of mel features from the full mel spectrogram
MelFeatures extract_mel_chunk(const MelFeatures& full_mel, size_t offset, size_t chunk_size) {
  MelFeatures chunk;
  chunk.frames = chunk_size;
  chunk.data.resize(128 * chunk_size);

  // Copy mel data for this chunk (mel is stored as [mel_bins][time])
  for (size_t bin = 0; bin < 128; ++bin) {
    const float* src = full_mel.data.data() + bin * full_mel.frames + offset;
    float* dst = chunk.data.data() + bin * chunk_size;
    std::copy(src, src + chunk_size, dst);
  }

  return chunk;
}

// Process the first chunk - keep all tokens without deduplication
void process_first_chunk(
    OpenVINOParakeet::Impl& impl,
    std::vector<int>& chunk_tokens,
    std::vector<TokenTiming>& chunk_timings,
    size_t offset,
    std::vector<int>& token_ids,
    std::vector<TokenTiming>& all_timings,
    size_t& last_emitted_global_frame,
    bool& have_last_emitted_frame,
    InferenceResult::ChunkInfo& ci) {

  token_ids = std::move(chunk_tokens);

  // Convert timings to global frame indices
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
  ci.appended_text = impl.tokenizer.decode(token_ids);
}

// Process subsequent chunks with deduplication
void process_subsequent_chunk(
    OpenVINOParakeet::Impl& impl,
    std::vector<int>& chunk_tokens,
    std::vector<TokenTiming>& chunk_timings,
    size_t offset,
    size_t chunk_size,
    size_t overlap_frames,
    bool is_last_chunk,
    std::vector<int>& token_ids,
    std::vector<TokenTiming>& all_timings,
    size_t& last_emitted_global_frame,
    bool& have_last_emitted_frame,
    InferenceResult::ChunkInfo& ci) {

  auto dedup_result = deduplicate_chunk(
      impl,
      token_ids,
      chunk_tokens,
      chunk_timings,
      offset,
      chunk_size,
      overlap_frames,
      is_last_chunk,
      last_emitted_global_frame,
      have_last_emitted_frame
  );

  const size_t skip_count = dedup_result.skip_prefix;
  const size_t emit_end = dedup_result.emit_end;

  // Append tokens and timings, skipping duplicates and any filtered prefix/suffix
  if (skip_count >= emit_end) {
    if (std::getenv("EDDY_DEBUG")) {
      std::cerr << "[INFO] Entire chunk consists of overlapped region; appending nothing\n";
    }
  } else {
    const size_t append_count = emit_end - skip_count;

    if (token_ids.capacity() < token_ids.size() + append_count) {
      token_ids.reserve(token_ids.size() + append_count + 64);
    }
    token_ids.insert(token_ids.end(), chunk_tokens.begin() + skip_count, chunk_tokens.begin() + emit_end);

    if (all_timings.capacity() < all_timings.size() + append_count) {
      all_timings.reserve(all_timings.size() + append_count + 64);
    }
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

  if (appended > 0) {
    ci.appended_text = impl.tokenizer.decode_span(chunk_tokens.data() + skip_count, appended);
  } else {
    ci.appended_text.clear();
  }
}


}  // namespace

std::shared_ptr<OpenVINOParakeet> make_openvino_parakeet(std::shared_ptr<eddy::OpenVINOBackend> backend,
                                                        ModelPaths model_paths,
                                                        RuntimeConfig runtime_cfg) {
  return std::make_shared<OpenVINOParakeet>(std::move(backend), std::move(model_paths), std::move(runtime_cfg));
}

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
    // ========================================
    // Device configuration
    // ========================================
    auto& core = impl_->backend->core();
    const std::string device = impl_->runtime_cfg.device.empty() ? "AUTO" : impl_->runtime_cfg.device;
    const bool target_npu = (device == "NPU");

    // ========================================
    // Compile preprocessor (mel spectrogram)
    // ========================================
    // Melspectogram has to run on CPU
    std::string preproc_device = std::string("CPU");
    if (const char* env_pre = std::getenv("EDDY_PREPROC_DEVICE")) {
      if (*env_pre) preproc_device = env_pre;
    }

    impl_->preproc_model = compile_component(core, impl_->model_paths.preprocessor, preproc_device);
    impl_->preproc_request = impl_->preproc_model.create_infer_request();

    // ========================================
    // Compile encoder, decoder, and joint models
    // ========================================
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

    // Encoder
    impl_->encoder_model = compile_with_fallback(impl_->model_paths.encoder, "encoder");
    impl_->encoder_request = impl_->encoder_model.create_infer_request();

    // Decoder
    impl_->decoder_model = compile_with_fallback(impl_->model_paths.decoder, "decoder");
    impl_->decoder_request = impl_->decoder_model.create_infer_request();

    // Joint
    impl_->joint_model = compile_with_fallback(impl_->model_paths.joint, "joint");
    impl_->joint_request = impl_->joint_model.create_infer_request();

    // ========================================
    // Load tokenizer
    // ========================================
    impl_->tokenizer.load(impl_->model_paths.tokenizer_json, impl_->runtime_cfg.blank_token_id);

    // ========================================
    // Load metadata (optional port name overrides)
    // ========================================
    try {
      const std::filesystem::path enc_path = impl_->model_paths.encoder.path;
      const std::filesystem::path model_dir = std::filesystem::path(enc_path).parent_path();
      const std::filesystem::path meta_path = model_dir / "parakeet_metadata.json";

      if (std::filesystem::exists(meta_path)) {
        std::ifstream meta_stream(meta_path);
        nlohmann::json meta_json;
        meta_stream >> meta_json;

        if (meta_json.contains("encoder")) {
          const auto& enc = meta_json["encoder"];
          if (enc.contains("mel_input")) impl_->enc_mel_name = enc["mel_input"].get<std::string>();
          if (enc.contains("length_input")) impl_->enc_len_name = enc["length_input"].get<std::string>();
          if (enc.contains("output")) impl_->enc_out_name = enc["output"].get<std::string>();
          if (enc.contains("length_output")) impl_->enc_len_out_name = enc["length_output"].get<std::string>();
        }
      }
    } catch (...) {
      // Ignore metadata errors and continue with defaults
    }

    // ========================================
    // Resolve encoder ports by name
    // ========================================
    try {
      impl_->encoder_ports.mel_in = impl_->encoder_model.input(impl_->enc_mel_name);
    } catch (...) {
      throw std::runtime_error(std::string("Missing encoder mel input port: ") + impl_->enc_mel_name);
    }

    try {
      impl_->encoder_ports.len_in = impl_->encoder_model.input(impl_->enc_len_name);
    } catch (...) {
      throw std::runtime_error(std::string("Missing encoder length input port: ") + impl_->enc_len_name);
    }

    try {
      impl_->encoder_ports.enc_out = impl_->encoder_model.output(impl_->enc_out_name);
    } catch (...) {
      throw std::runtime_error(std::string("Missing encoder output port: ") + impl_->enc_out_name);
    }

    impl_->encoder_expected_frames = required_length(impl_->encoder_ports.mel_in.value());

    // ========================================
    // Determine encoder output indices
    // ========================================
    const auto outs = impl_->encoder_model.outputs();
    bool found_output = false;
    bool found_len = false;

    for (size_t i = 0; i < outs.size(); ++i) {
      const auto& p = outs[i];

      // Check for main output port
      try {
        const auto named_out = impl_->encoder_model.output(impl_->enc_out_name);
        if (p == named_out) {
          impl_->encoder_output_index = i;

          const auto shape = p.get_shape();
          if (shape.size() >= 2 && shape[1] == 0) {
            throw std::runtime_error("Encoder hidden size is zero");
          }
          if (shape.size() >= 2) {
            impl_->encoder_hidden_size = shape[1];
          }

          found_output = true;
        }
      } catch (...) {}

      // Check for length output port
      if (!found_len) {
        try {
          const auto named_len = impl_->encoder_model.output(impl_->enc_len_out_name);
          if (p == named_len) {
            impl_->encoder_length_index = i;
            found_len = true;
          }
        } catch (...) {}
      }
    }

    if (!found_output) {
      throw std::runtime_error(std::string("Failed to locate encoder output: ") + impl_->enc_out_name);
    }
    if (!found_len) {
      throw std::runtime_error(std::string("Failed to locate encoder length output: ") + impl_->enc_len_out_name);
    }

    // ========================================
    // Extract decoder hidden size
    // ========================================
    const auto decoder_state_shape = impl_->decoder_model.input("h_in").get_shape();
    if (decoder_state_shape.size() != 3) {
      throw std::runtime_error("Unexpected decoder hidden state shape");
    }
    impl_->decoder_hidden_size = decoder_state_shape[2];

    // ========================================
    // Extract joint output size and validate
    // ========================================
    const auto joint_shape = impl_->joint_model.output("logits").get_shape();
    if (joint_shape.empty()) {
      throw std::runtime_error("Joint model logits tensor has no dimensions");
    }
    impl_->joint_output_size = joint_shape.back();

    // Validate joint output size matches vocab + duration bins
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
      // Short audio - process in single chunk
      DecoderState state;
      auto result = run_chunk_pipeline(*impl_, mel, options, state, /*is_last_chunk=*/true);

      t_encoder_ms += result.encoder_ms;
      t_decoder_ms += result.decoder_ms;
      t_joint_ms += result.joint_ms;
      token_ids = std::move(result.tokens);
      all_timings = std::move(result.timings);

      chunk_sizes_frames.push_back(mel.frames);
    } else {
      // Long audio - process in overlapping chunks
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
        const size_t chunk_size = std::min(max_frames, mel.frames - offset);
        const bool is_last_chunk = (offset + chunk_size >= mel.frames);
        chunk_sizes_frames.push_back(chunk_size);

        // Extract mel chunk
        MelFeatures chunk = extract_mel_chunk(mel, offset, chunk_size);

        // Run pipeline on chunk
        auto pipeline_result = run_chunk_pipeline(*impl_, chunk, options, decoder_state, is_last_chunk);
        t_encoder_ms += pipeline_result.encoder_ms;
        t_decoder_ms += pipeline_result.decoder_ms;
        t_joint_ms += pipeline_result.joint_ms;

        // Prepare chunk info
        InferenceResult::ChunkInfo ci;
        ci.index = chunk_idx;
        ci.offset_frames = offset;
        ci.size_frames = chunk_size;
        ci.is_last = is_last_chunk;
        ci.tokens_predicted = pipeline_result.tokens.size();

        // Process chunk based on whether it's first or subsequent
        if (chunk_idx == 0) {
          process_first_chunk(*impl_, pipeline_result.tokens, pipeline_result.timings, offset,
                              token_ids, all_timings, last_emitted_global_frame,
                              have_last_emitted_frame, ci);
        } else {
          process_subsequent_chunk(*impl_, pipeline_result.tokens, pipeline_result.timings,
                                   offset, chunk_size, overlap_frames, is_last_chunk,
                                   token_ids, all_timings, last_emitted_global_frame,
                                   have_last_emitted_frame, ci);
        }

        chunk_logs.push_back(ci);
        chunk_idx++;

        if (offset + chunk_size >= mel.frames) {
          break;
        }

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

