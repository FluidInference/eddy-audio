#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "eddy/models/parakeet-v2/detail/parakeet_impl.hpp"
#include "eddy/models/parakeet-v2/parakeet_preprocessor.hpp"
#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"
#include "eddy/models/parakeet-v2/parakeet_decoder.hpp"
#include "eddy/models/parakeet-v2/parakeet_chunking.hpp"
#include "eddy/utils/openvino_utils.hpp"

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
#include "eddy/models/parakeet-v2/tokenizer.hpp"

namespace eddy::parakeet {

namespace {

constexpr size_t MEL_BINS = 128;  // Standard mel-spectrogram bin count
constexpr size_t VECTOR_RESERVE_SLACK = 64;  // Extra capacity when reserving vector space

// Whether debug logging is enabled (cached for performance)
bool is_debug_enabled() {
  static bool cached = (std::getenv("EDDY_DEBUG") != nullptr);
  return cached;
}

struct ChunkPipelineResult {
  std::vector<int> tokens;
  std::vector<TokenTiming> timings;
  double encoder_ms;
  double decoder_ms;
  double joint_ms;
};

struct MelProcessingResult {
  std::vector<int> token_ids;
  std::vector<TokenTiming> all_timings;
  std::vector<size_t> chunk_sizes_frames;
  std::vector<InferenceResult::ChunkInfo> chunk_logs;
  double encoder_ms;
  double decoder_ms;
  double joint_ms;
};

// Helper function to run encoder + decoder pipeline on a chunk
// Note: decoder_state is passed by reference and will be updated with LSTM state
// for continuity across chunks
ChunkPipelineResult run_chunk_pipeline(
    ParakeetImpl& impl,
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
  // Validate bounds
  if (offset + chunk_size > full_mel.frames) {
    throw std::runtime_error("Chunk extends beyond mel frames: offset=" +
                             std::to_string(offset) + " + chunk_size=" +
                             std::to_string(chunk_size) + " > frames=" +
                             std::to_string(full_mel.frames));
  }

  // Validate data buffer size
  const size_t required_size = MEL_BINS * full_mel.frames;
  if (full_mel.data.size() < required_size) {
    throw std::runtime_error("Mel data buffer too small: expected at least " +
                             std::to_string(required_size) + " elements, got " +
                             std::to_string(full_mel.data.size()));
  }

  MelFeatures chunk;
  chunk.frames = chunk_size;
  chunk.data.resize(MEL_BINS * chunk_size);

  // Copy mel data for this chunk (mel is stored as [mel_bins][time])
  for (size_t bin = 0; bin < MEL_BINS; ++bin) {
    const float* src = full_mel.data.data() + bin * full_mel.frames + offset;
    float* dst = chunk.data.data() + bin * chunk_size;
    std::copy(src, src + chunk_size, dst);
  }

  return chunk;
}

// Process the first chunk - keep all tokens without deduplication
void process_first_chunk(
    ParakeetImpl& impl,
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
    if (t.frame_index > SIZE_MAX - offset) {
      throw std::runtime_error("Integer overflow in frame_index adjustment: frame_index=" +
                               std::to_string(t.frame_index) + " offset=" + std::to_string(offset));
    }
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
    ParakeetImpl& impl,
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
    if (is_debug_enabled()) {
      std::cerr << "[EDDY_DEBUG] Entire chunk consists of overlapped region; appending nothing\n";
    }
  } else {
    const size_t append_count = emit_end - skip_count;

    if (token_ids.capacity() < token_ids.size() + append_count) {
      token_ids.reserve(token_ids.size() + append_count + VECTOR_RESERVE_SLACK);
    }
    token_ids.insert(token_ids.end(), chunk_tokens.begin() + skip_count, chunk_tokens.begin() + emit_end);

    if (all_timings.capacity() < all_timings.size() + append_count) {
      all_timings.reserve(all_timings.size() + append_count + VECTOR_RESERVE_SLACK);
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

// Process mel features - handles both single chunk and multi-chunk (long audio) cases
MelProcessingResult process_mel_features(
    ParakeetImpl& impl,
    const MelFeatures& mel,
    const SegmentOptions& options,
    size_t max_frames) {

  MelProcessingResult result;

  if (mel.frames <= max_frames) {
    // Short audio - process in single chunk
    DecoderState state;
    auto pipeline_result = run_chunk_pipeline(impl, mel, options, state, /*is_last_chunk=*/true);

    result.encoder_ms = pipeline_result.encoder_ms;
    result.decoder_ms = pipeline_result.decoder_ms;
    result.joint_ms = pipeline_result.joint_ms;
    result.token_ids = std::move(pipeline_result.tokens);
    result.all_timings = std::move(pipeline_result.timings);
    result.chunk_sizes_frames.push_back(mel.frames);
  } else {
    // Long audio - process in overlapping chunks
    const size_t overlap_frames = std::min<size_t>(
        max_frames > 1 ? max_frames - 1 : 1,
        std::max<size_t>(4, max_frames / 9));
    const size_t stride = max_frames - overlap_frames;

    DecoderState decoder_state;
    size_t offset = 0;
    size_t chunk_idx = 0;
    size_t last_emitted_global_frame = 0;
    bool have_last_emitted_frame = false;

    while (offset < mel.frames) {
      const size_t chunk_size = std::min(max_frames, mel.frames - offset);
      const bool is_last_chunk = (offset + chunk_size >= mel.frames);
      result.chunk_sizes_frames.push_back(chunk_size);

      // Extract and process chunk
      MelFeatures chunk = extract_mel_chunk(mel, offset, chunk_size);
      auto pipeline_result = run_chunk_pipeline(impl, chunk, options, decoder_state, is_last_chunk);

      result.encoder_ms += pipeline_result.encoder_ms;
      result.decoder_ms += pipeline_result.decoder_ms;
      result.joint_ms += pipeline_result.joint_ms;

      // Prepare chunk info
      InferenceResult::ChunkInfo ci;
      ci.index = chunk_idx;
      ci.offset_frames = offset;
      ci.size_frames = chunk_size;
      ci.is_last = is_last_chunk;
      ci.tokens_predicted = pipeline_result.tokens.size();

      // Process chunk based on whether it's first or subsequent
      if (chunk_idx == 0) {
        process_first_chunk(impl, pipeline_result.tokens, pipeline_result.timings, offset,
                            result.token_ids, result.all_timings, last_emitted_global_frame,
                            have_last_emitted_frame, ci);
      } else {
        process_subsequent_chunk(impl, pipeline_result.tokens, pipeline_result.timings,
                                 offset, chunk_size, overlap_frames, is_last_chunk,
                                 result.token_ids, result.all_timings, last_emitted_global_frame,
                                 have_last_emitted_frame, ci);
      }

      result.chunk_logs.push_back(ci);
      chunk_idx++;

      if (offset + chunk_size >= mel.frames) {
        break;
      }

      offset += stride;
    }
  }

  return result;
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

std::string OpenVINOParakeet::decode_tokens(const std::vector<int>& token_ids) const {
  ensure_compiled_model();
  return impl_->tokenizer.decode(token_ids);
}

void OpenVINOParakeet::ensure_compiled_model() const {
  std::call_once(impl_->compile_once, [this]() {
    // ========================================
    // Device configuration
    // ========================================
    auto& core = impl_->backend->core();
    const std::string device = impl_->runtime_cfg.device.empty() ? "AUTO" : impl_->runtime_cfg.device;

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
    impl_->encoder_model = compile_with_npu_fallback(core, impl_->model_paths.encoder, device, "encoder");
    impl_->encoder_request = impl_->encoder_model.create_infer_request();

    impl_->decoder_model = compile_with_npu_fallback(core, impl_->model_paths.decoder, device, "decoder");
    impl_->decoder_request = impl_->decoder_model.create_infer_request();

    impl_->joint_model = compile_with_npu_fallback(core, impl_->model_paths.joint, device, "joint");
    impl_->joint_request = impl_->joint_model.create_infer_request();

    // ========================================
    // Load tokenizer
    // ========================================
    impl_->tokenizer.load(impl_->model_paths.tokenizer_json, impl_->runtime_cfg.blank_token_id);

    // ========================================
    // Resolve encoder ports and extract metadata
    // ========================================
    try {
      impl_->encoder_ports.mel_in = impl_->encoder_model.input("melspectogram");
      impl_->encoder_ports.len_in = impl_->encoder_model.input("melspectogram_length");
      impl_->encoder_ports.enc_out = impl_->encoder_model.output("encoder_output");
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Failed to resolve encoder ports: ") + e.what());
    }

    const auto mel_shape = impl_->encoder_ports.mel_in.value().get_shape();
    impl_->encoder_expected_frames = mel_shape.empty() ? 0U : mel_shape.back();

    // Fallback for dynamic shapes: Use 1250 frames (10 seconds at 125 frames/sec mel rate)
    // This enables chunking for models with dynamic encoder inputs
    if (impl_->encoder_expected_frames == 0) {
        impl_->encoder_expected_frames = 1250;
    }

    // ========================================
    // Determine encoder output indices and hidden size
    // ========================================
    const auto enc_out = impl_->encoder_model.output("encoder_output");
    const auto enc_len = impl_->encoder_model.output("encoder_output_length");
    const auto outs = impl_->encoder_model.outputs();

    for (size_t i = 0; i < outs.size(); ++i) {
      if (outs[i] == enc_out) {
        impl_->encoder_output_index = i;

        const auto shape = enc_out.get_shape();
        if (shape.size() >= 2) {
          impl_->encoder_hidden_size = shape[1];
          if (shape[1] == 0) {
            throw std::runtime_error("Encoder hidden size cannot be zero");
          }
        }
      }

      if (outs[i] == enc_len) {
        impl_->encoder_length_index = i;
      }
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
    // Extract joint output size and validate config
    // ========================================
    const auto joint_shape = impl_->joint_model.output("logits").get_shape();
    if (joint_shape.empty()) {
      throw std::runtime_error("Joint model logits tensor has no dimensions");
    }

    impl_->joint_output_size = joint_shape.back();

    const auto effective_vocab_size = static_cast<size_t>(impl_->runtime_cfg.blank_token_id) + 1;
    const auto total_heads = effective_vocab_size + impl_->runtime_cfg.duration_bins.size();

    if (total_heads > impl_->joint_output_size) {
      throw std::runtime_error("Joint model output smaller than token+duration heads");
    }

    if (std::getenv("EDDY_DEBUG")) {
      std::cerr << "[DEBUG] Joint output size: " << impl_->joint_output_size
                << ", token head: " << effective_vocab_size
                << ", duration bins: " << impl_->runtime_cfg.duration_bins.size() << "\n";
    }
  });
}

InferenceResult OpenVINOParakeet::infer(const AudioSegment& segment, const SegmentOptions& options) {
  ensure_compiled_model();

  const auto start = std::chrono::steady_clock::now();

  // Run preprocessing and inference pipeline
  MelProcessingResult mel_result;
  double t_preproc_ms = 0.0;
  {
    std::lock_guard<std::mutex> lock(impl_->request_guard);

    auto t0 = std::chrono::steady_clock::now();
    const auto mel = run_preprocessor(*impl_, segment);
    auto t1 = std::chrono::steady_clock::now();
    t_preproc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    mel_result = process_mel_features(*impl_, mel, options, impl_->encoder_expected_frames);
  }

  const auto end = std::chrono::steady_clock::now();
  const auto latency_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();

  // Build result
  InferenceResult result;
  result.token_ids = std::move(mel_result.token_ids);
  result.text = impl_->tokenizer.decode(result.token_ids);
  result.latency_ms = latency_ms;
  result.token_timings = std::move(mel_result.all_timings);
  result.chunk_sizes_frames = std::move(mel_result.chunk_sizes_frames);
  result.chunks = std::move(mel_result.chunk_logs);

  // Calculate overall confidence
  if (!result.token_timings.empty()) {
    float sum = 0.0F;
    for (const auto& timing : result.token_timings) {
      sum += timing.confidence;
    }
    result.overall_confidence = sum / static_cast<float>(result.token_timings.size());
  } else {
    result.overall_confidence = 0.1F;
  }

  return result;
}

void OpenVINOParakeet::warmup() {
  ensure_compiled_model();
}

}  // namespace eddy::parakeet

