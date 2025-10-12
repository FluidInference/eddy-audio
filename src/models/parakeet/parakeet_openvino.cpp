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

private:
  std::vector<std::string> vocab_;
  int blank_id_ = 1024;
};

ov::CompiledModel compile_component(ov::Core& core, const ModelFile& file, const std::string& device) {
  if (file.path.empty()) {
    throw std::invalid_argument("Parakeet component path is empty");
  }

  if (file.compiled) {
    std::ifstream blob_stream(file.path, std::ios::binary);
    if (!blob_stream.good()) {
      throw std::runtime_error("Failed to open compiled blob: " + file.path);
    }
    return core.import_model(blob_stream, device);
  }

  return core.compile_model(file.path, device);
}

struct MelFeatures {
  std::vector<float> data;
  size_t frames = 0;
};

struct EncoderActivations {
  std::vector<float> data;
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

void OpenVINOParakeet::ensure_compiled_model() {
  std::call_once(impl_->compile_once, [this]() {
    auto& core = impl_->backend->core();
    const std::string device = impl_->runtime_cfg.device.empty() ? "AUTO" : impl_->runtime_cfg.device;

    impl_->preproc_model = compile_component(core, impl_->model_paths.preprocessor, device);
    impl_->preproc_request = impl_->preproc_model.create_infer_request();

    impl_->encoder_model = compile_component(core, impl_->model_paths.encoder, device);
    impl_->encoder_request = impl_->encoder_model.create_infer_request();

    impl_->decoder_model = compile_component(core, impl_->model_paths.decoder, device);
    impl_->decoder_request = impl_->decoder_model.create_infer_request();

    impl_->joint_model = compile_component(core, impl_->model_paths.joint, device);
    impl_->joint_request = impl_->joint_model.create_infer_request();

    impl_->tokenizer.load(impl_->model_paths.tokenizer_json, impl_->runtime_cfg.blank_token_id);

    impl_->encoder_expected_frames = required_length(impl_->encoder_model.input("melspectogram"));
    const auto encoder_output_shape = impl_->encoder_model.output("encoder_output").get_shape();
    if (encoder_output_shape.size() != 3) {
      throw std::runtime_error("Unexpected encoder output rank; expected [1, hidden, time]");
    }
    impl_->encoder_hidden_size = encoder_output_shape[1];

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

  ov::Tensor audio_signal(ov::element::f32, {1, segment.pcm.size()});
  std::copy(segment.pcm.begin(), segment.pcm.end(), audio_signal.data<float>());

  ov::Tensor audio_length(ov::element::i64, {1});
  audio_length.data<int64_t>()[0] = static_cast<int64_t>(segment.pcm.size());

  impl.preproc_request.set_input_tensor(0, audio_signal);  // audio_signal
  impl.preproc_request.set_input_tensor(1, audio_length);  // audio_length
  impl.preproc_request.infer();

  const auto mel_tensor = impl.preproc_request.get_output_tensor(0);  // melspectrogram
  const auto length_tensor = impl.preproc_request.get_output_tensor(1);  // melspectrogram_length

  const int64_t valid_frames = length_tensor.data<int64_t>()[0];
  if (valid_frames <= 0) {
    throw std::runtime_error("Preprocessor returned zero mel frames");
  }

  MelFeatures features;
  features.frames = static_cast<size_t>(valid_frames);

  const auto mel_shape = mel_tensor.get_shape();
  if (mel_shape.size() != 3 || mel_shape[1] != 128) {
    throw std::runtime_error("Unexpected mel tensor shape from preprocessor");
  }

  const size_t mel_bins = mel_shape[1];
  const size_t time_steps = mel_shape[2];
  const size_t elements = mel_bins * time_steps;

  features.data.resize(elements);
  std::copy(mel_tensor.data<float>(), mel_tensor.data<float>() + elements, features.data.begin());

  return features;
}

EncoderActivations run_encoder(OpenVINOParakeet::Impl& impl, const MelFeatures& mel) {
  if (impl.encoder_expected_frames == 0) {
    throw std::runtime_error("Encoder expected frame count is zero");
  }

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

  ov::Tensor mel_length(ov::element::i32, {1});
  mel_length.data<int32_t>()[0] = static_cast<int32_t>(frames_to_copy);

  impl.encoder_request.set_input_tensor(0, mel_tensor);  // melspectogram
  impl.encoder_request.set_input_tensor(1, mel_length);  // melspectogram_length
  impl.encoder_request.infer();

  const auto encoder_tensor = impl.encoder_request.get_output_tensor(0);  // encoder_output
  const auto encoder_length_tensor = impl.encoder_request.get_output_tensor(1);  // encoder_output_length

  EncoderActivations activations;
  activations.hidden_size = impl.encoder_hidden_size;
  const auto shape = encoder_tensor.get_shape();
  if (shape.size() != 3 || shape[1] != impl.encoder_hidden_size) {
    throw std::runtime_error("Unexpected encoder output tensor shape");
  }
  activations.time_steps = shape[2];
  activations.valid_frames = static_cast<size_t>(std::min<int64_t>(encoder_length_tensor.data<int64_t>()[0], static_cast<int64_t>(activations.time_steps)));

  const size_t elements = encoder_tensor.get_size();
  activations.data.resize(elements);
  std::copy(encoder_tensor.data<float>(), encoder_tensor.data<float>() + elements, activations.data.begin());

  return activations;
}

struct DecoderResult {
  std::vector<int> tokens;
  std::vector<TokenTiming> timings;
};

DecoderResult run_greedy_decoder(OpenVINOParakeet::Impl& impl,
                                 const EncoderActivations& encoder,
                                 const SegmentOptions& options,
                                 DecoderState& state) {
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

  ov::Tensor encoder_step(ov::element::f32, {1, 1, impl.encoder_hidden_size});
  ov::Tensor decoder_step(ov::element::f32, {1, 1, impl.decoder_hidden_size});

  // Initialize or reuse LSTM state
  ov::Tensor hidden_state(ov::element::f32, {2, 1, impl.decoder_hidden_size});
  ov::Tensor cell_state(ov::element::f32, {2, 1, impl.decoder_hidden_size});

  if (state.has_lstm_state) {
    // Continue from previous chunk's LSTM state
    std::cerr << "[INFO] Continuing with preserved LSTM state from previous chunk\n";
    std::memcpy(hidden_state.data<float>(), state.hidden_state.data<float>(), hidden_state.get_byte_size());
    std::memcpy(cell_state.data<float>(), state.cell_state.data<float>(), cell_state.get_byte_size());
  } else {
    // First chunk: zero-initialize LSTM state
    std::cerr << "[INFO] Starting with fresh LSTM state (first chunk)\n";
    std::fill(hidden_state.data<float>(), hidden_state.data<float>() + hidden_state.get_size(), 0.0F);
    std::fill(cell_state.data<float>(), cell_state.data<float>() + cell_state.get_size(), 0.0F);
  }

  // Use last token from previous chunk if available, otherwise blank
  int starting_token = state.last_token.value_or(impl.runtime_cfg.blank_token_id);
  std::cerr << "[INFO] Starting decoder with token: " << starting_token << "\n";

  ov::Tensor token_input(ov::element::i32, {1, 1});
  token_input.data<int32_t>()[0] = starting_token;

  std::vector<int> tokens;
  std::vector<TokenTiming> timings;
  tokens.reserve(options.max_tokens);
  timings.reserve(options.max_tokens);

  size_t frame_index = 0;
  int last_token = starting_token;

  while (frame_index < valid_frames && tokens.size() < options.max_tokens) {
    token_input.data<int32_t>()[0] = last_token;

    impl.decoder_request.set_input_tensor(0, token_input);  // targets
    impl.decoder_request.set_input_tensor(1, hidden_state);  // h_in
    impl.decoder_request.set_input_tensor(2, cell_state);  // c_in
    impl.decoder_request.infer();

    const auto decoder_output = impl.decoder_request.get_output_tensor(0);  // decoder_output
    const auto next_hidden = impl.decoder_request.get_output_tensor(1);  // h_out
    const auto next_cell = impl.decoder_request.get_output_tensor(2);  // c_out

    std::memcpy(decoder_step.data<float>(), decoder_output.data<float>(), decoder_output.get_byte_size());

    // Extract encoder frame for current timestep
    for (size_t channel = 0; channel < impl.encoder_hidden_size; ++channel) {
      const size_t offset = channel * encoder.time_steps + frame_index;
      encoder_step.data<float>()[channel] = encoder.data[offset];
    }

    impl.joint_request.set_input_tensor(0, encoder_step);  // encoder_outputs
    impl.joint_request.set_input_tensor(1, decoder_step);  // decoder_outputs
    impl.joint_request.infer();

    const auto logits_tensor = impl.joint_request.get_output_tensor(0);  // logits
    const float* logits = logits_tensor.data<float>();

    size_t best_token = 0;
    float best_token_score = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
      if (logits[i] > best_token_score) {
        best_token_score = logits[i];
        best_token = i;
      }
    }

    // Apply softmax to token logits to get probability
    float token_sum_exp = 0.0F;
    for (size_t i = 0; i < vocab_size; ++i) {
      token_sum_exp += std::exp(logits[i]);
    }
    float token_confidence = std::exp(best_token_score) / token_sum_exp;

    size_t best_duration_idx = 0;
    float best_duration_score = logits[vocab_size];
    for (size_t i = 1; i < duration_head; ++i) {
      const float score = logits[vocab_size + i];
      if (score > best_duration_score) {
        best_duration_score = score;
        best_duration_idx = i;
      }
    }

    int duration = impl.runtime_cfg.duration_bins[best_duration_idx];
    if (duration <= 0) {
      duration = 1;
    }

    const bool is_blank = static_cast<int>(best_token) == impl.runtime_cfg.blank_token_id;

    if (!is_blank) {
      const int token_id = static_cast<int>(best_token);
      tokens.push_back(token_id);

      // Record timing information
      timings.push_back({
        .token_id = token_id,
        .frame_index = frame_index,
        .confidence = token_confidence
      });

      last_token = token_id;
      std::memcpy(hidden_state.data<float>(), next_hidden.data<float>(), next_hidden.get_byte_size());
      std::memcpy(cell_state.data<float>(), next_cell.data<float>(), next_cell.get_byte_size());
    }

    frame_index = std::min(frame_index + static_cast<size_t>(duration), valid_frames);
  }

  // Save final LSTM state and last token for next chunk
  // Always allocate fresh tensors for state (OpenVINO tensors can't be default-constructed safely)
  state.hidden_state = ov::Tensor(ov::element::f32, {2, 1, impl.decoder_hidden_size});
  state.cell_state = ov::Tensor(ov::element::f32, {2, 1, impl.decoder_hidden_size});

  std::memcpy(state.hidden_state.data<float>(), hidden_state.data<float>(), hidden_state.get_byte_size());
  std::memcpy(state.cell_state.data<float>(), cell_state.data<float>(), cell_state.get_byte_size());
  state.has_lstm_state = true;

  if (!tokens.empty()) {
    state.last_token = tokens.back();
    std::cerr << "[INFO] Saved final LSTM state and last token: " << *state.last_token << "\n";
  } else {
    state.last_token = starting_token;
    std::cerr << "[INFO] No tokens emitted, keeping starting token: " << starting_token << "\n";
  }

  return {std::move(tokens), std::move(timings)};
}

}  // namespace

InferenceResult OpenVINOParakeet::infer(const AudioSegment& segment, const SegmentOptions& options) {
  ensure_compiled_model();

  const auto start = std::chrono::steady_clock::now();

  std::vector<int> token_ids;
  std::vector<TokenTiming> all_timings;
  {
    std::lock_guard<std::mutex> lock(impl_->request_guard);
    const auto mel = run_preprocessor(*impl_, segment);

    // Check if audio is longer than encoder can handle
    const size_t max_frames = impl_->encoder_expected_frames;

    if (mel.frames <= max_frames) {
      // Short audio - process normally
      DecoderState state;  // Fresh state for single-chunk audio
      const auto encoder = run_encoder(*impl_, mel);
      auto result = run_greedy_decoder(*impl_, encoder, options, state);
      token_ids = std::move(result.tokens);
      all_timings = std::move(result.timings);
    } else {
      // Long audio - process in overlapping chunks
      std::cerr << "[INFO] Audio too long (" << mel.frames << " frames), processing in chunks\n";

      // Use overlap to avoid losing audio at boundaries
      const size_t overlap_frames = 300;  // ~3 seconds of overlap
      const size_t stride = max_frames - overlap_frames;

      // Persistent decoder state across chunks for full state continuity
      DecoderState decoder_state;

      size_t offset = 0;
      size_t chunk_idx = 0;

      while (offset < mel.frames) {
        // Extract chunk with overlap
        const size_t chunk_size = std::min(max_frames, mel.frames - offset);

        MelFeatures chunk;
        chunk.frames = chunk_size;
        chunk.data.resize(128 * chunk_size);

        // Copy mel data for this chunk (mel is stored as [mel_bins][time])
        for (size_t bin = 0; bin < 128; ++bin) {
          const float* src = mel.data.data() + bin * mel.frames + offset;
          float* dst = chunk.data.data() + bin * chunk_size;
          std::copy(src, src + chunk_size, dst);
        }

        std::cerr << "[INFO] Processing chunk " << chunk_idx
                  << " at offset " << offset
                  << " (size: " << chunk_size << " frames)\n";

        const auto encoder = run_encoder(*impl_, chunk);
        // Pass persistent decoder_state - it will be updated with final state after decoding
        auto decoder_result = run_greedy_decoder(*impl_, encoder, options, decoder_state);
        auto& chunk_tokens = decoder_result.tokens;
        auto& chunk_timings = decoder_result.timings;

        if (chunk_idx == 0) {
          // First chunk: keep all tokens and timings
          token_ids = std::move(chunk_tokens);
          all_timings = std::move(chunk_timings);
        } else {
          // Subsequent chunks: use deduplication as safety net (FluidAudio approach)
          // Even with LSTM state continuity, decoder state reset at chunk boundaries
          // can cause some duplicates, so we apply deduplication as a safety net
          size_t skip_count = 0;

          std::cerr << "[DEBUG] Last 10 tokens of prev chunk: ";
          for (size_t i = std::max(size_t(0), token_ids.size() - 10); i < token_ids.size(); ++i) {
            std::cerr << token_ids[i] << " ";
          }
          std::cerr << "\n[DEBUG] First 10 tokens of curr chunk: ";
          for (size_t i = 0; i < std::min(size_t(10), chunk_tokens.size()); ++i) {
            std::cerr << chunk_tokens[i] << " ";
          }
          std::cerr << "\n";

          // Search for duplicate sequences (safety net for state continuity)
          const size_t curr_search_window = std::min(size_t(15), chunk_tokens.size());
          const size_t prev_tail_window = std::min(size_t(20), token_ids.size());
          const size_t min_match_len = 3;

          size_t best_skip_count = 0;
          size_t best_match_len = 0;

          for (size_t curr_pos = 0; curr_pos < curr_search_window && best_skip_count == 0; ++curr_pos) {
            for (size_t prev_offset = 0; prev_offset < prev_tail_window; ++prev_offset) {
              const size_t max_match_len = std::min({
                size_t(15),
                token_ids.size() - prev_offset,
                chunk_tokens.size() - curr_pos,
                prev_tail_window - prev_offset
              });

              for (size_t match_len = max_match_len; match_len >= min_match_len; --match_len) {
                bool match = true;
                const size_t prev_start = token_ids.size() - prev_tail_window + prev_offset;

                for (size_t i = 0; i < match_len; ++i) {
                  if (token_ids[prev_start + i] != chunk_tokens[curr_pos + i]) {
                    match = false;
                    break;
                  }
                }

                if (match && match_len > best_match_len) {
                  best_match_len = match_len;
                  best_skip_count = curr_pos + match_len;
                  std::cerr << "[INFO] Found " << match_len << " duplicate tokens at pos " << curr_pos
                            << ", skipping " << best_skip_count << " tokens (safety net)\n";
                  break;
                }
              }
            }
          }

          skip_count = best_skip_count;
          if (skip_count == 0) {
            std::cerr << "[INFO] No duplicates found - LSTM state continuity working perfectly!\n";
          }

          // Append tokens and timings, skipping duplicates
          token_ids.insert(token_ids.end(), chunk_tokens.begin() + skip_count, chunk_tokens.end());
          all_timings.insert(all_timings.end(), chunk_timings.begin() + skip_count, chunk_timings.end());
        }

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

  InferenceResult result;
  result.token_ids = std::move(token_ids);
  result.text = impl_->tokenizer.decode(result.token_ids);
  result.latency_ms = latency_ms;
  result.token_timings = std::move(all_timings);

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

  return result;
}

void OpenVINOParakeet::warmup() {
  ensure_compiled_model();
}

}  // namespace eddy::parakeet
