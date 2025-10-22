#include "eddy/models/parakeet-v2/parakeet_decoder.hpp"
#include "eddy/models/parakeet-v2/detail/parakeet_impl.hpp"
#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace eddy::parakeet {

namespace {

// Named constants for decoder behavior
constexpr size_t DEFAULT_MAX_ADDITIONAL_STEPS = 8;
constexpr size_t DEFAULT_MAX_CONSECUTIVE_BLANKS = 1;

// Helper: Find token with highest score (argmax)
size_t find_best_token(const float* logits, size_t offset, size_t vocab_size, float& out_score) {
  size_t best_token = 0;
  float best_score = logits[offset + 0];
  for (size_t i = 1; i < vocab_size; ++i) {
    const float score = logits[offset + i];
    if (score > best_score) {
      best_score = score;
      best_token = i;
    }
  }
  out_score = best_score;
  return best_token;
}

// Helper: Calculate softmax probability (confidence) for a token
float calculate_confidence(const float* logits, size_t offset, size_t vocab_size, float token_score) {
  float sum_exp = 0.0F;
  for (size_t i = 0; i < vocab_size; ++i) {
    sum_exp += std::exp(logits[offset + i]);
  }
  // Protect against division by zero (unlikely but possible if all logits are -inf)
  if (sum_exp > 0.0F) {
    return std::exp(token_score) / sum_exp;
  }
  return 0.0F;
}

// Helper: Extract encoder frame data into joint network input tensor
void extract_encoder_frame(const EncoderActivations& encoder,
                           size_t frame_index,
                           size_t encoder_hidden_size,
                           ov::Tensor& joint_enc_in) {
  const float* enc = encoder.tensor.data<float>();
  float* dst = joint_enc_in.data<float>();

  // Bounds check: verify we can read the last channel's frame
  const size_t max_offset = (encoder_hidden_size - 1) * encoder.time_steps + frame_index;
  if (max_offset >= encoder.tensor.get_size()) {
    throw std::runtime_error("Encoder tensor access out of bounds: offset=" +
                             std::to_string(max_offset) +
                             " size=" + std::to_string(encoder.tensor.get_size()));
  }

  // Copy encoder output for the specified frame
  for (size_t channel = 0; channel < encoder_hidden_size; ++channel) {
    const size_t offset = channel * encoder.time_steps + frame_index;
    dst[channel] = enc[offset];
  }
}

// Helper: Run decoder or use cached output
struct DecoderOutput {
  ov::Tensor next_hidden;
  ov::Tensor next_cell;
  bool used_cache;
};

DecoderOutput run_decoder_or_use_cache(ParakeetImpl& impl,
                                       DecoderState& state,
                                       int last_token,
                                       ov::Tensor& hidden_state,
                                       ov::Tensor& cell_state,
                                       ov::Tensor& token_input,
                                       ov::element::Type targets_et,
                                       ov::Tensor& joint_dec_in,
                                       double& t_decoder_ms) {
  DecoderOutput result;

  // Check if we can use cached decoder output
  if (state.has_cached_output && last_token == state.last_token.value_or(-1)) {
    // CACHE HIT: Reuse cached decoder output
    std::memcpy(joint_dec_in.data<float>(), state.cached_decoder_output.data<float>(),
                joint_dec_in.get_byte_size());
    result.next_hidden = hidden_state;
    result.next_cell = cell_state;
    result.used_cache = true;

  } else {
    // CACHE MISS: Need to run decoder LSTM
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

    ov::Tensor decoder_output = impl.decoder_request.get_output_tensor(0);
    result.next_hidden = impl.decoder_request.get_output_tensor(1);
    result.next_cell = impl.decoder_request.get_output_tensor(2);

    // Copy decoder output to joint network input
    std::memcpy(joint_dec_in.data<float>(), decoder_output.data<float>(),
                decoder_output.get_byte_size());

    // Cache this decoder output for next iteration
    state.cached_decoder_output = ov::Tensor(ov::element::f32, {1, 1, impl.decoder_hidden_size});
    std::memcpy(state.cached_decoder_output.data<float>(), decoder_output.data<float>(),
                decoder_output.get_byte_size());
    state.has_cached_output = true;
    result.used_cache = false;
  }

  return result;
}

}  // namespace

void initialize_decoder_state(ParakeetImpl& impl,
                              DecoderState& state,
                              int blank_token_id,
                              int& starting_token,
                              ov::Tensor& hidden_state,
                              ov::Tensor& cell_state,
                              ov::Tensor& token_input,
                              ov::element::Type& targets_et) {
  // Initialize LSTM state tensors
  hidden_state = ov::Tensor(ov::element::f32, {2, 1, impl.decoder_hidden_size});
  cell_state = ov::Tensor(ov::element::f32, {2, 1, impl.decoder_hidden_size});

  // Restore or initialize LSTM state
  if (state.has_lstm_state) {
    if (std::getenv("EDDY_DEBUG")) {
      std::cerr << "[INFO] Continuing with preserved LSTM state from previous chunk\n";
    }
    std::memcpy(hidden_state.data<float>(), state.hidden_state.data<float>(), hidden_state.get_byte_size());
    std::memcpy(cell_state.data<float>(), state.cell_state.data<float>(), cell_state.get_byte_size());
  } else {
    if (std::getenv("EDDY_DEBUG")) {
      std::cerr << "[INFO] Starting with fresh LSTM state (first chunk)\n";
    }
    std::fill(hidden_state.data<float>(), hidden_state.data<float>() + hidden_state.get_size(), 0.0F);
    std::fill(cell_state.data<float>(), cell_state.data<float>() + cell_state.get_size(), 0.0F);
  }

  // Set starting token
  starting_token = state.last_token.value_or(blank_token_id);

  // Create token input tensor with correct type
  auto targets_port = impl.decoder_model.input("targets");
  targets_et = targets_port.get_element_type();

  if (targets_et == ov::element::i64) {
    token_input = ov::Tensor(ov::element::i64, {1, 1});
    token_input.data<int64_t>()[0] = static_cast<int64_t>(starting_token);
  } else {
    token_input = ov::Tensor(ov::element::i32, {1, 1});
    token_input.data<int32_t>()[0] = static_cast<int32_t>(starting_token);
  }
}

void finalize_chunk_decoding(ParakeetImpl& impl,
                             const EncoderActivations& encoder,
                             size_t vocab_size,
                             size_t tokens_offset,
                             bool track_confidence,
                             ov::Tensor& hidden_state,
                             ov::Tensor& cell_state,
                             ov::Tensor& token_input,
                             ov::element::Type targets_et,
                             int& last_token,
                             std::vector<int>& tokens,
                             std::vector<TokenTiming>& timings,
                             double& t_decoder_ms,
                             double& t_joint_ms,
                             DecoderState& state,
                             size_t max_tokens) {
  // Get last valid encoder frame
  const size_t valid_frames = std::min(encoder.valid_frames, encoder.time_steps);
  if (valid_frames == 0) return;

  const size_t last_frame = valid_frames > 0 ? (valid_frames - 1) : 0;

  // Initialize stopping criteria
  size_t additional_steps = 0;
  size_t consecutive_blanks = 0;
  size_t max_additional_steps = DEFAULT_MAX_ADDITIONAL_STEPS;
  size_t max_consecutive_blanks = DEFAULT_MAX_CONSECUTIVE_BLANKS;

  // Allow environment variable overrides
  if (const char* env_steps = std::getenv("EDDY_MAX_ADDITIONAL_STEPS")) {
    try {
      int v = std::stoi(env_steps);
      if (v >= 0) max_additional_steps = static_cast<size_t>(v);
    } catch (const std::exception& e) {
      std::cerr << "[WARN] Invalid EDDY_MAX_ADDITIONAL_STEPS value '" << env_steps << "', using default\n";
    }
  }
  if (const char* env_blanks = std::getenv("EDDY_MAX_CONSEC_BLANKS")) {
    try {
      int v = std::stoi(env_blanks);
      if (v >= 1) max_consecutive_blanks = static_cast<size_t>(v);
    } catch (const std::exception& e) {
      std::cerr << "[WARN] Invalid EDDY_MAX_CONSEC_BLANKS value '" << env_blanks << "', using default\n";
    }
  }

  // Get joint network input tensors
  ov::Tensor joint_enc_in = impl.joint_request.get_input_tensor(0);
  ov::Tensor joint_dec_in = impl.joint_request.get_input_tensor(1);

  // Main decoding loop - continues until stopping criteria met
  while (additional_steps < max_additional_steps &&
         consecutive_blanks < max_consecutive_blanks &&
         tokens.size() < max_tokens) {

    // Run decoder or use cached output
    auto decoder_out = run_decoder_or_use_cache(impl, state, last_token, hidden_state, cell_state,
                                                 token_input, targets_et, joint_dec_in, t_decoder_ms);

    // Extract encoder frame for joint network
    extract_encoder_frame(encoder, last_frame, impl.encoder_hidden_size, joint_enc_in);

    // Run joint network inference
    auto tj0b = std::chrono::steady_clock::now();
    impl.joint_request.infer();
    auto tj1b = std::chrono::steady_clock::now();
    t_joint_ms += std::chrono::duration<double, std::milli>(tj1b - tj0b).count();

    // Extract logits and find best token
    const auto logits_tensor = impl.joint_request.get_output_tensor(0);
    const float* logits = logits_tensor.data<float>();

    float best_token_score;
    size_t best_token = find_best_token(logits, tokens_offset, vocab_size, best_token_score);

    // Calculate token confidence (softmax probability)
    float token_confidence = 0.0F;
    if (track_confidence) {
      token_confidence = calculate_confidence(logits, tokens_offset, vocab_size, best_token_score);
    }

    // Check if best token is blank
    const bool is_blank = static_cast<int>(best_token) == impl.runtime_cfg.blank_token_id;

    if (!is_blank) {
      // Non-blank token: emit it and update state
      const int token_id = static_cast<int>(best_token);
      tokens.push_back(token_id);
      timings.push_back({.token_id = token_id, .frame_index = last_frame, .confidence = token_confidence});

      std::memcpy(hidden_state.data<float>(), decoder_out.next_hidden.data<float>(), decoder_out.next_hidden.get_byte_size());
      std::memcpy(cell_state.data<float>(), decoder_out.next_cell.data<float>(), decoder_out.next_cell.get_byte_size());

      last_token = token_id;
      state.has_cached_output = false;
      consecutive_blanks = 0;
    } else {
      // Blank token: increment blank counter
      consecutive_blanks++;
    }

    additional_steps++;
  }
}

// TDT Decoder
//
// TDT (Token-and-Duration Transducer) decoding algorithm:
// 1. Run decoder LSTM once to get language model state
// 2. Inner loop: Process frames with joint network until non-blank token
//    - Blank tokens reuse same decoder output (key optimization)
//    - Non-blank tokens require new decoder run
// 3. Advance to next frame, repeat
//
// At each step, selects the token with highest probability (argmax).
// This is the standard TDT algorithm from the Parakeet paper.
DecoderResult run_decoder(ParakeetImpl& impl,
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
  ov::Tensor hidden_state;
  ov::Tensor cell_state;
  ov::Tensor token_input;
  ov::element::Type targets_et = ov::element::i32;
  int starting_token = impl.runtime_cfg.blank_token_id;
  initialize_decoder_state(impl, state, impl.runtime_cfg.blank_token_id, starting_token, hidden_state, cell_state, token_input, targets_et);
  if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Starting decoder with token: " << starting_token << "\n";

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
    // Run decoder or use cached output
    auto decoder_out = run_decoder_or_use_cache(impl, state, last_token, hidden_state, cell_state,
                                                 token_input, targets_et, joint_dec_in, t_decoder_ms);

    // Track cache statistics
    if (decoder_out.used_cache) {
      cache_hits++;
    } else {
      decoder_runs++;
    }

    // TDT Inner Loop: Process consecutive blank tokens without re-running decoder
    // This is the key optimization - decoder output is intentionally reused
    // because blank tokens (silence) shouldn't change language model context
    bool advance_mask = true;
    while (advance_mask && frame_index < valid_frames && tokens.size() < options.max_tokens) {
      total_joint_calls++;

      // Extract encoder frame for current timestep
      extract_encoder_frame(encoder, frame_index, impl.encoder_hidden_size, joint_enc_in);

      // Run joint network with encoder frame + REUSED decoder output
      auto tj0 = std::chrono::steady_clock::now();
      impl.joint_request.infer();
      auto tj1 = std::chrono::steady_clock::now();
      t_joint_ms += std::chrono::duration<double, std::milli>(tj1 - tj0).count();

      const auto logits_tensor = impl.joint_request.get_output_tensor(0);
      const float* logits = logits_tensor.data<float>();

      // Find best token within token head region
      float best_token_score;
      size_t best_token = find_best_token(logits, tokens_offset, vocab_size, best_token_score);

      // Calculate confidence only if requested
      float token_confidence = 0.0F;
      if (track_confidence) {
        token_confidence = calculate_confidence(logits, tokens_offset, vocab_size, best_token_score);
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
        duration = 1;
      }

      const bool is_blank = static_cast<int>(best_token) == impl.runtime_cfg.blank_token_id;

      if (!is_blank) {
        // Non-blank token: emit it and exit inner loop
        non_blank_tokens++;

        const int token_id = static_cast<int>(best_token);
        tokens.push_back(token_id);

        timings.push_back({
          .token_id = token_id,
          .frame_index = frame_index,
          .confidence = token_confidence
        });

        last_token = token_id;

        // Update LSTM state with new token's context
        std::memcpy(hidden_state.data<float>(), decoder_out.next_hidden.data<float>(), decoder_out.next_hidden.get_byte_size());
        std::memcpy(cell_state.data<float>(), decoder_out.next_cell.data<float>(), decoder_out.next_cell.get_byte_size());

        // Invalidate cache - force decoder run next iteration
        state.has_cached_output = false;

        advance_mask = false;

      } else {
        // Blank token: continue inner loop
        blank_tokens++;
        advance_mask = true;
      }

      // Advance frame index by predicted duration
      frame_index = std::min(frame_index + static_cast<size_t>(duration), valid_frames);
    }
  }

  // Last-chunk finalization: for the final audio chunk only, continue at
  // the last encoder frame until a small blank threshold or max steps.
  // This flushes trailing tokens that need extra predictor iterations.
  if (is_last_chunk) {
    finalize_chunk_decoding(impl, encoder, vocab_size, tokens_offset, track_confidence,
                            hidden_state, cell_state, token_input, targets_et,
                            last_token, tokens, timings, t_decoder_ms, t_joint_ms, state,
                            options.max_tokens);
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

}  // namespace eddy::parakeet
