#include "eddy/models/parakeet-v2/parakeet_decoder.hpp"
#include "parakeet_openvino_impl.hpp"
#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace eddy::parakeet {

// Named constants for decoder behavior
constexpr size_t DEFAULT_MAX_ADDITIONAL_STEPS = 8;
constexpr size_t DEFAULT_MAX_CONSECUTIVE_BLANKS = 1;

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

    // Declare tensors for this iteration
    ov::Tensor decoder_output;
    ov::Tensor next_hidden;
    ov::Tensor next_cell;

    // Use cached decoder output if available (optimization)
    if (state.has_cached_output && last_token == state.last_token.value_or(-1)) {
      std::memcpy(joint_dec_in.data<float>(), state.cached_decoder_output.data<float>(), joint_dec_in.get_byte_size());
      next_hidden = hidden_state;
      next_cell = cell_state;

    } else {
      // Run decoder inference
      if (targets_et == ov::element::i64) {
        token_input.data<int64_t>()[0] = static_cast<int64_t>(last_token);
      } else {
        token_input.data<int32_t>()[0] = static_cast<int32_t>(last_token);
      }

      impl.decoder_request.set_tensor(impl.decoder_model.input("targets"), token_input);
      impl.decoder_request.set_tensor(impl.decoder_model.input("h_in"), hidden_state);
      impl.decoder_request.set_tensor(impl.decoder_model.input("c_in"), cell_state);

      auto td0b = std::chrono::steady_clock::now();
      impl.decoder_request.infer();
      auto td1b = std::chrono::steady_clock::now();
      t_decoder_ms += std::chrono::duration<double, std::milli>(td1b - td0b).count();

      // Extract decoder outputs
      decoder_output = impl.decoder_request.get_output_tensor(0);
      next_hidden = impl.decoder_request.get_output_tensor(1);
      next_cell = impl.decoder_request.get_output_tensor(2);

      // Copy decoder output to joint network input
      std::memcpy(joint_dec_in.data<float>(), decoder_output.data<float>(), decoder_output.get_byte_size());

      // Cache decoder output for potential reuse
      state.cached_decoder_output = ov::Tensor(ov::element::f32, {1, 1, impl.decoder_hidden_size});
      std::memcpy(state.cached_decoder_output.data<float>(), decoder_output.data<float>(), decoder_output.get_byte_size());
      state.has_cached_output = true;
    }

    // Prepare encoder output for joint network
    const float* enc = encoder.tensor.data<float>();
    float* dst = joint_enc_in.data<float>();

    // Bounds check: verify we can read the last channel's last frame
    const size_t max_offset = (impl.encoder_hidden_size - 1) * encoder.time_steps + last_frame;
    if (max_offset >= encoder.tensor.get_size()) {
      throw std::runtime_error("Encoder tensor access out of bounds: offset=" +
                               std::to_string(max_offset) +
                               " size=" + std::to_string(encoder.tensor.get_size()));
    }

    // Copy encoder output for last frame to joint network input
    for (size_t channel = 0; channel < impl.encoder_hidden_size; ++channel) {
      const size_t off = channel * encoder.time_steps + last_frame;
      dst[channel] = enc[off];
    }

    // Run joint network inference
    auto tj0b = std::chrono::steady_clock::now();
    impl.joint_request.infer();
    auto tj1b = std::chrono::steady_clock::now();
    t_joint_ms += std::chrono::duration<double, std::milli>(tj1b - tj0b).count();

    // Extract logits and find best token
    const auto logits_tensor = impl.joint_request.get_output_tensor(0);
    const float* logits = logits_tensor.data<float>();

    size_t best_token = 0;
    float best_token_score = logits[tokens_offset + 0];
    for (size_t i = 1; i < vocab_size; ++i) {
      const float score = logits[tokens_offset + i];
      if (score > best_token_score) {
        best_token_score = score;
        best_token = i;
      }
    }

    // Calculate token confidence (softmax probability)
    float token_confidence = 0.0F;
    if (track_confidence) {
      float token_sum_exp = 0.0F;
      for (size_t i = 0; i < vocab_size; ++i) {
        token_sum_exp += std::exp(logits[tokens_offset + i]);
      }
      token_confidence = std::exp(best_token_score) / token_sum_exp;
    }

    // Check if best token is blank
    const bool is_blank = static_cast<int>(best_token) == impl.runtime_cfg.blank_token_id;

    if (!is_blank) {
      // Non-blank token: emit it and update state
      const int token_id = static_cast<int>(best_token);
      tokens.push_back(token_id);
      timings.push_back({.token_id = token_id, .frame_index = last_frame, .confidence = token_confidence});

      std::memcpy(hidden_state.data<float>(), next_hidden.data<float>(), next_hidden.get_byte_size());
      std::memcpy(cell_state.data<float>(), next_cell.data<float>(), next_cell.get_byte_size());

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
    // Check if we can use cached decoder output
    ov::Tensor decoder_output;
    ov::Tensor next_hidden;
    ov::Tensor next_cell;

    if (state.has_cached_output && last_token == state.last_token.value_or(-1)) {
      // CACHE HIT: Reuse cached decoder output - significant speedup!
      cache_hits++;
      std::memcpy(joint_dec_in.data<float>(), state.cached_decoder_output.data<float>(), decoder_step.get_byte_size());
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

      decoder_output = impl.decoder_request.get_output_tensor(0);
      next_hidden = impl.decoder_request.get_output_tensor(1);
      next_cell = impl.decoder_request.get_output_tensor(2);

      // Store decoder output for joint network
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

      // Extract encoder frame for current timestep
      {
        const float* enc = encoder.tensor.data<float>();
        float* dst = joint_enc_in.data<float>();

        const size_t max_offset = (impl.encoder_hidden_size - 1) * encoder.time_steps + frame_index;
        if (max_offset >= encoder.tensor.get_size()) {
          throw std::runtime_error("Encoder tensor access out of bounds in inner loop: offset=" +
                                   std::to_string(max_offset) +
                                   " size=" + std::to_string(encoder.tensor.get_size()));
        }

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

      const auto logits_tensor = impl.joint_request.get_output_tensor(0);
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
        std::memcpy(hidden_state.data<float>(), next_hidden.data<float>(), next_hidden.get_byte_size());
        std::memcpy(cell_state.data<float>(), next_cell.data<float>(), next_cell.get_byte_size());

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
