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
constexpr size_t DEFAULT_MAX_SYMBOLS_PER_STEP = 8;  // v3 guard against runaway emissions at same timestamp

// Read boundary search frames (used during finalization near chunk end)
size_t boundary_search_frames() {
  static size_t cached = []() {
    if (const char* e = std::getenv("EDDY_BOUNDARY_SEARCH_FRAMES")) {
      try {
        int v = std::stoi(e);
        if (v > 0 && v < 10000) return static_cast<size_t>(v);
      } catch (...) {
      }
    }
    return size_t(20);
  }();
  return cached;
}

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
                                       ov::Tensor& joint_dec_step,
                                       double& t_decoder_ms) {
  DecoderOutput result;

  // Check if we can use cached decoder output
  if (state.has_cached_output && last_token == state.last_token.value_or(-1)) {
    // CACHE HIT: Reuse cached decoder output
    std::memcpy(joint_dec_step.data<float>(), state.cached_decoder_output.data<float>(),
                joint_dec_step.get_byte_size());
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

    auto td0 = std::chrono::steady_clock::now();
    impl.decoder_request.infer();
    auto td1 = std::chrono::steady_clock::now();
    t_decoder_ms += std::chrono::duration<double, std::milli>(td1 - td0).count();

    ov::Tensor decoder_output = impl.decoder_request.get_output_tensor(impl.decoder_proj_index);
    ov::Tensor h_out = impl.decoder_request.get_output_tensor(impl.decoder_h_index);
    ov::Tensor c_out = impl.decoder_request.get_output_tensor(impl.decoder_c_index);
    if (std::getenv("EDDY_SWAP_DEC_STATE") && std::string(std::getenv("EDDY_SWAP_DEC_STATE")) == "1") {
      result.next_hidden = c_out;
      result.next_cell = h_out;
    } else {
      result.next_hidden = h_out;
      result.next_cell = c_out;
    }

    // Copy decoder output to joint network input
    std::memcpy(joint_dec_step.data<float>(), decoder_output.data<float>(),
                decoder_output.get_byte_size());

    // Cache this decoder output for next iteration
    state.cached_decoder_output = ov::Tensor(ov::element::f32, {1, 1, impl.decoder_hidden_size});
    std::memcpy(state.cached_decoder_output.data<float>(), decoder_output.data<float>(),
                decoder_output.get_byte_size());
    state.has_cached_output = true;
    state.last_token = last_token;
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

  // Pre-bind decoder inputs once; we will only mutate their contents per step
  impl.decoder_request.set_tensor(impl.decoder_model.input("targets"), token_input);
  impl.decoder_request.set_tensor(impl.decoder_model.input("h_in"), hidden_state);
  impl.decoder_request.set_tensor(impl.decoder_model.input("c_in"), cell_state);
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
  // Preserve v2 behavior: skip boundary finalization when using v2 blank id (1024)
  if (impl.runtime_cfg.blank_token_id != 8192) {
    return;
  }
  // Get last valid encoder frame
  const size_t valid_frames = std::min(encoder.valid_frames, encoder.time_steps);
  if (valid_frames == 0) return;

  const size_t last_frame = valid_frames - 1;

  // Limits and knobs
  size_t max_additional_steps = DEFAULT_MAX_ADDITIONAL_STEPS;
  size_t max_consecutive_blanks = DEFAULT_MAX_CONSECUTIVE_BLANKS;
  size_t max_symbols_per_step = DEFAULT_MAX_SYMBOLS_PER_STEP;

  if (const char* env_steps = std::getenv("EDDY_MAX_ADDITIONAL_STEPS")) {
    try { int v = std::stoi(env_steps); if (v >= 0) max_additional_steps = static_cast<size_t>(v); } catch (...) {}
  }
  if (const char* env_blanks = std::getenv("EDDY_MAX_CONSEC_BLANKS")) {
    try { int v = std::stoi(env_blanks); if (v >= 1) max_consecutive_blanks = static_cast<size_t>(v); } catch (...) {}
  }
  if (const char* env_symbols = std::getenv("EDDY_MAX_SYMBOLS_PER_STEP")) {
    try { int v = std::stoi(env_symbols); if (v >= 1) max_symbols_per_step = static_cast<size_t>(v); } catch (...) {}
  }

  // Joint step tensors for dynamic input binding (v3 only)
  ov::Tensor joint_enc_in = ov::Tensor(ov::element::f32, {1, 1, impl.encoder_hidden_size});
  ov::Tensor joint_dec_in = ov::Tensor(ov::element::f32, {1, 1, impl.decoder_hidden_size});

  // Sliding window over boundary frames to flush any residual tokens
  const size_t durations_offset = (tokens_offset == 0) ? vocab_size : 0;
  const size_t search_span = std::min(boundary_search_frames(), valid_frames);
  size_t additional_steps = 0;
  size_t consecutive_blanks = 0;
  size_t emissions_at_ts = 0;
  size_t last_emission_frame = SIZE_MAX;

  size_t probe_frame = last_frame;  // start at last frame

  while (additional_steps < max_additional_steps &&
         consecutive_blanks < max_consecutive_blanks &&
         tokens.size() < max_tokens) {

    // Run decoder or use cached projection
    auto decoder_out = run_decoder_or_use_cache(impl, state, last_token,
                                               hidden_state, cell_state, token_input,
                                               targets_et, joint_dec_in, t_decoder_ms);

    // Try multiple frames near boundary to solicit a non-blank emission
    bool emitted_token_this_step = false;
    size_t local_best_frame = probe_frame;
    float local_best_conf = -1.0f;
    size_t local_best_token = static_cast<size_t>(impl.runtime_cfg.blank_token_id);
    size_t local_best_duration_idx = 0;

    for (size_t k = 0; k < search_span; ++k) {
      size_t f = (probe_frame >= k) ? (probe_frame - k) : 0;
      extract_encoder_frame(encoder, f, impl.encoder_hidden_size, joint_enc_in);

      // Bind inputs for this probe
      impl.joint_request.set_input_tensor(impl.joint_enc_input_index, joint_enc_in);
      impl.joint_request.set_input_tensor(impl.joint_dec_input_index, joint_dec_in);

      auto tj0b = std::chrono::steady_clock::now();
      impl.joint_request.infer();
      auto tj1b = std::chrono::steady_clock::now();
      t_joint_ms += std::chrono::duration<double, std::milli>(tj1b - tj0b).count();

      const auto logits_tensor = impl.joint_request.get_output_tensor(0);
      const float* logits = logits_tensor.data<float>();

      float best_token_score;
      size_t best_token = find_best_token(logits, tokens_offset, vocab_size, best_token_score);
      const bool is_blank = static_cast<int>(best_token) == impl.runtime_cfg.blank_token_id;

      // Track most confident non-blank across the search window
      if (!is_blank) {
        float conf = track_confidence ? calculate_confidence(logits, tokens_offset, vocab_size, best_token_score)
                                      : 1.0f;
        if (conf > local_best_conf) {
          local_best_conf = conf;
          local_best_token = best_token;
          local_best_frame = f;
          // Also pick best duration for timestamp advance afterward
          size_t best_duration_idx = 0; float best_duration_score = logits[durations_offset + 0];
          for (size_t i = 1; i < impl.runtime_cfg.duration_bins.size(); ++i) {
            const float s = logits[durations_offset + i];
            if (s > best_duration_score) { best_duration_score = s; best_duration_idx = i; }
          }
          local_best_duration_idx = best_duration_idx;
        }
      }
    }

    // Emit the best candidate if found
    if (local_best_conf >= 0.0f && static_cast<int>(local_best_token) != impl.runtime_cfg.blank_token_id) {
      const int token_id = static_cast<int>(local_best_token);
      tokens.push_back(token_id);
      float token_conf = track_confidence ? local_best_conf : 0.0f;
      timings.push_back({.token_id = token_id, .frame_index = local_best_frame, .confidence = token_conf});

      // Update decoder state with LSTM outputs
      std::memcpy(hidden_state.data<float>(), decoder_out.next_hidden.data<float>(), decoder_out.next_hidden.get_byte_size());
      std::memcpy(cell_state.data<float>(), decoder_out.next_cell.data<float>(), decoder_out.next_cell.get_byte_size());
      last_token = token_id;
      state.has_cached_output = false;  // next step must run decoder

      // Timestamp guard (avoid runaway at same frame)
      if (last_emission_frame == local_best_frame) { emissions_at_ts++; } else { emissions_at_ts = 1; last_emission_frame = local_best_frame; }
      if (emissions_at_ts >= max_symbols_per_step) {
        emissions_at_ts = 0;
        consecutive_blanks = max_consecutive_blanks;  // exit
      }

      emitted_token_this_step = true;
    } else {
      // No non-blank found in window: treat as blank
      consecutive_blanks++;
    }

    additional_steps++;
    if (probe_frame > 0) probe_frame--; else break;
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
  // Dynamic-binding step tensors (used when use_dynamic_binding=true)
  ov::Tensor joint_enc_step(ov::element::f32, {1, 1, impl.encoder_hidden_size});
  ov::Tensor joint_dec_step(ov::element::f32, {1, 1, impl.decoder_hidden_size});
  // Pre-bound joint encoder input (used when not using dynamic binding)
  ov::Tensor joint_enc_in = impl.joint_request.get_input_tensor(impl.joint_enc_input_index);
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

  // Guard against runaway emissions at same timestamp (v3 behavior)
  size_t emissions_at_ts = 0;
  size_t last_emission_frame = SIZE_MAX;
  size_t max_symbols_per_step = DEFAULT_MAX_SYMBOLS_PER_STEP;
  if (const char* env_symbols = std::getenv("EDDY_MAX_SYMBOLS_PER_STEP")) {
    try { int v = std::stoi(env_symbols); if (v >= 1) max_symbols_per_step = static_cast<size_t>(v); } catch (...) {}
  }

  // Prepare step tensors for dynamic joint inputs
  // (defined once above to avoid reallocation)

  // Confidence computation toggle via env (default off)
  const bool track_confidence = (std::getenv("EDDY_TRACK_CONFIDENCE") && std::string(std::getenv("EDDY_TRACK_CONFIDENCE")) == "1");

  // TDT timings
  double t_decoder_ms = 0.0;
  double t_joint_ms = 0.0;

  // Decide binding strategy: v3 uses dynamic per-step binding; v2 uses pre-bound tensors
  const bool use_dynamic_binding = (impl.runtime_cfg.blank_token_id == 8192);

  // Outer loop: runs decoder, then enters inner loop for blank processing
  while (frame_index < valid_frames && tokens.size() < options.max_tokens) {
    // Run decoder or use cached output
    ov::Tensor& joint_dec_for_step = const_cast<ov::Tensor&>(use_dynamic_binding ? joint_dec_step :
                                       impl.joint_request.get_input_tensor(impl.joint_dec_input_index));
    auto decoder_out = run_decoder_or_use_cache(impl, state, last_token, hidden_state, cell_state,
                                                 token_input, targets_et, joint_dec_for_step, t_decoder_ms);

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

      if (use_dynamic_binding) {
        // Extract encoder frame and bind per step
        extract_encoder_frame(encoder, frame_index, impl.encoder_hidden_size, joint_enc_step);
        impl.joint_request.set_input_tensor(impl.joint_enc_input_index, joint_enc_step);
        impl.joint_request.set_input_tensor(impl.joint_dec_input_index, joint_dec_step);
      } else {
        // Write directly into pre-bound input tensors
        extract_encoder_frame(encoder, frame_index, impl.encoder_hidden_size, joint_enc_in);
        // joint_dec tensor already contains the latest predictor projection
      }

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
      const bool is_blank = static_cast<int>(best_token) == impl.runtime_cfg.blank_token_id;
      // Only force a minimum advance for blank predictions to guarantee progress
      if (is_blank && duration <= 0) {
        duration = 1;
      }

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

        // Timestamp guard
        if (last_emission_frame == frame_index) { emissions_at_ts++; } else { emissions_at_ts = 1; last_emission_frame = frame_index; }
        if (emissions_at_ts >= max_symbols_per_step) {
          // force advancement to prevent getting stuck
          emissions_at_ts = 0;
          advance_mask = false;
        } else {
          advance_mask = false;
        }

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

// V3-specific decoder path. Differences vs v2:
// - Strict predictor reuse across blanks (decoder not re-run for blanks)
// - Blank duration clamp to >=1 only for blanks
// - No boundary finalization pass (handled upstream if needed)
DecoderResult run_decoder_v3(ParakeetImpl& impl,
                                     const EncoderActivations& encoder,
                                     const SegmentOptions& options,
                                     DecoderState& state,
                                     bool /*is_last_chunk*/) {
  const size_t valid_frames = std::min(encoder.valid_frames, encoder.time_steps);
  if (valid_frames == 0) {
    return {{}, {}};
  }

  if (std::getenv("EDDY_DEBUG")) {
    std::cerr << "[DEBUG] run_decoder_v3: valid_frames=" << valid_frames
              << " time_steps=" << encoder.time_steps << "\n";
  }

  const size_t vocab_size = static_cast<size_t>(impl.runtime_cfg.blank_token_id) + 1;
  const size_t duration_head = impl.runtime_cfg.duration_bins.size();
  if (vocab_size == 0 || vocab_size + duration_head > impl.joint_output_size) {
    throw std::runtime_error("Invalid joint head configuration");
  }

  // Always tokens-first for v3
  const size_t tokens_offset = 0;
  const size_t durations_offset = vocab_size;

  ov::Tensor hidden_state;
  ov::Tensor cell_state;
  ov::Tensor token_input;
  ov::element::Type targets_et = ov::element::i32;
  int starting_token = impl.runtime_cfg.blank_token_id;
  initialize_decoder_state(impl, state, impl.runtime_cfg.blank_token_id, starting_token, hidden_state, cell_state, token_input, targets_et);

  std::vector<int> tokens;
  std::vector<TokenTiming> timings;
  tokens.reserve(options.max_tokens);
  timings.reserve(options.max_tokens);

  size_t frame_index = 0;
  int last_token = starting_token;

  size_t total_joint_calls = 0;
  size_t decoder_runs = 0;
  size_t cache_hits = 0;

  // Input binding strategy: by default, pre-bind tensors and mutate memory.
  // Some devices (e.g., NPU) may require explicit set_input_tensor per step.
  const bool use_dynamic_binding = (std::getenv("EDDY_V3_DYNAMIC_BINDING") && std::string(std::getenv("EDDY_V3_DYNAMIC_BINDING")) == "1");
  ov::Tensor joint_enc_in;
  ov::Tensor joint_dec_in;
  ov::Tensor joint_enc_step;
  ov::Tensor joint_dec_step;
  if (use_dynamic_binding) {
    // Keep encoder input pre-bound; only dynamic-bind decoder input per step
    joint_enc_in = impl.joint_request.get_input_tensor(impl.joint_enc_input_index);
    joint_dec_step = ov::Tensor(ov::element::f32, {1, 1, impl.decoder_hidden_size});
  } else {
    joint_enc_in = impl.joint_request.get_input_tensor(impl.joint_enc_input_index);
    joint_dec_in = impl.joint_request.get_input_tensor(impl.joint_dec_input_index);
  }

  double t_decoder_ms = 0.0;
  double t_joint_ms = 0.0;

  // Optional v3 stride: multiply duration advances to reduce joint calls (WER trade-off)
  int v3_stride = 1;
  if (const char* s = std::getenv("EDDY_V3_STRIDE")) {
    try {
      int v = std::stoi(s);
      if (v >= 1 && v <= 16) v3_stride = v;
    } catch (...) {}
  }

  // v3: Optional control prompt (disabled by default). Enable with EDDY_V3_PROMPT=1
  if (impl.runtime_cfg.blank_token_id == 8192) {
    const bool enable_prompt = (std::getenv("EDDY_V3_PROMPT") && std::string(std::getenv("EDDY_V3_PROMPT")) == "1");
    if (enable_prompt) {
      const int start_of_transcript = 4;   // <|startoftranscript|>
      const int english_lang = 64;         // <|en|>
      const int no_timestamps = 11;        // <|notimestamp|>
      const int enable_punc = 5;           // <|pnc|>
      const int prompt_seq[] = { start_of_transcript, english_lang, no_timestamps, enable_punc };

      ov::Tensor joint_dec_step(ov::element::f32, {1, 1, impl.decoder_hidden_size});
      for (int tok : prompt_seq) {
        last_token = tok;
        state.last_token = tok;
        auto dec = run_decoder_or_use_cache(impl, state, last_token, hidden_state, cell_state,
                                            token_input, targets_et, joint_dec_step, t_decoder_ms);
        // After each prompt token, update LSTM state
        std::memcpy(hidden_state.data<float>(), dec.next_hidden.data<float>(), dec.next_hidden.get_byte_size());
        std::memcpy(cell_state.data<float>(), dec.next_cell.data<float>(), dec.next_cell.get_byte_size());
        state.has_cached_output = false;  // ensure a fresh run at start of decoding loop
      }
    }
  }

  // Emission guard per timestamp
  size_t emissions_at_ts = 0;
  size_t last_emission_frame = SIZE_MAX;
  // v3 should allow multiple emissions per frame like Swift implementation (up to 8)
  size_t max_symbols_per_step = DEFAULT_MAX_SYMBOLS_PER_STEP;
  if (const char* env_symbols = std::getenv("EDDY_MAX_SYMBOLS_PER_STEP")) {
    try { int v = std::stoi(env_symbols); if (v >= 1) max_symbols_per_step = static_cast<size_t>(v); } catch (...) {}
  }

  // CRITICAL FIX: Prime decoder with SOS token (blank_id) if needed
  // This matches FluidAudio's v3 implementation which runs decoder once before main loop
  // to initialize predictorOutput and establish proper language model context
  if (!state.has_cached_output && state.last_token == std::nullopt) {
    ov::Tensor initial_proj(ov::element::f32, {1, 1, impl.decoder_hidden_size});
    auto priming = run_decoder_or_use_cache(impl, state, impl.runtime_cfg.blank_token_id,
                                           hidden_state, cell_state, token_input, targets_et,
                                           (use_dynamic_binding ? joint_dec_step : initial_proj), t_decoder_ms);
    // Cache this initial decoder output for first iteration
    state.cached_decoder_output = use_dynamic_binding ? joint_dec_step : initial_proj;
    state.has_cached_output = true;
    state.last_token = impl.runtime_cfg.blank_token_id;
    if (std::getenv("EDDY_DEBUG")) {
      std::cerr << "[DEBUG] Primed v3 decoder with SOS token (blank_id="
                << impl.runtime_cfg.blank_token_id << ")\n";
    }
  }

  while (frame_index < valid_frames && tokens.size() < options.max_tokens) {
    auto decoder_out = run_decoder_or_use_cache(impl, state, last_token, hidden_state, cell_state,
                                                token_input, targets_et,
                                                (use_dynamic_binding ? joint_dec_step : joint_dec_in),
                                                t_decoder_ms);
    if (decoder_out.used_cache) ++cache_hits; else ++decoder_runs;

    bool advance_mask = true;
    while (advance_mask && frame_index < valid_frames && tokens.size() < options.max_tokens) {
      ++total_joint_calls;

      // Fill inputs for this step - use safe index to avoid out-of-bounds
      size_t safe_frame_index = std::min(frame_index, valid_frames - 1);
      extract_encoder_frame(encoder, safe_frame_index, impl.encoder_hidden_size, joint_enc_in);
      if (use_dynamic_binding) {
        impl.joint_request.set_input_tensor(impl.joint_dec_input_index, joint_dec_step);
      }

      // Joint inference
      auto tj0 = std::chrono::steady_clock::now();
      impl.joint_request.infer();
      auto tj1 = std::chrono::steady_clock::now();
      t_joint_ms += std::chrono::duration<double, std::milli>(tj1 - tj0).count();

      const auto logits_tensor = impl.joint_request.get_output_tensor(0);
      const float* logits = logits_tensor.data<float>();

      // Argmax token
      float best_token_score;
      size_t best_token = find_best_token(logits, tokens_offset, vocab_size, best_token_score);

      // Argmax duration
      size_t best_duration_idx = 0; float best_duration_score = logits[durations_offset + 0];
      for (size_t i = 1; i < duration_head; ++i) {
        const float score = logits[durations_offset + i];
        if (score > best_duration_score) { best_duration_score = score; best_duration_idx = i; }
      }
      int duration = impl.runtime_cfg.duration_bins[best_duration_idx];
      if (v3_stride > 1) {
        long long scaled = 1LL * duration * v3_stride;
        if (scaled > 0) duration = static_cast<int>(std::min<long long>(scaled, static_cast<long long>(std::numeric_limits<int>::max())));
      }

      // Optional detailed trace for v3 debugging
      static const bool trace_enabled = (std::getenv("EDDY_V3_TRACE") != nullptr);
      static const size_t trace_limit = [](){
        if (const char* t = std::getenv("EDDY_V3_TRACE_LIMIT")) { try { int v = std::stoi(t); if (v>0) return static_cast<size_t>(v);} catch (...) {} }
        return static_cast<size_t>(128);
      }();
      static size_t trace_count = 0;

      const bool is_blank = static_cast<int>(best_token) == impl.runtime_cfg.blank_token_id;
      if (is_blank) {
        if (duration <= 0) duration = 1;  // guarantee progress on blanks
        if (trace_enabled && trace_count < trace_limit) {
          std::string piece;
          const int tok = static_cast<int>(best_token);
          if (tok == impl.runtime_cfg.blank_token_id) piece = "<blank>"; else piece = impl.tokenizer.decode_span(&tok, 1);
          std::cerr << "[V3-TRACE] frame=" << frame_index
                    << " token=" << tok << " ('" << piece << "')"
                    << " dur_bin=" << best_duration_idx << " dur=" << duration
                    << " cached=" << (decoder_out.used_cache?"1":"0")
                    << " -> advance(blank)\n";
          ++trace_count;
        }
        frame_index = std::min(frame_index + static_cast<size_t>(duration), valid_frames);
        advance_mask = true;  // continue inner loop
        continue;
      }

      // Non-blank: check for control/leading punctuation and possibly treat as blank
      const int token_id = static_cast<int>(best_token);
      // Resolve decoded piece (after removing word-boundary marker)
      std::string candidate_piece = impl.tokenizer.decode_span(&token_id, 1);
      bool is_control = (!candidate_piece.empty() && candidate_piece.rfind("<|", 0) == 0);
      bool leading_punct = (tokens.empty() && impl.tokenizer.is_punctuation(token_id));
      if (is_control || leading_punct) {
        // Treat as blank: advance by duration (ensure progress)
        if (duration <= 0) duration = 1;
        if (trace_enabled && trace_count < trace_limit) {
          std::cerr << "[V3-TRACE] frame=" << frame_index
                    << " token=" << token_id << " ('" << candidate_piece << "')"
                    << " dur_bin=" << best_duration_idx << " dur=" << duration
                    << " cached=" << (decoder_out.used_cache?"1":"0")
                    << " -> skip/control-as-blank\n";
          ++trace_count;
        }
        frame_index = std::min(frame_index + static_cast<size_t>(duration), valid_frames);
        advance_mask = true;
        continue;
      }

      // Non-blank: emit
      tokens.push_back(token_id);
      // Use applied duration for trace; record timing at current frame (clamped to valid range)
      size_t clamped_frame = std::min(frame_index, valid_frames - 1);
      if (std::getenv("EDDY_DEBUG") && tokens.size() <= 3) {
        std::cerr << "[DEBUG] Token " << tokens.size() << ": frame_index=" << frame_index
                  << " clamped_frame=" << clamped_frame << " valid_frames=" << valid_frames
                  << " token_id=" << token_id << "\n";
      }
      timings.push_back({.token_id = token_id, .frame_index = clamped_frame, .confidence = 0.0F});
      last_token = token_id;

      // Update LSTM state with new token context by immediately running predictor
      // This mirrors RNNT/TDT symbol update semantics (priming predictor for next steps)
      {
        ov::Tensor tmp_proj(ov::element::f32, {1, 1, impl.decoder_hidden_size});
        auto dec2 = run_decoder_or_use_cache(impl, state, last_token, hidden_state, cell_state,
                                             token_input, targets_et, tmp_proj, t_decoder_ms);
        // Adopt new hidden/cell state
        std::memcpy(hidden_state.data<float>(), dec2.next_hidden.data<float>(), dec2.next_hidden.get_byte_size());
        std::memcpy(cell_state.data<float>(), dec2.next_cell.data<float>(), dec2.next_cell.get_byte_size());
        // Cache the projection for immediate reuse on next outer iteration
        state.cached_decoder_output = tmp_proj;
        state.has_cached_output = true;
        state.last_token = last_token;
        // Copy tmp_proj into currently used joint decoder input buffer
        std::memcpy(joint_dec_in.data<float>(), tmp_proj.data<float>(), tmp_proj.get_byte_size());
      }

      if (trace_enabled && trace_count < trace_limit) {
        std::string piece = impl.tokenizer.decode_span(&token_id, 1);
        std::cerr << "[V3-TRACE] frame=" << frame_index
                  << " token=" << token_id << " ('" << piece << "')"
                  << " dur_bin=" << best_duration_idx << " dur=" << (duration<0?0:duration)
                  << " cached=" << (decoder_out.used_cache?"1":"0")
                  << " -> emit\n";
        ++trace_count;
      }

      // Emission guard
      if (last_emission_frame == frame_index) { ++emissions_at_ts; } else { emissions_at_ts = 1; last_emission_frame = frame_index; }
      if (emissions_at_ts >= max_symbols_per_step) {
        emissions_at_ts = 0;  // reset
      }

      // Advance by predicted duration even for non-blanks (as in Swift)
      if (duration <= 0) duration = 1;  // ensure progress on non-blank dur=0
      frame_index = std::min(frame_index + static_cast<size_t>(duration), valid_frames);
      advance_mask = false;  // leave inner loop
    }
  }

  // Save LSTM state for next chunk
  state.hidden_state = ov::Tensor(ov::element::f32, {2, 1, impl.decoder_hidden_size});
  state.cell_state = ov::Tensor(ov::element::f32, {2, 1, impl.decoder_hidden_size});
  std::memcpy(state.hidden_state.data<float>(), hidden_state.data<float>(), hidden_state.get_byte_size());
  std::memcpy(state.cell_state.data<float>(), cell_state.data<float>(), cell_state.get_byte_size());
  state.has_lstm_state = true;
  if (!tokens.empty()) state.last_token = tokens.back(); else state.last_token = starting_token;

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
