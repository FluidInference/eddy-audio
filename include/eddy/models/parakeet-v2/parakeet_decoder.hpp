#pragma once

#include <openvino/openvino.hpp>
#include <vector>
#include "eddy/models/parakeet-v2/parakeet.hpp"
#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"

namespace eddy::parakeet {

// Forward declarations
struct ParakeetImpl;
struct EncoderActivations;

// Decoder result containing tokens, timings, and performance metrics
struct DecoderResult {
  std::vector<int> tokens;
  std::vector<TokenTiming> timings;
  double t_decoder_ms = 0.0;
  double t_joint_ms = 0.0;
  size_t decoder_runs = 0;
  size_t joint_calls = 0;
};

// Initialize decoder state at the beginning of chunk decoding
void initialize_decoder_state(ParakeetImpl& impl,
                              DecoderState& state,
                              int blank_token_id,
                              int& starting_token,
                              ov::Tensor& hidden_state,
                              ov::Tensor& cell_state,
                              ov::Tensor& token_input,
                              ov::element::Type& targets_et);

// Finalize chunk decoding by processing the last encoder frame
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
                             size_t max_tokens);

// Run TDT decoder on encoder activations
DecoderResult run_decoder(ParakeetImpl& impl,
                                const EncoderActivations& encoder,
                                const SegmentOptions& options,
                                DecoderState& state,
                                bool is_last_chunk);

}  // namespace eddy::parakeet
