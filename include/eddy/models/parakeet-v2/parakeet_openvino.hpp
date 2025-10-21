#pragma once

#include <memory>
#include <optional>
#include <string>

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/models/parakeet-v2/parakeet.hpp"

#include <openvino/openvino.hpp>

namespace ov {
class infer_request;
class CompiledModel;
}

namespace eddy::parakeet {

/// Decoder state for maintaining context across audio chunks
/// Enables state continuity similar to FluidAudio's TdtDecoderState
struct DecoderState {
    /// Last decoded token from previous chunk
    /// Used for maintaining linguistic context across chunk boundaries
    std::optional<int> last_token;

    /// Time jump tracking for chunk alignment
    /// Represents how far the decoder progressed beyond encoder frames
    /// - nullopt: First chunk or no streaming context
    /// - negative: Decoder hasn't processed all encoder frames yet
    /// - zero: Decoder exactly at the end of encoder frames
    /// - positive: Decoder has advanced beyond current encoder frames
    std::optional<int> time_jump;

    /// LSTM hidden state (shape: [2, 1, decoder_hidden_size])
    /// Preserves the decoder's internal linguistic context across chunks
    ov::Tensor hidden_state;

    /// LSTM cell state (shape: [2, 1, decoder_hidden_size])
    /// Preserves the decoder's internal memory across chunks
    ov::Tensor cell_state;

    /// Flag indicating whether LSTM state tensors contain valid data
    /// false = use zero-initialized state (first chunk)
    /// true = use preserved state from previous chunk
    bool has_lstm_state = false;

    /// Cached decoder output tensor (shape: [1, 1, decoder_hidden_size])
    /// Reused across iterations when the token hasn't changed
    /// Key optimization: avoids redundant LSTM computations for same token
    ov::Tensor cached_decoder_output;

    /// Flag indicating whether cached_decoder_output contains valid data
    /// Set to true after running decoder, false when token changes
    bool has_cached_output = false;

    /// Reset all state to initial values
    void reset() {
        last_token.reset();
        time_jump.reset();
        has_lstm_state = false;
        has_cached_output = false;
        // Note: Tensors themselves are not deallocated, just marked as invalid
    }

    /// Check if state contains any data
    [[nodiscard]] bool has_state() const {
        return last_token.has_value() || time_jump.has_value() || has_lstm_state;
    }
};

// Forward declaration
class OpenVINOParakeet;

std::shared_ptr<OpenVINOParakeet> make_openvino_parakeet(std::shared_ptr<eddy::OpenVINOBackend> backend,
                                                        ModelPaths model_paths,
                                                        RuntimeConfig runtime_cfg);

class OpenVINOParakeet : public IParakeetModel {
public:
  OpenVINOParakeet(std::shared_ptr<eddy::OpenVINOBackend> backend,
                   ModelPaths model_paths,
                   RuntimeConfig runtime_cfg);
  ~OpenVINOParakeet() override;

  InferenceResult infer(const AudioSegment& segment, const SegmentOptions& options) override;

  std::string decode_tokens(const std::vector<int>& token_ids) const override;

  void warmup();

  // Public to allow helper functions in implementation file
  struct Impl;

private:
  void ensure_compiled_model() const;

  std::unique_ptr<Impl> impl_;
};

}  // namespace eddy::parakeet
