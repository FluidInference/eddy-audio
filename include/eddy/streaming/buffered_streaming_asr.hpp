#pragma once

#include "eddy/models/parakeet/parakeet.hpp"
#include <deque>
#include <memory>
#include <vector>
#include <string>

namespace eddy::streaming {

/// Configuration for buffered streaming ASR
struct BufferedStreamingConfig {
    /// Length of the center chunk to transcribe (default: 6.0 seconds)
    float chunk_len_in_secs = 6.0f;

    /// Total buffer size including context (default: 10.0 seconds)
    float total_buffer_in_secs = 10.0f;

    /// Sample rate (default: 16000 Hz)
    int sample_rate = 16000;

    /// Calculated: left context in samples
    int left_context_samples() const {
        int chunk_samples = static_cast<int>(chunk_len_in_secs * sample_rate);
        int total_samples = static_cast<int>(total_buffer_in_secs * sample_rate);
        return (total_samples - chunk_samples) / 2;
    }

    /// Calculated: right context in samples
    int right_context_samples() const {
        return left_context_samples(); // Same as left for symmetry
    }

    /// Calculated: center chunk in samples
    int center_chunk_samples() const {
        return static_cast<int>(chunk_len_in_secs * sample_rate);
    }

    /// Calculated: total buffer in samples
    int total_buffer_samples() const {
        return static_cast<int>(total_buffer_in_secs * sample_rate);
    }
};

/// Streaming ASR result for a single chunk
struct StreamingResult {
    std::string text;                    // Partial transcription
    std::vector<int> token_ids;          // Token IDs (after overlap removal)
    float confidence;                    // Average confidence
    bool is_final;                       // True if this is the last chunk
};

/// Buffered Streaming ASR
///
/// Uses a sliding window approach with left/center/right context:
/// ┌─────────────────────────────────────────┐
/// │  LEFT   │   CENTER   │   RIGHT          │
/// │ (2 sec) │   (6 sec)  │  (2 sec)         │
/// │ context │  new audio │  context         │
/// └─────────────────────────────────────────┘
///
/// Example usage:
/// ```cpp
/// BufferedStreamingASR asr(model, config);
///
/// // Process audio chunks as they arrive
/// for (const auto& chunk : audio_stream) {
///     auto result = asr.process_chunk(chunk);
///     if (!result.text.empty()) {
///         std::cout << "Partial: " << result.text << std::endl;
///     }
/// }
///
/// // Finalize when stream ends
/// auto final = asr.finalize();
/// std::cout << "Final: " << final.text << std::endl;
/// ```
class BufferedStreamingASR {
public:
    /// Constructor
    /// @param model Parakeet model instance
    /// @param config Streaming configuration
    BufferedStreamingASR(
        std::shared_ptr<parakeet::IParakeetModel> model,
        const BufferedStreamingConfig& config = {}
    );

    /// Process an incoming audio chunk
    /// @param audio_chunk Audio samples (mono, 16kHz)
    /// @return Streaming result with partial transcription
    StreamingResult process_chunk(const std::vector<float>& audio_chunk);

    /// Finalize streaming and process remaining audio
    /// @return Final streaming result
    StreamingResult finalize();

    /// Reset state for a new audio stream
    void reset();

    /// Get total samples received
    size_t total_samples_received() const { return samples_received_; }

    /// Get total samples processed
    size_t total_samples_processed() const { return samples_processed_; }

private:
    /// Process the current buffer window
    StreamingResult process_current_window();

    /// Remove overlapping tokens between previous and current output
    std::vector<int> remove_token_overlap(
        const std::vector<int>& prev_tokens,
        const std::vector<int>& current_tokens
    );

    /// Check if buffer has enough data to process
    bool has_enough_data() const {
        return audio_buffer_.size() >= static_cast<size_t>(config_.total_buffer_samples());
    }

private:
    std::shared_ptr<parakeet::IParakeetModel> model_;
    BufferedStreamingConfig config_;

    // Audio buffer (sliding window)
    std::deque<float> audio_buffer_;

    // State tracking
    std::vector<int> previous_tokens_;  // Tokens from previous window
    size_t samples_received_;           // Total audio samples received
    size_t samples_processed_;          // Total audio samples transcribed

    // LSTM state is maintained by the model between infer() calls
    // (assuming the model preserves state across chunks)
};

} // namespace eddy::streaming
