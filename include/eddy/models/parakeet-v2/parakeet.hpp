#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eddy::parakeet {

struct ModelFile {
  std::string path;
  bool compiled = false;
};

struct ModelPaths {
  ModelFile preprocessor;
  ModelFile encoder;
  ModelFile decoder;
  ModelFile joint;
  std::string tokenizer_json;
};

struct RuntimeConfig {
  std::string device = "AUTO";
  int blank_token_id = 1024;
  std::vector<int> duration_bins = {0, 1, 2, 3, 4};
};

struct SegmentOptions {
  size_t max_tokens = 256;
  float temperature = 0.0F;
};

struct AudioSegment {
  std::vector<float> pcm;          // Raw audio samples
  std::vector<float> features;     // Pre-computed mel spectrogram features (optional, computed if empty)
  size_t sample_rate = 16000;      // Sample rate in Hz (must be 16kHz for Parakeet)
  double timestamp_seconds = 0.0;  // Starting timestamp for this segment
};

/// Token timing information for a single decoded token
struct TokenTiming {
  int token_id;           // Token ID (vocabulary index)
  size_t frame_index;     // Encoder frame index (convert to seconds: frame * 0.08)
  float confidence;       // Token confidence score from joint network [0.0-1.0]
};

struct InferenceResult {
  std::string text;
  double latency_ms = 0.0;
  std::vector<int> token_ids;

  /// Overall transcription confidence (average of token confidences)
  float overall_confidence = 0.0F;

  /// Per-token timing and confidence information
  /// Empty if token timing tracking is disabled
  std::vector<TokenTiming> token_timings;

  /// Chunking metadata (for long audio processed in overlapping windows)
  /// Number of chunks is chunk_sizes_frames.size(); for short audio this
  /// contains a single entry with total valid frames.
  std::vector<size_t> chunk_sizes_frames;

  /// Detailed per-chunk log for benchmarking/export
  struct ChunkInfo {
    size_t index = 0;            // chunk index
    size_t offset_frames = 0;    // starting frame offset in global mel
    size_t size_frames = 0;      // frames processed in this chunk
    bool is_last = false;        // whether last chunk
    size_t tokens_predicted = 0; // tokens produced by decoder for this chunk
    size_t tokens_appended = 0;  // tokens appended after dedup/holdback
    size_t skip_prefix = 0;      // dedup prefix skipped from current chunk
    size_t holdback = 0;         // tokens held back for right-context lookahead
    std::string appended_text;   // decoded text of the appended token slice
  };
  std::vector<ChunkInfo> chunks;
};

class IParakeetModel {
public:
  virtual ~IParakeetModel() noexcept = default;

  // Prevent copying and moving (use shared_ptr/unique_ptr instead)
  IParakeetModel(const IParakeetModel&) = delete;
  IParakeetModel& operator=(const IParakeetModel&) = delete;
  IParakeetModel(IParakeetModel&&) = delete;
  IParakeetModel& operator=(IParakeetModel&&) = delete;

  virtual InferenceResult infer(const AudioSegment& segment, const SegmentOptions& options) = 0;

  /// Decode token IDs to text using the tokenizer
  /// @param token_ids Vector of token IDs to decode
  /// @return Decoded text string
  virtual std::string decode_tokens(const std::vector<int>& token_ids) const = 0;

protected:
  IParakeetModel() = default;
};

}  // namespace eddy::parakeet
