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
  std::vector<float> pcm;
  std::vector<float> features;
  size_t sample_rate = 16000;
  double timestamp_seconds = 0.0;
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
};

class IParakeetModel {
public:
  virtual ~IParakeetModel() = default;
  virtual InferenceResult infer(const AudioSegment& segment, const SegmentOptions& options) = 0;

  /// Decode token IDs to text using the tokenizer
  /// @param token_ids Vector of token IDs to decode
  /// @return Decoded text string
  virtual std::string decode_tokens(const std::vector<int>& token_ids) = 0;
};

}  // namespace eddy::parakeet
