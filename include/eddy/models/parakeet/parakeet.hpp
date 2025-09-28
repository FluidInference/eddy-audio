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

struct InferenceResult {
  std::string text;
  double latency_ms = 0.0;
  std::vector<int> token_ids;
};

class IParakeetModel {
public:
  virtual ~IParakeetModel() = default;
  virtual InferenceResult infer(const AudioSegment& segment, const SegmentOptions& options) = 0;
};

}  // namespace eddy::parakeet
