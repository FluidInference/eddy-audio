#pragma once

#include <openvino/openvino.hpp>
#include <vector>
#include "eddy/models/parakeet-v2/parakeet.hpp"

namespace eddy::parakeet {

// Forward declarations
struct ParakeetImpl;

// Mel-spectrogram features from preprocessor
struct MelFeatures {
  // Native tensors from preprocessor (for zero-copy encoder handoff when shapes match)
  ov::Tensor mel_tensor;       // [1, 128, T]
  ov::Tensor length_tensor;    // [1]
  size_t time_steps = 0;       // T

  // Convenience buffer for chunking path (time-major [mel_bins][time])
  std::vector<float> data;
  size_t frames = 0;           // valid frames (<= time_steps)
};

// Run the mel-spectrogram preprocessor on audio segment
MelFeatures run_preprocessor(ParakeetImpl& impl, const AudioSegment& segment);

}  // namespace eddy::parakeet
