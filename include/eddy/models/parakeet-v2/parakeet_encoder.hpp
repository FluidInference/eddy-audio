#pragma once

#include <openvino/openvino.hpp>
#include <optional>
#include <string>
#include <vector>

namespace eddy::parakeet {

// Forward declarations
struct ParakeetImpl;
struct MelFeatures;

// Helper to resolve encoder ports in a name/shape-robust way.
struct EncoderPorts {
  std::optional<ov::Output<const ov::Node>> mel_in;   // [1, 128, T]
  std::optional<ov::Output<const ov::Node>> len_in;   // [1]
  std::optional<ov::Output<const ov::Node>> enc_out;  // [1, hidden, time]
};

// Encoder activations output
struct EncoderActivations {
  ov::Tensor tensor;      // [1, hidden, time]
  size_t hidden_size = 0;
  size_t time_steps = 0;
  size_t valid_frames = 0;
};

// Run the encoder on mel-spectrogram features
EncoderActivations run_encoder(ParakeetImpl& impl, const MelFeatures& mel);

}  // namespace eddy::parakeet
