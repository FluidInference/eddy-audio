#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"
#include "eddy/models/parakeet-v2/detail/parakeet_impl.hpp"
#include "eddy/models/parakeet-v2/parakeet_preprocessor.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace eddy::parakeet {

namespace {

constexpr size_t MEL_BINS = 128;  // Standard mel-spectrogram bin count

// Read a single scalar length value from a tensor that may be i32 or i64.
// Validates that the length is non-negative.
int64_t read_length_scalar(const ov::Tensor& t) {
  const auto et = t.get_element_type();
  int64_t value;

  if (et == ov::element::i64) {
    value = t.data<int64_t>()[0];
  } else if (et == ov::element::i32) {
    int32_t val32 = t.data<int32_t>()[0];
    if (val32 < 0) {
      throw std::runtime_error("Encoder length is negative: " + std::to_string(val32));
    }
    value = static_cast<int64_t>(val32);
  } else {
    throw std::runtime_error("Encoder length tensor has unexpected element type (expected i32 or i64)");
  }

  if (value < 0) {
    throw std::runtime_error("Encoder length is negative: " + std::to_string(value));
  }

  return value;
}

}  // namespace

EncoderActivations run_encoder(ParakeetImpl& impl, const MelFeatures& mel) {
  if (impl.encoder_expected_frames == 0) {
    throw std::runtime_error("Encoder expected frame count is zero");
  }

  // Determine encoder length input element type to avoid device-specific mismatches (e.g., NPU strict typing)
  const auto len_port = impl.encoder_ports.len_in.value();
  const auto len_et = len_port.get_element_type();

  // Prepare mel tensor and determine actual frame count
  ov::Tensor mel_tensor;
  size_t actual_frames;

  // Check if we need to pad/trim
  const bool needs_padding = (mel.time_steps != impl.encoder_expected_frames || !mel.mel_tensor);

  if (needs_padding) {
    // Pad/trim: copy into a tensor matching encoder's expected frames
    mel_tensor = ov::Tensor(ov::element::f32, {1, MEL_BINS, impl.encoder_expected_frames});
    std::fill(mel_tensor.data<float>(), mel_tensor.data<float>() + mel_tensor.get_size(), 0.0F);

    actual_frames = std::min(impl.encoder_expected_frames, mel.frames);

    // Validate mel data buffer size
    const size_t required_size = MEL_BINS * mel.frames;
    if (mel.data.size() < required_size) {
      throw std::runtime_error("Mel data buffer too small: expected at least " +
                               std::to_string(required_size) + " elements, got " +
                               std::to_string(mel.data.size()));
    }

    const size_t src_stride = mel.frames;
    const size_t dst_stride = impl.encoder_expected_frames;
    for (size_t bin = 0; bin < MEL_BINS; ++bin) {
      const float* src = mel.data.data() + bin * src_stride;
      float* dst = mel_tensor.data<float>() + bin * dst_stride;
      std::copy(src, src + actual_frames, dst);
    }
  } else {
    // Fast path: use preprocessor tensor directly (no copy)
    mel_tensor = mel.mel_tensor;
    actual_frames = std::min(mel.frames, mel.time_steps);
  }

  // Create length tensor with correct element type
  ov::Tensor length_tensor;
  if (len_et == ov::element::i64) {
    length_tensor = ov::Tensor(ov::element::i64, {1});
    length_tensor.data<int64_t>()[0] = static_cast<int64_t>(actual_frames);
  } else {  // default to i32
    length_tensor = ov::Tensor(ov::element::i32, {1});
    length_tensor.data<int32_t>()[0] = static_cast<int32_t>(actual_frames);
  }

  // Set tensors and run inference
  impl.encoder_request.set_tensor(impl.encoder_ports.mel_in.value(), mel_tensor);
  impl.encoder_request.set_tensor(len_port, length_tensor);
  impl.encoder_request.infer();

  const auto encoder_tensor = impl.encoder_request.get_output_tensor(impl.encoder_output_index);  // encoder_output
  const auto encoder_length_tensor = impl.encoder_request.get_output_tensor(impl.encoder_length_index);  // encoder_output_length

  EncoderActivations activations;
  activations.hidden_size = impl.encoder_hidden_size;
  const auto shape = encoder_tensor.get_shape();
  if (shape.size() != 3 || shape[1] != impl.encoder_hidden_size) {
    throw std::runtime_error("Unexpected encoder output tensor shape");
  }
  activations.time_steps = shape[2];

  // Read encoder length (validated to be non-negative)
  const int64_t encoder_length = read_length_scalar(encoder_length_tensor);

  // Safe conversion: encoder_length is non-negative, so cast is safe
  // Use min to ensure valid_frames doesn't exceed time_steps
  activations.valid_frames = std::min(static_cast<size_t>(encoder_length), activations.time_steps);

  // Zero-copy: retain tensor and read directly downstream
  activations.tensor = encoder_tensor;
  return activations;
}

}  // namespace eddy::parakeet
