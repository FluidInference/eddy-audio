#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"
#include "parakeet_openvino_impl.hpp"
#include "eddy/models/parakeet-v2/parakeet_preprocessor.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace eddy::parakeet {

namespace {

// Read a single scalar length value from a tensor that may be i32 or i64.
int64_t read_length_scalar(const ov::Tensor& t) {
  const auto et = t.get_element_type();
  if (et == ov::element::i64) {
    return t.data<int64_t>()[0];
  } else if (et == ov::element::i32) {
    return static_cast<int64_t>(t.data<int32_t>()[0]);
  } else {
    throw std::runtime_error("Encoder length tensor has unexpected element type (expected i32 or i64)");
  }
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
    mel_tensor = ov::Tensor(ov::element::f32, {1, 128, impl.encoder_expected_frames});
    std::fill(mel_tensor.data<float>(), mel_tensor.data<float>() + mel_tensor.get_size(), 0.0F);

    actual_frames = std::min(impl.encoder_expected_frames, mel.frames);
    const size_t src_stride = mel.frames;
    const size_t dst_stride = impl.encoder_expected_frames;
    for (size_t bin = 0; bin < 128; ++bin) {
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
  activations.valid_frames = static_cast<size_t>(std::min<int64_t>(read_length_scalar(encoder_length_tensor), static_cast<int64_t>(activations.time_steps)));

  // Zero-copy: retain tensor and read directly downstream
  activations.tensor = encoder_tensor;
  return activations;
}

}  // namespace eddy::parakeet
