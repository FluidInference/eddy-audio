#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"
#include "parakeet_openvino_impl.hpp"
#include "eddy/models/parakeet-v2/parakeet_preprocessor.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace eddy::parakeet {

namespace {

// Try to get an input/output port by any of the provided names; if none found,
// return an empty optional.
std::optional<ov::Output<const ov::Node>> try_input_by_names(const ov::CompiledModel& model,
                                                             const std::vector<std::string>& names) {
  for (const auto& name : names) {
    try {
      return model.input(name);
    } catch (...) {
      // ignore and continue
    }
  }
  return std::nullopt;
}

// Read a single scalar length value from a tensor that may be i32 or i64.
int64_t read_length_scalar(const ov::Tensor& t) {
  const auto et = t.get_element_type();
  if (et == ov::element::i64) {
    return t.data<int64_t>()[0];
  } else if (et == ov::element::i32) {
    return static_cast<int64_t>(t.data<int32_t>()[0]);
  } else {
    // Fallback: try to interpret as i64
    return t.data<int64_t>()[0];
  }
}

}  // namespace

EncoderPorts select_encoder_ports(const ov::CompiledModel& model) {
  EncoderPorts ports;

  // Prefer common friendly names first
  ports.mel_in = try_input_by_names(model, {"melspectogram", "melspectrogram", "mel", "melspec"});
  ports.len_in = try_input_by_names(model, {"melspectogram_length", "melspectrogram_length", "mel_length", "length"});

  // Fallback: scan inputs by shape
  if (!ports.mel_in.has_value() || !ports.len_in.has_value()) {
    for (const auto& p : model.inputs()) {
      try {
        const auto pshape = p.get_partial_shape();
        if (!ports.mel_in.has_value()) {
          if (pshape.rank().is_static() && pshape.rank().get_length() == 3) {
            const auto d1 = pshape[1];
            if (d1.is_static() && d1.get_length() == 128) {
              ports.mel_in = p;
              continue;
            }
          }
        }
        if (!ports.len_in.has_value()) {
          if (pshape.rank().is_static() && pshape.rank().get_length() == 1) {
            ports.len_in = p;
            continue;
          }
        }
      } catch (...) {
        // ignore this port
      }
    }
  }

  // Resolve encoder output: name first, then by shape
  try {
    ports.enc_out = model.output("encoder_output");
  } catch (...) {
    for (const auto& p : model.outputs()) {
      try {
        const auto shape = p.get_partial_shape();
        if (shape.rank().is_static() && shape.rank().get_length() == 3) {
          ports.enc_out = p;
          break;
        }
      } catch (...) {
        // keep searching
      }
    }
  }

  return ports;
}

EncoderActivations run_encoder(ParakeetImpl& impl, const MelFeatures& mel) {
  if (impl.encoder_expected_frames == 0) {
    throw std::runtime_error("Encoder expected frame count is zero");
  }

  // Determine encoder length input element type to avoid device-specific mismatches (e.g., NPU strict typing)
  const auto len_port = impl.encoder_ports.len_in.value();
  const auto len_et = len_port.get_element_type();

  // Fast path: if preprocessor produced exactly the expected time axis, bind tensors directly
  if (mel.time_steps == impl.encoder_expected_frames && mel.mel_tensor) {
    // Bind mel using resolved port
    impl.encoder_request.set_tensor(impl.encoder_ports.mel_in.value(), mel.mel_tensor);
    // Create a length tensor with the encoder's required element type
    const size_t frames_for_len = std::min(mel.frames, mel.time_steps);
    ov::Tensor enc_len_tensor;
    if (len_et == ov::element::i64) {
      enc_len_tensor = ov::Tensor(ov::element::i64, {1});
      enc_len_tensor.data<int64_t>()[0] = static_cast<int64_t>(frames_for_len);
    } else {  // default to i32
      enc_len_tensor = ov::Tensor(ov::element::i32, {1});
      enc_len_tensor.data<int32_t>()[0] = static_cast<int32_t>(frames_for_len);
    }
    impl.encoder_request.set_tensor(len_port, enc_len_tensor);
  } else {
    // Fallback: copy into a padded/trimmed tensor matching encoder's expected frames
    ov::Tensor mel_tensor(ov::element::f32, {1, 128, impl.encoder_expected_frames});
    std::fill(mel_tensor.data<float>(), mel_tensor.data<float>() + mel_tensor.get_size(), 0.0F);

    const size_t frames_to_copy = std::min(impl.encoder_expected_frames, mel.frames);
    const size_t src_stride = mel.frames;
    const size_t dst_stride = impl.encoder_expected_frames;
    for (size_t bin = 0; bin < 128; ++bin) {
      const float* src = mel.data.data() + bin * src_stride;
      float* dst = mel_tensor.data<float>() + bin * dst_stride;
      std::copy(src, src + frames_to_copy, dst);
    }

    // Length tensor with correct element type
    ov::Tensor mel_length;
    if (len_et == ov::element::i64) {
      mel_length = ov::Tensor(ov::element::i64, {1});
      mel_length.data<int64_t>()[0] = static_cast<int64_t>(frames_to_copy);
    } else {
      mel_length = ov::Tensor(ov::element::i32, {1});
      mel_length.data<int32_t>()[0] = static_cast<int32_t>(frames_to_copy);
    }

    impl.encoder_request.set_tensor(impl.encoder_ports.mel_in.value(), mel_tensor);
    impl.encoder_request.set_tensor(impl.encoder_ports.len_in.value(), mel_length);
  }
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
