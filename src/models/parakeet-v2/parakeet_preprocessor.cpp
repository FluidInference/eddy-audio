#include "eddy/models/parakeet-v2/parakeet_preprocessor.hpp"
#include "eddy/models/parakeet-v2/detail/parakeet_impl.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace eddy::parakeet {

namespace {

// Read a single scalar length value from a tensor that may be i32 or i64.
// Validates that the length is non-negative.
[[nodiscard]] int64_t read_length_scalar(const ov::Tensor& t) {
  const auto et = t.get_element_type();
  int64_t value;

  if (et == ov::element::i64) {
    value = t.data<int64_t>()[0];
  } else if (et == ov::element::i32) {
    int32_t val32 = t.data<int32_t>()[0];
    if (val32 < 0) {
      throw std::runtime_error("Preprocessor length is negative: " + std::to_string(val32));
    }
    value = static_cast<int64_t>(val32);
  } else {
    throw std::runtime_error("Unsupported length tensor element type");
  }

  if (value < 0) {
    throw std::runtime_error("Preprocessor length is negative: " + std::to_string(value));
  }

  return value;
}

}  // namespace

MelFeatures run_preprocessor(ParakeetImpl& impl, const AudioSegment& segment) {
  if (segment.pcm.empty()) {
    throw std::invalid_argument("Audio segment contains no PCM samples");
  }
  if (segment.sample_rate != 16000) {
    throw std::invalid_argument("Parakeet OpenVINO pipeline expects 16 kHz audio samples");
  }

  // Workaround for v3 preprocessor bug: round sample count to nearest 1000
  // v3 fails on specific odd-length audio (e.g., 240,135 samples)
  AudioSegment working_segment = segment;
  const size_t original_size = segment.pcm.size();
  const size_t round_to = 1000;
  const size_t rounded_size = ((original_size + round_to - 1) / round_to) * round_to;

  if (rounded_size != original_size) {
    working_segment.pcm.resize(rounded_size, 0.0F);  // Pad with zeros
  }

  // Query preprocessor window size from compiled model shape.
  // The Parakeet ONNX model always has static shape [1, 160000], but OpenVINO
  // may compile it with dynamic shapes for optimization (especially on CPU).
  //
  // window_samples = 160000 → Static compilation, use fixed-size windows
  // window_samples = 0      → Dynamic compilation, create tensor sized to actual audio
  //
  // Note: We cannot assume audio is pre-padded because:
  // 1. User audio files can be any length
  // 2. Long audio (>160k samples) requires chunking
  // 3. Padding happens inside run_window() to handle both cases
  size_t window_samples = 0;
  const auto pshape = impl.preproc_model.input(0).get_partial_shape();

  if (pshape.rank().is_static() && pshape.rank().get_length() >= 2) {
    const auto len_dim = pshape[1];
    if (len_dim.is_static() && len_dim.get_length() > 0) {
      window_samples = static_cast<size_t>(len_dim.get_length());
    }
  }

  // Enforce maximum window size for dynamic models
  // v3 preprocessor has dynamic shape - use same 10s windows as v2
  if (window_samples == 0 && working_segment.pcm.size() > 160000) {
    window_samples = 160000;  // 10 seconds at 16kHz (matches v2)
  }

  // Query length input type once (fixed at model export)
  const auto len_et = impl.preproc_model.input(1).get_element_type();
  const bool use_i64 = (len_et == ov::element::i64);

  MelFeatures features;

  auto run_window = [&](const float* pcm_ptr, size_t pcm_count) -> std::pair<ov::Tensor, ov::Tensor> {
    const size_t req = window_samples > 0 ? window_samples : pcm_count;

    // Create zero-padded audio tensor
    ov::Tensor audio_signal(ov::element::f32, {1, req});
    std::fill(audio_signal.data<float>(), audio_signal.data<float>() + req, 0.0F);

    const size_t samples_to_copy = std::min(req, pcm_count);
    if (samples_to_copy > 0) {
      std::copy(pcm_ptr, pcm_ptr + samples_to_copy, audio_signal.data<float>());
    }

    // Create length tensor matching the model's expected input type
    // Type is determined at model export time (queried once at line 69):
    //   - i64: Common in PyTorch → ONNX → OpenVINO conversions
    //   - i32: Common in TensorFlow → OpenVINO conversions
    // OpenVINO requires exact type matching - wrong type will cause inference failure
    ov::Tensor audio_length;
    if (use_i64) {
      audio_length = ov::Tensor(ov::element::i64, {1});
      audio_length.data<int64_t>()[0] = static_cast<int64_t>(samples_to_copy);
    } else {
      audio_length = ov::Tensor(ov::element::i32, {1});
      audio_length.data<int32_t>()[0] = static_cast<int32_t>(samples_to_copy);
    }

    impl.preproc_request.set_input_tensor(0, audio_signal);
    impl.preproc_request.set_input_tensor(1, audio_length);
    impl.preproc_request.infer();

    return {
      impl.preproc_request.get_output_tensor(0),
      impl.preproc_request.get_output_tensor(1)
    };
  };

  // ========================================
  // Single-shot path: Dynamically compiled model OR short audio that fits in one window
  // ========================================
  if (window_samples == 0 || working_segment.pcm.size() <= window_samples) {
    auto [mel_tensor, length_tensor] = run_window(working_segment.pcm.data(), working_segment.pcm.size());

    const int64_t valid_frames = read_length_scalar(length_tensor);
    if (valid_frames <= 0) {
      throw std::runtime_error("Preprocessor returned zero mel frames");
    }

    const auto mel_shape = mel_tensor.get_shape();
    if (mel_shape.size() != 3 || mel_shape[1] != 128) {
      throw std::runtime_error("Unexpected mel tensor shape from preprocessor");
    }

    const size_t mel_bins = mel_shape[1];
    const size_t time_steps = mel_shape[2];

    // Check for overflow in mel_bins * time_steps
    if (mel_bins > 0 && time_steps > SIZE_MAX / mel_bins) {
      throw std::runtime_error("Mel tensor too large: " + std::to_string(mel_bins) +
                               " bins × " + std::to_string(time_steps) + " frames would overflow");
    }
    const size_t elements = mel_bins * time_steps;

    // Retain tensors for zero-copy into encoder when shapes match
    features.mel_tensor = mel_tensor;
    features.length_tensor = length_tensor;
    features.time_steps = time_steps;
    features.frames = static_cast<size_t>(valid_frames);

    // Keep host copy for chunking operations
    features.data.resize(elements);
    std::copy(mel_tensor.data<float>(), mel_tensor.data<float>() + elements, features.data.begin());

  // ========================================
  // Windowed path: long audio, process in chunks
  // ========================================
  } else {
    constexpr size_t kMelBins = 128;
    const size_t total_samples = working_segment.pcm.size();

    std::vector<std::vector<float>> mel_bins(kMelBins);
    size_t offset = 0;
    size_t total_frames = 0;

    while (offset < total_samples) {
      const size_t remaining = total_samples - offset;
      const size_t this_count = std::min(window_samples, remaining);

      auto [mel_tensor, length_tensor] = run_window(working_segment.pcm.data() + offset, this_count);
      const int64_t vframes = read_length_scalar(length_tensor);

      if (vframes <= 0) {
        offset += this_count;
        continue;
      }

      const auto mel_shape = mel_tensor.get_shape();
      if (mel_shape.size() != 3 || mel_shape[1] != kMelBins) {
        throw std::runtime_error("Unexpected mel tensor shape from preprocessor");
      }

      const size_t time_steps = mel_shape[2];
      const size_t frames_to_append = static_cast<size_t>(std::min<int64_t>(vframes, static_cast<int64_t>(time_steps)));

      // Append frames for each mel bin (layout: [bin][time])
      const float* src_base = mel_tensor.data<float>();
      for (size_t bin = 0; bin < kMelBins; ++bin) {
        const float* src = src_base + bin * time_steps;
        mel_bins[bin].insert(mel_bins[bin].end(), src, src + frames_to_append);
      }

      total_frames += frames_to_append;
      offset += this_count;
    }

    if (total_frames == 0) {
      throw std::runtime_error("Preprocessor produced no frames for long audio");
    }

    // Check for overflow in kMelBins * total_frames
    if (total_frames > SIZE_MAX / kMelBins) {
      throw std::runtime_error("Mel tensor too large: " + std::to_string(kMelBins) +
                               " bins × " + std::to_string(total_frames) + " frames would overflow");
    }

    // Flatten into time-major buffer [bin][time]
    std::vector<float> mel_concat(kMelBins * total_frames);
    for (size_t bin = 0; bin < kMelBins; ++bin) {
      std::copy(mel_bins[bin].begin(), mel_bins[bin].end(), mel_concat.data() + bin * total_frames);
    }

    features.frames = total_frames;
    features.time_steps = total_frames;
    features.data = std::move(mel_concat);
  }

  return features;
}

}  // namespace eddy::parakeet
