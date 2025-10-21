#include "eddy/models/parakeet-v2/parakeet_preprocessor.hpp"
#include "parakeet_openvino_impl.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace eddy::parakeet {

namespace {

int64_t read_length_scalar(const ov::Tensor& t) {
  const auto et = t.get_element_type();

  if (et == ov::element::i64) {
    return t.data<int64_t>()[0];
  }

  if (et == ov::element::i32) {
    return static_cast<int64_t>(t.data<int32_t>()[0]);
  }

  throw std::runtime_error("Unsupported length tensor element type");
}

}  // namespace

MelFeatures run_preprocessor(ParakeetImpl& impl, const AudioSegment& segment) {
  if (segment.pcm.empty()) {
    throw std::invalid_argument("Audio segment contains no PCM samples");
  }
  if (segment.sample_rate != 16000) {
    throw std::invalid_argument("Parakeet OpenVINO pipeline expects 16 kHz audio samples");
  }

  // Detect static preprocessor input length (window size).
  // 0 = dynamic length supported, >0 = fixed window size (process in chunks)
  size_t window_samples = 0;
  const auto pshape = impl.preproc_model.input(0).get_partial_shape();

  if (pshape.rank().is_static() && pshape.rank().get_length() >= 2) {
    const auto len_dim = pshape[1];
    if (len_dim.is_static() && len_dim.get_length() > 0) {
      window_samples = static_cast<size_t>(len_dim.get_length());
    }
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

    // Create length tensor with correct type
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
  // Single-shot path: dynamic model or short audio
  // ========================================
  if (window_samples == 0 || segment.pcm.size() <= window_samples) {
    auto [mel_tensor, length_tensor] = run_window(segment.pcm.data(), segment.pcm.size());

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
    const size_t total_samples = segment.pcm.size();

    std::vector<std::vector<float>> mel_bins(kMelBins);
    size_t offset = 0;
    size_t total_frames = 0;

    while (offset < total_samples) {
      const size_t remaining = total_samples - offset;
      const size_t this_count = std::min(window_samples, remaining);

      auto [mel_tensor, length_tensor] = run_window(segment.pcm.data() + offset, this_count);
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
