// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/pipelines/audio_utils.hpp"

#include <stdexcept>
#include <cstdint>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

namespace eddy {
namespace audio {

constexpr int REQUIRED_SAMPLE_RATE = 16000;

std::vector<float> read_wav(const std::string& filename) {
    drwav wav;
    if (!drwav_init_file(&wav, filename.c_str(), nullptr)) {
        throw std::runtime_error("Failed to open WAV file: " + filename);
    }

    if (wav.channels != 1 && wav.channels != 2) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be mono or stereo, got " + std::to_string(wav.channels) + " channels");
    }

    if (wav.sampleRate != REQUIRED_SAMPLE_RATE) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be 16kHz, got " + std::to_string(wav.sampleRate) + " Hz");
    }

  const uint64_t n = wav.totalPCMFrameCount;

  // Read as float32 directly to avoid intermediate int16 copy
  std::vector<float> buf(n * wav.channels);
  drwav_read_pcm_frames_f32(&wav, n, buf.data());
  drwav_uninit(&wav);

  // Convert to mono float32 (average if stereo)
  if (wav.channels == 1) return buf;

  const size_t frames = static_cast<size_t>(n);
  std::vector<float> mono(frames);
  for (size_t i = 0; i < frames; ++i) {
    mono[i] = 0.5f * (buf[2 * i] + buf[2 * i + 1]);
  }
  return mono;
}

std::vector<float> pcm16_to_float32(const int16_t* data, size_t size, int channels) {
    if (channels != 1 && channels != 2) {
        throw std::runtime_error("Only mono or stereo audio supported");
    }

    const size_t num_frames = size / channels;
    std::vector<float> pcmf32(num_frames);

    if (channels == 1) {
        // Mono: direct conversion
        for (size_t i = 0; i < num_frames; i++) {
            pcmf32[i] = static_cast<float>(data[i]) / 32768.0f;
        }
    } else {
        // Stereo: average both channels
        for (size_t i = 0; i < num_frames; i++) {
            pcmf32[i] = static_cast<float>(data[2 * i] + data[2 * i + 1]) / 65536.0f;
        }
    }

    return pcmf32;
}

}  // namespace audio
}  // namespace eddy
