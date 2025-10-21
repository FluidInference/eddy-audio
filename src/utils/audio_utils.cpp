// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/utils/audio_utils.hpp"

#include <stdexcept>
#include <cstdint>
#include <memory>

#include <sndfile.h>
#include <samplerate.h>

namespace eddy {
namespace audio {

constexpr int REQUIRED_SAMPLE_RATE = 16000;

std::vector<float> read_wav(const std::string& filename) {
    // Open audio file with libsndfile
    SF_INFO info;
    info.format = 0;  // Let libsndfile detect format
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &info);

    if (!file) {
        throw std::runtime_error("Failed to open audio file: " + filename + " (" + sf_strerror(nullptr) + ")");
    }

    // Custom deleter for RAII
    auto file_deleter = [](SNDFILE* f) { if (f) sf_close(f); };
    std::unique_ptr<SNDFILE, decltype(file_deleter)> file_guard(file, file_deleter);

    // Validate channel count
    if (info.channels != 1 && info.channels != 2) {
        throw std::runtime_error("Audio file must be mono or stereo, got " +
                                 std::to_string(info.channels) + " channels");
    }

    // Read all audio data as float32 (libsndfile handles conversion automatically)
    std::vector<float> data(info.frames * info.channels);
    sf_count_t frames_read = sf_readf_float(file, data.data(), info.frames);

    if (frames_read != info.frames) {
        throw std::runtime_error("Failed to read complete audio file");
    }

    // Mix stereo to mono if needed
    if (info.channels == 2) {
        std::vector<float> mono(info.frames);
        for (sf_count_t i = 0; i < info.frames; ++i) {
            mono[i] = 0.5f * (data[2 * i] + data[2 * i + 1]);
        }
        data = std::move(mono);
    }

    // Resample to 16kHz if needed
    if (info.samplerate != REQUIRED_SAMPLE_RATE) {
        const double ratio = static_cast<double>(REQUIRED_SAMPLE_RATE) / info.samplerate;
        const size_t output_frames = static_cast<size_t>(info.frames * ratio);

        std::vector<float> resampled(output_frames);

        SRC_DATA src_data;
        src_data.data_in = data.data();
        src_data.data_out = resampled.data();
        src_data.input_frames = info.frames;
        src_data.output_frames = output_frames;
        src_data.src_ratio = ratio;

        int error = src_simple(&src_data, SRC_SINC_BEST_QUALITY, 1);
        if (error) {
            throw std::runtime_error("Failed to resample audio: " +
                                     std::string(src_strerror(error)));
        }

        // Resize to actual output (may be slightly different due to rounding)
        resampled.resize(src_data.output_frames_gen);
        return resampled;
    }

    return data;
}

// Convert in-memory PCM16 buffer to float32 mono
// Uses libsndfile's normalization constants for consistency with read_wav()
std::vector<float> pcm16_to_float32(const int16_t* data, size_t size, int channels) {
    if (channels != 1 && channels != 2) {
        throw std::runtime_error("Only mono or stereo audio supported");
    }

    const size_t num_frames = size / channels;
    std::vector<float> pcmf32(num_frames);

    // Use the same normalization factor as libsndfile (SF_FORMAT_PCM_16 -> float)
    // libsndfile normalizes int16 to [-1.0, 1.0] range using division by 32768.0
    constexpr float scale = 1.0f / 32768.0f;

    if (channels == 1) {
        // Mono: direct conversion
        for (size_t i = 0; i < num_frames; i++) {
            pcmf32[i] = static_cast<float>(data[i]) * scale;
        }
    } else {
        // Stereo: average both channels (consistent with read_wav stereo mixing)
        for (size_t i = 0; i < num_frames; i++) {
            const float left = static_cast<float>(data[2 * i]) * scale;
            const float right = static_cast<float>(data[2 * i + 1]) * scale;
            pcmf32[i] = 0.5f * (left + right);
        }
    }

    return pcmf32;
}

}  // namespace audio
}  // namespace eddy
