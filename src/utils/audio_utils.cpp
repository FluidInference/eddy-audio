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

    // Validate audio dimensions to prevent overflow
    if (info.frames < 0) {
        throw std::runtime_error("Invalid audio file: negative frame count");
    }
    if (info.frames > 0 && static_cast<size_t>(info.channels) > SIZE_MAX / static_cast<size_t>(info.frames)) {
        throw std::runtime_error("Audio file too large: " + std::to_string(info.frames) +
                                 " frames × " + std::to_string(info.channels) + " channels would overflow");
    }

    // Read all audio data as float32 (libsndfile handles conversion automatically)
    std::vector<float> data(static_cast<size_t>(info.frames) * static_cast<size_t>(info.channels));
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
        // Validate sample rate to prevent division by zero
        if (info.samplerate <= 0) {
            throw std::runtime_error("Invalid audio file: sample rate must be positive, got " +
                                     std::to_string(info.samplerate));
        }

        const double ratio = static_cast<double>(REQUIRED_SAMPLE_RATE) / info.samplerate;
        const double output_frames_double = static_cast<double>(info.frames) * ratio;

        // Check for overflow when converting to size_t
        if (output_frames_double < 0 || output_frames_double > static_cast<double>(SIZE_MAX)) {
            throw std::runtime_error("Resampled audio too large: " + std::to_string(info.frames) +
                                     " frames × ratio " + std::to_string(ratio) + " would overflow");
        }

        const size_t output_frames = static_cast<size_t>(output_frames_double);

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

}  // namespace audio
}  // namespace eddy
