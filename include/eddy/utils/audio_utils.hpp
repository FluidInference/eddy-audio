// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace eddy {
namespace audio {

/**
 * @brief Read a WAV file and convert to float32 PCM format
 * @param filename Path to WAV file (must be 16kHz mono or stereo)
 * @return Vector of float32 PCM samples normalized to [-1, 1]
 */
std::vector<float> read_wav(const std::string& filename);

/**
 * @brief Read raw PCM data from buffer
 * @param data Pointer to int16 PCM data
 * @param size Number of samples
 * @param channels Number of channels (1 or 2)
 * @return Vector of float32 PCM samples normalized to [-1, 1]
 */
std::vector<float> pcm16_to_float32(const int16_t* data, size_t size, int channels = 1);

}  // namespace audio
}  // namespace eddy