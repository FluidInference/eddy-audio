// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace eddy {
namespace audio {

/**
 * @brief Read audio file and convert to 16kHz mono float32 PCM
 *
 * Supports multiple formats via libsndfile: WAV, FLAC, OGG, AU, etc.
 * Automatically handles:
 * - Format conversion (int16/int24/int32 → float32)
 * - Stereo → mono mixing
 * - Sample rate conversion (44.1kHz/48kHz/etc → 16kHz)
 *
 * @param filename Path to audio file
 * @return Vector of float32 PCM samples normalized to [-1, 1] at 16kHz mono
 * @throws std::runtime_error if file cannot be opened or processed
 */
std::vector<float> read_wav(const std::string& filename);

/**
 * @brief Convert in-memory PCM16 buffer to float32 mono
 *
 * For in-memory buffers only. For file I/O, use read_wav() instead.
 *
 * @param data Pointer to int16 PCM data
 * @param size Number of samples (frames × channels)
 * @param channels Number of channels (1 or 2)
 * @return Vector of float32 PCM samples normalized to [-1, 1]
 * @throws std::runtime_error if channels not 1 or 2
 */
std::vector<float> pcm16_to_float32(const int16_t* data, size_t size, int channels = 1);

}  // namespace audio
}  // namespace eddy