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
[[nodiscard]] std::vector<float> read_wav(const std::string& filename);

}  // namespace audio
}  // namespace eddy