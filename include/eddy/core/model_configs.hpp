// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <map>

namespace eddy {

/// Configuration for a specific model variant
struct ModelConfig {
    std::string repo_id;                      // HuggingFace repository ID (e.g., "org/model-name")
    std::vector<std::string> required_files;  // List of required model files (xml, bin, json)
    std::string cache_subdir;                 // Subdirectory name in cache (e.g., "parakeet-v2")
};

// Available model configurations (similar to FluidAudio's ModelNames.swift)
namespace model_configs {

    // Standard Parakeet model files (shared across versions)
    inline const std::vector<std::string> PARAKEET_STANDARD_FILES = {
        "parakeet_encoder.xml", "parakeet_encoder.bin",
        "parakeet_decoder.xml", "parakeet_decoder.bin",
        "parakeet_joint.xml", "parakeet_joint.bin",
        // Note: "melspectogram" spelling matches upstream HuggingFace repository
        "parakeet_melspectogram.xml", "parakeet_melspectogram.bin",
        "parakeet_vocab.json"
    };

    // Parakeet v3 requires the v3-specific vocabulary file
    inline const std::vector<std::string> PARAKEET_V3_FILES = {
        "parakeet_encoder.xml", "parakeet_encoder.bin",
        "parakeet_decoder.xml", "parakeet_decoder.bin",
        "parakeet_joint.xml", "parakeet_joint.bin",
        "parakeet_melspectogram.xml", "parakeet_melspectogram.bin",
        "parakeet_v3_vocab.json"
    };

    inline const ModelConfig PARAKEET_V2 = {
        .repo_id = "FluidInference/parakeet-tdt-0.6b-v2-ov",
        .required_files = PARAKEET_STANDARD_FILES,
        .cache_subdir = "parakeet-v2"
    };

    inline const ModelConfig PARAKEET_V3 = {
        .repo_id = "FluidInference/parakeet-tdt-0.6b-v3-ov",
        .required_files = PARAKEET_V3_FILES,
        .cache_subdir = "parakeet-v3"
    };

    // Model name lookup map
    inline const std::map<std::string, ModelConfig> MODEL_MAP = {
        {"parakeet-v2", PARAKEET_V2},
        {"parakeet-v3", PARAKEET_V3}
    };

    // Default model
    inline const ModelConfig DEFAULT = PARAKEET_V2;

} // namespace model_configs

} // namespace eddy
