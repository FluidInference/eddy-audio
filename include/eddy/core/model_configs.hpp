// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <map>

namespace eddy {

// Model configuration for different Parakeet versions
struct ModelConfig {
    std::string repo_id;
    std::vector<std::string> required_files;
    std::string cache_subdir;
};

// Available model configurations (similar to FluidAudio's ModelNames.swift)
namespace model_configs {

    inline const ModelConfig PARAKEET_V2 = {
        .repo_id = "FluidInference/parakeet-tdt-0.6b-v2-ov",
        .required_files = {
            "parakeet_encoder.xml", "parakeet_encoder.bin",
            "parakeet_decoder.xml", "parakeet_decoder.bin",
            "parakeet_joint.xml", "parakeet_joint.bin",
            "parakeet_melspectogram.xml", "parakeet_melspectogram.bin",
            "parakeet_vocab.json"
        },
        .cache_subdir = "parakeet-v2"
    };

    inline const ModelConfig PARAKEET_V3 = {
        .repo_id = "FluidInference/parakeet-tdt-0.6b-v3-ov",
        .required_files = {
            "parakeet_encoder.xml", "parakeet_encoder.bin",
            "parakeet_decoder.xml", "parakeet_decoder.bin",
            "parakeet_joint.xml", "parakeet_joint.bin",
            "parakeet_melspectogram.xml", "parakeet_melspectogram.bin",
            "parakeet_vocab.json"
        },
        .cache_subdir = "parakeet-v3"
    };

    // Model name lookup map
    inline const std::map<std::string, ModelConfig> MODEL_MAP = {
        {"v2", PARAKEET_V2},
        {"v3", PARAKEET_V3},
        {"parakeet-v2", PARAKEET_V2},
        {"parakeet-v3", PARAKEET_V3},
    };

    // Default model
    inline const ModelConfig& DEFAULT = PARAKEET_V2;

} // namespace model_configs

} // namespace eddy
