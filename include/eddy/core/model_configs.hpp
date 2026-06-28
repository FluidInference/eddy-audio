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
    std::string repo_subdir;                  // Optional subfolder within the repo (e.g., "fp16");
                                              // files download from <repo>/resolve/main/<repo_subdir>/<file>
                                              // but are stored flat in the cache. Empty => repo root.
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

    inline const ModelConfig PARAKEET_V2 = {
        .repo_id = "FluidInference/parakeet-tdt-0.6b-v2-ov",
        .required_files = PARAKEET_STANDARD_FILES,
        .cache_subdir = "parakeet-v2"
    };

    inline const ModelConfig PARAKEET_V3 = {
        .repo_id = "FluidInference/parakeet-tdt-1.1b-v3-ov",
        .required_files = PARAKEET_STANDARD_FILES,
        .cache_subdir = "parakeet-v3"
    };

    // NVIDIA Nemotron-3.5-ASR-Streaming-Multilingual 0.6B (cache-aware
    // streaming FastConformer-RNNT, prompt-conditioned multilingual).
    // Distinct file set + metadata.json (cache shapes, prompt_dictionary,
    // lang_tag_token_ids) consumed by the eddy::nemotron backend.
    inline const std::vector<std::string> NEMOTRON_FILES = {
        "nemotron_encoder.xml", "nemotron_encoder.bin",
        "nemotron_decoder.xml", "nemotron_decoder.bin",
        "nemotron_joint.xml", "nemotron_joint.bin",
        "nemotron_preprocessor.xml", "nemotron_preprocessor.bin",
        "nemotron_vocab.json", "metadata.json"
    };

    inline const ModelConfig NEMOTRON_STREAMING = {
        .repo_id = "FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-ov",
        .required_files = NEMOTRON_FILES,
        .cache_subdir = "nemotron-streaming",
        // FP16 IR (identical transcripts to FP32, ~half the size, NPU-friendly).
        // FP32 also available under the "fp32" subfolder of the same repo.
        .repo_subdir = "fp16"
    };

    // Model name lookup map
    inline const std::map<std::string, ModelConfig> MODEL_MAP = {
        {"parakeet-v2", PARAKEET_V2},
        {"parakeet-v3", PARAKEET_V3},
        {"nemotron-streaming", NEMOTRON_STREAMING}
    };

    // Default model
    inline const ModelConfig DEFAULT = PARAKEET_V2;

} // namespace model_configs

} // namespace eddy
