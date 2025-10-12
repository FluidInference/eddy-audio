// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>

namespace eddy {

/**
 * @brief Get the application cache directory
 * @return Path to cache directory (e.g., C:\Users\user\AppData\Local\eddy\cache\)
 */
std::filesystem::path get_cache_dir();

/**
 * @brief Get the cache directory for a specific model (for OpenVINO compiled cache)
 * @param model_name Model identifier (e.g., "parakeet-v2", "whisper-large")
 * @return Path to model cache directory
 */
std::filesystem::path get_model_cache_dir(const std::string& model_name);

/**
 * @brief Get the directory where model files (.xml, .json, .bin) are stored
 * @param model_name Model identifier (e.g., "parakeet-v2", "whisper-large")
 * @return Path to model files directory
 */
std::filesystem::path get_model_files_dir(const std::string& model_name);

/**
 * @brief Ensure cache directory exists
 * @param path Directory path to create
 * @return true if directory exists or was created successfully
 */
bool ensure_cache_dir(const std::filesystem::path& path);

}  // namespace eddy
