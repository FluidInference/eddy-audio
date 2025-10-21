// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>

namespace eddy {

// New names (rebrand)
std::filesystem::path get_app_data_dir();
std::filesystem::path get_models_dir();
std::filesystem::path get_model_dir(const std::string& model_name);
std::filesystem::path get_model_assets_dir(const std::string& model_name);

// Backward-compat aliases (deprecated)
std::filesystem::path get_cache_dir();
std::filesystem::path get_model_cache_dir(const std::string& model_name);
std::filesystem::path get_model_files_dir(const std::string& model_name);

bool ensure_directory(const std::filesystem::path& path);

}  // namespace eddy
