// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>

namespace eddy {

// New names (rebrand)
[[nodiscard]] std::filesystem::path get_app_data_dir();
[[nodiscard]] std::filesystem::path get_models_dir();
[[nodiscard]] std::filesystem::path get_model_dir(const std::string& model_name);
[[nodiscard]] std::filesystem::path get_model_assets_dir(const std::string& model_name);

// Backward-compat aliases (deprecated)
[[deprecated("Use get_app_data_dir() instead")]]
[[nodiscard]] std::filesystem::path get_cache_dir();

[[deprecated("Use get_model_dir() instead")]]
[[nodiscard]] std::filesystem::path get_model_cache_dir(const std::string& model_name);

[[deprecated("Use get_model_assets_dir() instead")]]
[[nodiscard]] std::filesystem::path get_model_files_dir(const std::string& model_name);

[[nodiscard]] bool ensure_directory(const std::filesystem::path& path);

}  // namespace eddy
