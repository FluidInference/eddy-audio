// Centralized helper to check if Parakeet OpenVINO model files exist in the target directory.

#pragma once

#include "eddy/core/model_configs.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace eddy::parakeet {

// Checks if all required model files exist in target_dir.
// Returns true if all files are present, false otherwise.
// Optionally fills last_error with a descriptive message on failure.
//
// Note: This function only checks for existence - it does NOT automatically download models.
// Users should download models manually using hf_fetch_models or from HuggingFace.
[[nodiscard]] bool check_models_available(
    const std::filesystem::path& target_dir,
    std::string* last_error = nullptr,
    const std::vector<std::string>& required = eddy::model_configs::PARAKEET_STANDARD_FILES);

}  // namespace eddy::parakeet

