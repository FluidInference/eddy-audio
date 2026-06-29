// Centralized helper to check and download OpenVINO model files.
// Model-agnostic: operates on any eddy::ModelConfig (Parakeet, Nemotron, ...).

#pragma once

#include "eddy/core/model_configs.hpp"
#include <filesystem>
#include <string>
#include <vector>
#include <functional>

namespace eddy::model_utils {

// Checks if all required model files exist in target_dir.
// Returns true if all files are present, false otherwise.
// Optionally fills last_error with a descriptive message on failure.
[[nodiscard]] bool check_models_available(
    const std::filesystem::path& target_dir,
    std::string* last_error = nullptr,
    const std::vector<std::string>& required = eddy::model_configs::PARAKEET_STANDARD_FILES);

// Progress callback: (filename, current_file_index, total_files) -> void
using DownloadProgressCallback = std::function<void(const std::string&, int, int)>;

// Downloads model files from HuggingFace to target directory.
// Returns true if all files downloaded successfully, false otherwise.
//
// Parameters:
//   - config: Model configuration (repo_id, required_files)
//   - target_dir: Directory to download files to (will be created if needed)
//   - last_error: Optional pointer to receive error message on failure
//   - progress_callback: Optional callback for progress updates
//   - skip_existing: If true, skip files that already exist (default: true)
//
// Note: Uses curl to download files. Curl must be available in PATH.
[[nodiscard]] bool download_models(
    const eddy::ModelConfig& config,
    const std::filesystem::path& target_dir,
    std::string* last_error = nullptr,
    DownloadProgressCallback progress_callback = nullptr,
    bool skip_existing = true);

}  // namespace eddy::model_utils

