// Centralized helper to check if Parakeet OpenVINO model files exist in the target directory.

#pragma once

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
bool ensure_models_available(const std::filesystem::path& target_dir,
                             std::string* last_error = nullptr,
                             const std::vector<std::string>& required = {
                                 "parakeet_melspectogram.xml",
                                 "parakeet_melspectogram.bin",
                                 "parakeet_encoder.xml",
                                 "parakeet_encoder.bin",
                                 "parakeet_decoder.xml",
                                 "parakeet_decoder.bin",
                                 "parakeet_joint.xml",
                                 "parakeet_joint.bin",
                                 "parakeet_vocab.json",
                             });

}  // namespace eddy::parakeet

