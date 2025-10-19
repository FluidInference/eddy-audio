// Centralized helper to ensure Parakeet OpenVINO model files exist in the Eddy cache.
// If files are missing, attempts a best-effort fetch using the bundled hf_fetch_models tool when available.

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace eddy::parakeet {

// Returns true if all required files are present after the call. On failure,
// returns false and optionally fills last_error.
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

