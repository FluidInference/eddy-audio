// Centralized check for Parakeet model files.

#include "eddy/utils/ensure_models.hpp"

#include <sstream>
#include <system_error>

namespace eddy::parakeet {

static bool file_nonempty(const std::filesystem::path& p) {
  std::error_code ec;
  return std::filesystem::exists(p, ec) &&
         std::filesystem::is_regular_file(p, ec) &&
         std::filesystem::file_size(p, ec) > 0;
}

bool check_models_available(const std::filesystem::path& target_dir,
                            std::string* last_error,
                            const std::vector<std::string>& required) {
  // Check if all required files exist
  std::vector<std::string> missing;
  for (const auto& f : required) {
    if (!file_nonempty(target_dir / f)) {
      missing.push_back(f);
    }
  }

  if (missing.empty()) {
    return true;
  }

  // Build error message with missing files
  if (last_error) {
    std::ostringstream msg;
    msg << "Missing model files in " << target_dir.string() << ": ";
    for (size_t i = 0; i < missing.size(); ++i) {
      if (i > 0) msg << ", ";
      msg << missing[i];
    }
    msg << ". Download models using 'hf_fetch_models' or from HuggingFace.";
    *last_error = msg.str();
  }

  return false;
}

}  // namespace eddy::parakeet
