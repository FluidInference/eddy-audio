// Centralized fetch/ensure logic for Parakeet model files.

#include "eddy/models/parakeet/ensure_models.hpp"
#include "eddy/core/cache.hpp"

#include <cstdlib>
#include <sstream>
#include <system_error>
#include <iostream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace eddy::parakeet {

static bool file_nonempty(const std::filesystem::path& p) {
  std::error_code ec;
  return std::filesystem::exists(p, ec) && std::filesystem::is_regular_file(p, ec) && std::filesystem::file_size(p, ec) > 0;
}

static std::filesystem::path try_find_hf_fetch_models() {
#if defined(_WIN32)
  char exe_buf[MAX_PATH] = {0};
  DWORD n = GetModuleFileNameA(nullptr, exe_buf, static_cast<DWORD>(sizeof(exe_buf)));
  if (n > 0) {
    std::filesystem::path exe_dir = std::filesystem::path(exe_buf).parent_path();
    std::filesystem::path candidate = exe_dir / "hf_fetch_models.exe";
    if (std::filesystem::exists(candidate)) return candidate;
  }
  // Fallback: empty path implies use PATH
  return {};
#else
  // Non-Windows: rely on PATH
  return {};
#endif
}

bool ensure_models_available(const std::filesystem::path& target_dir,
                             std::string* last_error,
                             const std::vector<std::string>& required) {
  // Quick success path
  bool all_present = true;
  for (const auto& f : required) {
    if (!file_nonempty(target_dir / f)) { all_present = false; break; }
  }
  if (all_present) return true;

  // Honor opt-out: EDDY_DISABLE_AUTO_FETCH=1
  if (const char* dis = std::getenv("EDDY_DISABLE_AUTO_FETCH")) {
    if (std::string(dis) == "1") {
      if (last_error) *last_error = "auto-fetch disabled by EDDY_DISABLE_AUTO_FETCH=1";
      return false;
    }
  }

  std::error_code ec;
  std::filesystem::create_directories(target_dir, ec);

  // Prepare a comma-separated file list for missing files only
  std::vector<std::string> missing;
  for (const auto& f : required) {
    if (!file_nonempty(target_dir / f)) missing.push_back(f);
  }
  if (missing.empty()) return true;  // race; now present

  std::ostringstream files_opt;
  for (size_t i = 0; i < missing.size(); ++i) {
    if (i) files_opt << ',';
    files_opt << missing[i];
  }

  // Try invoking hf_fetch_models (bundled or via PATH)
  auto hf_path = try_find_hf_fetch_models();
  std::ostringstream cmd;
#if defined(_WIN32)
  if (!hf_path.empty()) {
    cmd << '"' << hf_path.string() << '"';
  } else {
    cmd << "hf_fetch_models";  // rely on PATH
  }
  cmd << " --files \"" << files_opt.str() << "\"";
#else
  (void)hf_path;
  cmd << "hf_fetch_models --files '" << files_opt.str() << "'";
#endif

  int rc = std::system(cmd.str().c_str());
  (void)rc;

  // Re-check
  for (const auto& f : required) {
    if (!file_nonempty(target_dir / f)) {
      if (last_error) {
        std::ostringstream msg;
        msg << "Missing after fetch attempt: " << f << ". Install models manually or run scripts/setup_parakeet_models.ps1.";
        *last_error = msg.str();
      }
      return false;
    }
  }
  return true;
}

}  // namespace eddy::parakeet

