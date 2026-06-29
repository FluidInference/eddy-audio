// Centralized check and download for OpenVINO model files (model-agnostic).

#include "eddy/utils/ensure_models.hpp"

#include <cctype>
#include <sstream>
#include <system_error>
#include <cstdlib>

namespace eddy::model_utils {

static bool file_nonempty(const std::filesystem::path& p) {
  std::error_code ec;
  return std::filesystem::exists(p, ec) &&
         std::filesystem::is_regular_file(p, ec) &&
         std::filesystem::file_size(p, ec) > 0;
}

// The download shells out via std::system, so any interpolated component must
// be free of characters that could break out of the double-quoted argument.
// Today every field is a compile-time constant, but ModelConfig is caller-
// supplied, so reject anything outside a conservative charset rather than risk
// command injection. The long-term fix is to drop std::system for a direct
// libcurl call and eliminate this class of concern entirely.
//
// The URL and the local output path get *separate* allowlists: the URL is the
// real injection surface (it's assembled from caller-supplied ModelConfig
// fields) so it stays tight — only the characters an https HuggingFace URL
// needs, and notably no '~'. The output path is application-controlled (the
// Eddy cache dir) but must also tolerate native Windows paths, so it
// additionally allows the native separator '\\' and spaces (the drive-letter
// ':' is already covered by the shared charset).
static bool charset_ok(const std::string& s, bool allow_path_chars) {
  for (const unsigned char c : s) {
    bool ok = std::isalnum(c) || c == '.' || c == '_' || c == '-' ||
              c == '/' || c == ':';
    if (!ok && allow_path_chars) {
      // Native Windows paths: backslash separators and spaces inside the
      // double-quoted argument. None of these can break out of the quotes.
      ok = (c == '\\' || c == ' ');
    }
    if (!ok) return false;
  }
  return true;
}

static bool is_url_safe(const std::string& s) { return charset_ok(s, /*allow_path_chars=*/false); }
static bool is_path_safe(const std::string& s) { return charset_ok(s, /*allow_path_chars=*/true); }

static bool download_single_file(const std::string& url,
                                  const std::filesystem::path& output_path,
                                  std::string* error_msg = nullptr) {
  // Refuse to build a shell command from unsafe components.
  if (!is_url_safe(url) || !is_path_safe(output_path.string())) {
    if (error_msg) {
      *error_msg = "Refusing to download: unsafe characters in URL or path: " + url;
    }
    return false;
  }
  // The charset permits '.' and '/', so reject ".." components explicitly to
  // stop a caller-supplied filename from writing outside the target directory.
  for (const auto& part : output_path) {
    if (part == "..") {
      if (error_msg) *error_msg = "Refusing to download: path traversal in " + output_path.string();
      return false;
    }
  }

  // Create parent directory
  std::error_code ec;
  std::filesystem::create_directories(output_path.parent_path(), ec);
  if (ec) {
    if (error_msg) {
      *error_msg = "Failed to create directory: " + ec.message();
    }
    return false;
  }

  // Build curl command (curl must be in PATH)
  std::string curl_cmd = "curl -L --progress-bar --fail \"" + url + "\" -o \"" +
                         output_path.string() + "\"";

  // Execute download
  int ret = std::system(curl_cmd.c_str());
  if (ret != 0) {
    // Remove any partial file: an interrupted transfer leaves a truncated file
    // that file_nonempty() would later accept, silently skipping a re-download
    // and loading a corrupt model.
    std::filesystem::remove(output_path, ec);
    if (error_msg) {
      *error_msg = "curl failed with exit code " + std::to_string(ret) + " for URL: " + url;
    }
    return false;
  }

  // Verify downloaded file
  auto size = std::filesystem::file_size(output_path, ec);
  if (ec || size == 0) {
    std::filesystem::remove(output_path, ec);
    if (error_msg) {
      *error_msg = "Downloaded file is missing or empty: " + output_path.string();
    }
    return false;
  }

  return true;
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
    msg << ". Use download_models() or download from HuggingFace.";
    *last_error = msg.str();
  }

  return false;
}

bool download_models(const eddy::ModelConfig& config,
                     const std::filesystem::path& target_dir,
                     std::string* last_error,
                     DownloadProgressCallback progress_callback,
                     bool skip_existing) {
  const auto& required_files = config.required_files;
  const int total_files = static_cast<int>(required_files.size());
  int current_file = 0;
  int downloaded = 0;
  int skipped = 0;
  int failed = 0;

  for (const auto& filename : required_files) {
    current_file++;
    const std::filesystem::path file_path = target_dir / filename;

    // Skip if file already exists and skip_existing is true
    if (skip_existing && file_nonempty(file_path)) {
      skipped++;
      if (progress_callback) {
        progress_callback(filename + " (skipped)", current_file, total_files);
      }
      continue;
    }

    // Construct HuggingFace URL. When repo_subdir is set, files live in a
    // subfolder of the repo (e.g. "fp16/") but are still stored flat locally.
    const std::string remote_rel =
        config.repo_subdir.empty() ? filename : config.repo_subdir + "/" + filename;
    const std::string url = "https://huggingface.co/" + config.repo_id + "/resolve/main/" + remote_rel;

    // Notify progress
    if (progress_callback) {
      progress_callback(filename, current_file, total_files);
    }

    // Download file
    std::string download_error;
    if (download_single_file(url, file_path, &download_error)) {
      downloaded++;
    } else {
      failed++;
      if (last_error) {
        if (!last_error->empty()) {
          *last_error += "\n";
        }
        *last_error += "Failed to download " + filename + ": " + download_error;
      }
    }
  }

  // Return true only if no failures occurred
  if (failed > 0) {
    if (last_error && last_error->empty()) {
      *last_error = "Failed to download " + std::to_string(failed) + " file(s)";
    }
    return false;
  }

  return true;
}

}  // namespace eddy::model_utils
