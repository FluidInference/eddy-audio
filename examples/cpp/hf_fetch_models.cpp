// Minimal C++ Hugging Face model fetcher for Eddy Parakeet assets
// - Uses libcurl if available (HF_FETCH_HAVE_LIBCURL)
// - Otherwise falls back to invoking system 'curl -L'

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(HF_FETCH_HAVE_LIBCURL)
#include <curl/curl.h>
#endif

namespace fs = std::filesystem;

struct Options {
  std::string repo = "alexwengg/parakeet-tdt-0.6b-v2-ov";
  std::string revision = "main";
  fs::path target;  // resolved later (defaults to Eddy cache path)
  std::vector<std::string> files = {
      // Default to only the vocabulary file unless overridden via --files
      "parakeet_vocab.json",
  };
};

static fs::path default_cache_dir() {
#if defined(_WIN32)
  const char* local = std::getenv("LOCALAPPDATA");
  if (!local) {
    throw std::runtime_error("LOCALAPPDATA not set; cannot resolve cache directory on Windows");
  }
  return fs::path(local) / "eddy" / "cache" / "models" / "parakeet-v2" / "files";
#elif defined(__APPLE__)
  const char* home = std::getenv("HOME");
  if (!home) {
    throw std::runtime_error("HOME not set; cannot resolve cache directory on macOS");
  }
  return fs::path(home) / "Library" / "Caches" / "eddy" / "models" / "parakeet-v2" / "files";
#else
  if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
    return fs::path(xdg) / "eddy" / "models" / "parakeet-v2" / "files";
  }
  const char* home = std::getenv("HOME");
  fs::path base = home ? fs::path(home) / ".cache" : fs::path(".cache");
  return base / "eddy" / "models" / "parakeet-v2" / "files";
#endif
}

static std::string resolve_url(const std::string& repo, const std::string& rev, const std::string& file) {
  std::ostringstream oss;
  oss << "https://huggingface.co/" << repo << "/resolve/" << rev << "/" << file << "?download=true";
  return oss.str();
}

static bool file_nonempty(const fs::path& p) {
  std::error_code ec;
  return fs::exists(p, ec) && fs::file_size(p, ec) > 0;
}

#if defined(HF_FETCH_HAVE_LIBCURL)
static size_t curl_write(void* ptr, size_t size, size_t nmemb, void* userdata) {
  std::ofstream* out = static_cast<std::ofstream*>(userdata);
  out->write(static_cast<const char*>(ptr), static_cast<std::streamsize>(size * nmemb));
  return size * nmemb;
}

static bool download_via_libcurl(const std::string& url, const fs::path& dst) {
  CURL* curl = curl_easy_init();
  if (!curl) return false;

  // Determine if we can resume
  std::ios_base::openmode mode = std::ios::binary;
  long long resume_from = 0;
  std::error_code fec;
  if (fs::exists(dst, fec)) {
    resume_from = static_cast<long long>(fs::file_size(dst, fec));
    if (!fec && resume_from > 0) {
      mode |= std::ios::app;  // append to continue
    }
  }

  std::ofstream out(dst, mode);
  if (!out.is_open()) {
    curl_easy_cleanup(curl);
    return false;
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &curl_write);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);  // quiet
  // Timeouts
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);   // abort if below threshold for 60s
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);   // 1 byte/sec
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);  // no overall timeout
  // User agent
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "eddy-hf-fetch/1.0");

  if (resume_from > 0) {
    curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(resume_from));
  }

  CURLcode rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_easy_cleanup(curl);
  out.close();

  if (rc != CURLE_OK || http_code >= 400 || !file_nonempty(dst)) {
    return false;
  }
  return true;
}
#endif

static bool have_system_curl() {
#if defined(_WIN32)
  // Assume curl available on modern Windows; probing is non-trivial without spawning.
  // We'll try to run it and rely on exit code later.
  return true;
#else
  // Try `which curl`
  auto* p = std::popen("which curl 2>/dev/null", "r");
  if (!p) return false;
  char buf[8];
  size_t n = std::fread(buf, 1, sizeof(buf), p);
  std::pclose(p);
  return n > 0;
#endif
}

static bool download_via_system_curl(const std::string& url, const fs::path& dst) {
  std::ostringstream cmd;
#if defined(_WIN32)
  // PowerShell-compatible quoting; rely on bundled curl on Windows 10+
  cmd << "curl -sS -L --fail --retry 3 --retry-delay 1 -C - \"" << url << "\" -o \"" << dst.string() << "\"";
#else
  cmd << "curl -sS -L --fail --retry 3 --retry-delay 1 -C - '" << url << "' -o '" << dst.string() << "'";
#endif
  int code = std::system(cmd.str().c_str());
  (void)code;
  return file_nonempty(dst);
}

static void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--repo <id>] [--rev <rev>] [--target <dir>] [--files <comma-list>]" << '\n'
            << "Defaults: repo=alexwengg/parakeet-tdt-0.6b-v2-ov, rev=main, target=<eddy cache>, files=parakeet_vocab.json" << '\n';
}

int main(int argc, char** argv) {
  Options opt;

  // CLI parse (very lightweight)
  for (int i = 1; i < argc; ++i) {
    std::string_view a(argv[i]);
    auto need = [&](const char* name) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return false;
      }
      return true;
    };
    if (a == "-h" || a == "--help") {
      print_help(argv[0]);
      return 0;
    } else if (a == "--repo") {
      if (!need("--repo")) return 2;
      opt.repo = argv[++i];
    } else if (a == "--rev" || a == "--revision") {
      if (!need("--rev")) return 2;
      opt.revision = argv[++i];
    } else if (a == "--target") {
      if (!need("--target")) return 2;
      opt.target = fs::path(argv[++i]);
    } else if (a == "--files") {
      if (!need("--files")) return 2;
      opt.files.clear();
      std::string list = argv[++i];
      std::stringstream ss(list);
      std::string item;
      while (std::getline(ss, item, ',')) {
        if (!item.empty()) opt.files.push_back(item);
      }
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      print_help(argv[0]);
      return 2;
    }
  }

  try {
    if (opt.target.empty()) opt.target = default_cache_dir();
    fs::create_directories(opt.target);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: Unable to create target directory: " << e.what() << "\n";
    return 3;
  }

#if defined(HF_FETCH_HAVE_LIBCURL)
  curl_global_init(CURL_GLOBAL_DEFAULT);
#endif

  std::cout << "Downloading from repo: " << opt.repo << ", rev: " << opt.revision << "\n";
  std::cout << "Target directory: " << opt.target.string() << "\n";

  int failures = 0;
  for (const auto& f : opt.files) {
    const auto url = resolve_url(opt.repo, opt.revision, f);
    const auto dst = opt.target / f;
    std::cout << " - " << f << " ... ";
    std::error_code ec;
    fs::create_directories(dst.parent_path(), ec);

    bool ok = false;
#if defined(HF_FETCH_HAVE_LIBCURL)
    ok = download_via_libcurl(url, dst);
    if (!ok) {
      std::cout << "(libcurl failed, trying system curl) ";
    }
#endif
    if (!ok) {
      if (!have_system_curl()) {
        std::cout << "[FAIL] (no curl)\n";
        ++failures;
        continue;
      }
      ok = download_via_system_curl(url, dst);
    }

    if (ok) {
      std::cout << "[OK]" << '\n';
    } else {
      std::cout << "[FAIL]" << '\n';
      ++failures;
    }
  }

#if defined(HF_FETCH_HAVE_LIBCURL)
  curl_global_cleanup();
#endif

  if (failures) {
    std::cerr << "Completed with " << failures << " failures." << '\n';
    return 10;
  }
  std::cout << "All requested files downloaded." << '\n';
  return 0;
}
