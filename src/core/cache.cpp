// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/core/cache.hpp"

#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#endif

namespace eddy {

std::filesystem::path get_cache_dir() {
#ifdef _WIN32
    // Windows: Use LOCALAPPDATA
    wchar_t* localAppData = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &localAppData))) {
        std::filesystem::path cache_path = localAppData;
        CoTaskMemFree(localAppData);
        return cache_path / "eddy" / "cache";
    }

    // Fallback: Use %LOCALAPPDATA% environment variable
    const char* localAppDataEnv = std::getenv("LOCALAPPDATA");
    if (localAppDataEnv) {
        return std::filesystem::path(localAppDataEnv) / "eddy" / "cache";
    }

    throw std::runtime_error("Failed to get Windows LocalAppData directory");

#elif defined(__APPLE__)
    // macOS: Use ~/Library/Caches
    const char* home = std::getenv("HOME");
    if (!home) {
        throw std::runtime_error("Failed to get HOME directory");
    }
    return std::filesystem::path(home) / "Library" / "Caches" / "eddy";

#else
    // Linux: Use XDG_CACHE_HOME or ~/.cache
    const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
    if (xdg_cache) {
        return std::filesystem::path(xdg_cache) / "eddy";
    }

    const char* home = std::getenv("HOME");
    if (!home) {
        throw std::runtime_error("Failed to get HOME directory");
    }
    return std::filesystem::path(home) / ".cache" / "eddy";
#endif
}

std::filesystem::path get_model_cache_dir(const std::string& model_name) {
    return get_cache_dir() / "models" / model_name;
}

std::filesystem::path get_model_files_dir(const std::string& model_name) {
    // Model files go in cache/models/<name>/files/
    // Compiled cache goes in cache/models/<name>/ (set via ov::cache_dir)
    return get_cache_dir() / "models" / model_name / "files";
}

bool ensure_cache_dir(const std::filesystem::path& path) {
    std::error_code ec;

    if (std::filesystem::exists(path, ec)) {
        return std::filesystem::is_directory(path, ec);
    }

    return std::filesystem::create_directories(path, ec);
}

}  // namespace eddy
