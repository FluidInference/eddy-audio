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

std::filesystem::path get_app_data_dir() {
#ifdef _WIN32
    // Windows: %LOCALAPPDATA%\eddy
    wchar_t* localAppData = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &localAppData)) &&
        localAppData != nullptr) {
        std::filesystem::path base = localAppData;
        CoTaskMemFree(localAppData);
        return base / "eddy";
    }

    // Fallback: Use %LOCALAPPDATA% environment variable
    const char* localAppDataEnv = std::getenv("LOCALAPPDATA");
    if (localAppDataEnv != nullptr) {
        return std::filesystem::path(localAppDataEnv) / "eddy";
    }

    throw std::runtime_error("Failed to get Windows LocalAppData directory: both API and environment variable failed");

#else
    // Linux/Unix: Use XDG_CACHE_HOME or ~/.cache
    const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
    if (xdg_cache != nullptr) {
        return std::filesystem::path(xdg_cache) / "eddy";
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr) {
        throw std::runtime_error("Failed to get cache directory: both XDG_CACHE_HOME and HOME are unavailable");
    }
    return std::filesystem::path(home) / ".cache" / "eddy";
#endif
}

std::filesystem::path get_models_dir() {
    return get_app_data_dir() / "models";
}

std::filesystem::path get_model_dir(const std::string& model_name) {
    return get_models_dir() / model_name;
}

std::filesystem::path get_model_assets_dir(const std::string& model_name) {
    return get_model_dir(model_name) / "files";
}

// Backward-compat aliases
std::filesystem::path get_cache_dir() { return get_app_data_dir(); }
std::filesystem::path get_model_cache_dir(const std::string& model_name) { return get_model_dir(model_name); }
std::filesystem::path get_model_files_dir(const std::string& model_name) { return get_model_assets_dir(model_name); }

bool ensure_cache_dir(const std::filesystem::path& path) {
    std::error_code ec;

    if (std::filesystem::exists(path, ec)) {
        return std::filesystem::is_directory(path, ec);
    }

    return std::filesystem::create_directories(path, ec);
}

}  // namespace eddy
