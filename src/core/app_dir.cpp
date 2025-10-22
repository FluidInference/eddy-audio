// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/core/app_dir.hpp"

#include <cstdlib>
#include <stdexcept>
#include <memory>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#endif

namespace eddy {

std::filesystem::path get_app_data_dir() {
    // Cache the result to avoid repeated getenv calls and API calls
    static std::filesystem::path cached_path = []() -> std::filesystem::path {
#ifdef _WIN32
        // Windows: %LOCALAPPDATA%\eddy
        wchar_t* localAppData = nullptr;
        if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &localAppData)) &&
            localAppData != nullptr) {
            // Use RAII to ensure cleanup even if path construction throws
            std::unique_ptr<wchar_t, decltype(&CoTaskMemFree)> cleanup(localAppData, CoTaskMemFree);
            std::filesystem::path base = localAppData;
            return base / "eddy";
        }

        // Fallback: Use %LOCALAPPDATA% environment variable
        const char* localAppDataEnv = std::getenv("LOCALAPPDATA");
        if (localAppDataEnv != nullptr) {
            return std::filesystem::path(localAppDataEnv) / "eddy";
        }

        throw std::runtime_error("Failed to get Windows LocalAppData directory: both API and environment variable failed");

#else
        // Linux/Unix: Use XDG_CACHE_HOME or ~/.cache for app data
        const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
        if (xdg_cache != nullptr) {
            return std::filesystem::path(xdg_cache) / "eddy";
        }

        const char* home = std::getenv("HOME");
        if (home == nullptr) {
            throw std::runtime_error("Failed to get app data directory: both XDG_CACHE_HOME and HOME are unavailable");
        }
        return std::filesystem::path(home) / ".cache" / "eddy";
#endif
    }();

    return cached_path;
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

bool ensure_directory(const std::filesystem::path& path) {
    std::error_code ec;

    // Check if path already exists
    if (std::filesystem::exists(path, ec)) {
        if (ec) return false;  // Error checking existence
        bool is_dir = std::filesystem::is_directory(path, ec);
        return is_dir && !ec;
    }

    // Create directories if they don't exist
    std::filesystem::create_directories(path, ec);
    return !ec;
}

}  // namespace eddy
