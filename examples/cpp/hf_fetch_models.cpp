// Simple HuggingFace model downloader using curl
// Downloads model files from HuggingFace repositories to local cache

#include "eddy/core/model_configs.hpp"
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdlib>

namespace fs = std::filesystem;
using namespace eddy::model_configs;

bool download_file(const std::string& url, const fs::path& output_path) {
    // Create parent directory
    fs::create_directories(output_path.parent_path());

    std::string curl_cmd = "curl -L --progress-bar \"" + url + "\" -o \"" + output_path.string() + "\"";

    std::cout << "Downloading: " << output_path.filename().string() << "\n";
    int ret = std::system(curl_cmd.c_str());

    if (ret != 0) {
        std::cerr << "ERROR: Download failed (exit code " << ret << ")\n";
        std::cerr << "URL: " << url << "\n";
        return false;
    }

    if (!fs::exists(output_path) || fs::file_size(output_path) == 0) {
        std::cerr << "ERROR: Downloaded file is missing or empty\n";
        return false;
    }

    std::cout << "[OK] " << (fs::file_size(output_path) / (1024 * 1024)) << " MB\n\n";
    return true;
}

std::string get_cache_dir(const std::string& cache_subdir) {
#ifdef _WIN32
    const char* localappdata = std::getenv("LOCALAPPDATA");
    if (!localappdata) {
        std::cerr << "ERROR: LOCALAPPDATA not set\n";
        return "";
    }
    return std::string(localappdata) + "\\eddy\\models\\" + cache_subdir + "\\files";
#else
    const char* home = std::getenv("HOME");
    if (!home) {
        std::cerr << "ERROR: HOME not set\n";
        return "";
    }
    return std::string(home) + "/.cache/eddy/models/" + cache_subdir + "/files";
#endif
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <name>      Model name (v2, v3) (default: v2)\n";
    std::cout << "  --repo <repo_id>    Override HuggingFace repository\n";
    std::cout << "  --target <dir>      Target directory (default: cache directory)\n";
    std::cout << "  --files <list>      Comma-separated list of files to download\n";
    std::cout << "  --help              Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << " --model v2\n";
    std::cout << "  " << prog << " --model v3\n";
    std::cout << "  " << prog << " --repo FluidInference/parakeet-tdt-0.6b-v2-ov\n";
}

int main(int argc, char** argv) {
    // Start with default model configuration
    eddy::ModelConfig config = DEFAULT;
    std::string target_dir;
    bool custom_repo = false;
    bool custom_files = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--model" && i + 1 < argc) {
            std::string model_name = argv[++i];
            auto it = MODEL_MAP.find(model_name);
            if (it != MODEL_MAP.end()) {
                config = it->second;
            } else {
                std::cerr << "ERROR: Unknown model: " << model_name << "\n";
                std::cerr << "Available models: v2, v3, parakeet-v2, parakeet-v3\n";
                return 1;
            }
        }
        else if (arg == "--repo" && i + 1 < argc) {
            config.repo_id = argv[++i];
            custom_repo = true;
        }
        else if (arg == "--target" && i + 1 < argc) {
            target_dir = argv[++i];
        }
        else if (arg == "--files" && i + 1 < argc) {
            // Parse comma-separated file list
            config.required_files.clear();
            std::string files_str = argv[++i];
            size_t start = 0, end;
            while ((end = files_str.find(',', start)) != std::string::npos) {
                config.required_files.push_back(files_str.substr(start, end - start));
                start = end + 1;
            }
            config.required_files.push_back(files_str.substr(start));
            custom_files = true;
        }
        else {
            std::cerr << "ERROR: Unknown argument: " << arg << "\n\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Set target directory if not specified
    if (target_dir.empty()) {
        target_dir = get_cache_dir(config.cache_subdir);
    }

    if (target_dir.empty()) {
        std::cerr << "ERROR: Could not determine cache directory\n";
        return 1;
    }

    std::cout << "================================================================================\n";
    std::cout << "HuggingFace Model Downloader\n";
    std::cout << "================================================================================\n";
    std::cout << "Model:      " << config.cache_subdir << "\n";
    std::cout << "Repository: " << config.repo_id << "\n";
    std::cout << "Target:     " << target_dir << "\n";
    std::cout << "Files:      " << config.required_files.size() << " files\n";
    std::cout << "================================================================================\n\n";

    // Check if all files already exist
    bool all_exist = true;
    for (const auto& file : config.required_files) {
        fs::path file_path = fs::path(target_dir) / file;
        if (!fs::exists(file_path) || fs::file_size(file_path) == 0) {
            all_exist = false;
            break;
        }
    }

    if (all_exist) {
        std::cout << "All files already present. Skipping download.\n";
        return 0;
    }

    // Download each file
    int succeeded = 0;
    int failed = 0;

    for (const auto& file : config.required_files) {
        fs::path file_path = fs::path(target_dir) / file;

        // Skip if already exists
        if (fs::exists(file_path) && fs::file_size(file_path) > 0) {
            std::cout << "[SKIP] " << file << " (already exists)\n\n";
            succeeded++;
            continue;
        }

        // Construct HuggingFace URL
        std::string url = "https://huggingface.co/" + config.repo_id + "/resolve/main/" + file;

        if (download_file(url, file_path)) {
            succeeded++;
        } else {
            failed++;
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "Summary: " << succeeded << " succeeded, " << failed << " failed\n";
    std::cout << "================================================================================\n";

    return (failed > 0) ? 1 : 0;
}
