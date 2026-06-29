// Simple HuggingFace model downloader using curl
// Downloads model files from HuggingFace repositories to local cache

#include "eddy/core/app_dir.hpp"
#include "eddy/core/model_configs.hpp"
#include "eddy/utils/ensure_models.hpp"

#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace eddy::model_configs;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <name>      Model name (default: parakeet-v2)\n";
    std::cout << "  --target <dir>      Target directory (default: cache directory)\n";
    std::cout << "  --help              Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << " --model parakeet-v2\n";
    std::cout << "  " << prog << " --target C:\\path\\to\\models\n";
}

int main(int argc, char** argv) {
    // Start with default model configuration
    std::string model_name = "parakeet-v2";
    std::string target_dir;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--model" && i + 1 < argc) {
            model_name = argv[++i];
        }
        else if (arg == "--target" && i + 1 < argc) {
            target_dir = argv[++i];
        }
        else {
            std::cerr << "ERROR: Unknown argument: " << arg << "\n\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Look up model configuration
    auto it = MODEL_MAP.find(model_name);
    if (it == MODEL_MAP.end()) {
        std::cerr << "ERROR: Unknown model: " << model_name << "\n";
        std::cerr << "Available models:";
        for (const auto& [k, _] : MODEL_MAP) std::cerr << " " << k;
        std::cerr << "\n";
        return 1;
    }

    const eddy::ModelConfig& config = it->second;

    // Set target directory if not specified
    if (target_dir.empty()) {
        try {
            target_dir = (eddy::get_models_dir() / config.cache_subdir / "files").string();
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Could not determine cache directory: " << e.what() << "\n";
            return 1;
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "HuggingFace Model Downloader\n";
    std::cout << "================================================================================\n";
    std::cout << "Model:      " << config.cache_subdir << "\n";
    std::cout << "Repository: " << config.repo_id << "\n";
    std::cout << "Target:     " << target_dir << "\n";
    std::cout << "Files:      " << config.required_files.size() << " files\n";
    std::cout << "================================================================================\n\n";

    // Progress callback
    auto progress_callback = [](const std::string& filename, int current, int total) {
        std::cout << "[" << current << "/" << total << "] " << filename << "\n";
    };

    // Download models using library function
    std::string error_msg;
    bool success = eddy::model_utils::download_models(
        config,
        fs::path(target_dir),
        &error_msg,
        progress_callback,
        true  // skip_existing
    );

    std::cout << "\n================================================================================\n";
    if (success) {
        std::cout << "Download completed successfully!\n";
        std::cout << "================================================================================\n";
        return 0;
    } else {
        std::cout << "Download failed!\n";
        std::cout << "================================================================================\n";
        std::cerr << "\nError: " << error_msg << "\n";
        return 1;
    }
}
