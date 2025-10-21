// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/core/app_dir.hpp"
#include "eddy/models/parakeet-v2/parakeet.hpp"
#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "eddy/utils/ensure_models.hpp"
#include "eddy/utils/audio_utils.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <audio.wav> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --device <device>    OpenVINO device (default: CPU)\n";
    std::cout << "                       Options: CPU, GPU, AUTO\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Requirements:\n";
    std::cout << "  - Audio must be 16kHz mono or stereo WAV file\n";
    std::cout << "  - Models will be loaded from cache or models/parakeet/\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " test.wav\n";
    std::cout << "  " << program_name << " test.wav --device GPU\n";
}

int main(int argc, char* argv[]) {
    // Force unbuffered output for debugging
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    std::cerr << "[DEBUG] Program started, argc=" << argc << std::endl;

    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "[DEBUG] No args provided, showing usage" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::string audio_file;
    std::string device = "CPU";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--device") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --device requires an argument\n";
                return 1;
            }
            device = argv[++i];
        } else {
            // Assume it's the audio file
            audio_file = arg;
        }
    }

    if (audio_file.empty()) {
        std::cerr << "Error: No audio file specified\n\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== Parakeet TDT v2 Transcription CLI ===\n\n";

    try {
        // Load audio file
        std::cout << "Loading audio: " << audio_file << " ... ";
        std::cout.flush();
        auto audio_samples = eddy::audio::read_wav(audio_file);
        std::cout << "[OK]\n";
        std::cout << "  Samples: " << audio_samples.size() << "\n";
        std::cout << "  Duration: " << std::fixed << std::setprecision(2)
                  << (audio_samples.size() / 16000.0) << " seconds\n\n";

        // Create OpenVINO backend
        std::cout << "Initializing OpenVINO backend (" << device << ") ... ";
        std::cout.flush();
        // Set compiled model cache to the per-model cache dir
        auto compiled_cache_dir = eddy::get_model_dir("parakeet-v2").string();
        eddy::OpenVINOOptions ov_opts;
        ov_opts.device = device;
        ov_opts.cache_dir = compiled_cache_dir;
        auto backend = std::make_shared<eddy::OpenVINOBackend>(ov_opts);
        std::cout << "[OK]\n";

        // Determine model directory: ensure cache has required files (centralized helper)
        auto cache_model_dir = eddy::get_model_assets_dir("parakeet-v2");
        std::filesystem::path model_dir;
        std::string fetch_err;
        if (!eddy::parakeet::ensure_models_available(cache_model_dir, &fetch_err)) {
            if (!fetch_err.empty()) std::cout << "[INFO] " << fetch_err << "\n";
        }

        // Prefer cache if encoder xml exists (minimum signal of a complete set)
        auto exists_nonempty = [](const std::filesystem::path& p) -> bool {
            std::error_code ec;
            return std::filesystem::exists(p, ec) && std::filesystem::is_regular_file(p, ec) && std::filesystem::file_size(p, ec) > 0;
        };
        if (exists_nonempty(cache_model_dir / "parakeet_encoder.xml")) {
            model_dir = cache_model_dir;
            std::cout << "Using cached models at: " << cache_model_dir.string() << "\n\n";
        } else {
            // Fallback: legacy Windows path (%LOCALAPPDATA%\eddy\cache\models\<name>\files)
#if defined(_WIN32)
            auto legacy_dir = eddy::get_app_data_dir() / "cache" / "models" / "parakeet-v2" / "files";
            if (exists_nonempty(legacy_dir / "parakeet_encoder.xml")) {
                model_dir = legacy_dir;
                std::cout << "Using legacy cached models at: " << legacy_dir.string() << "\n\n";
            } else
#endif
            {
                model_dir = "models/parakeet";
                std::cout << "Using local models at: " << model_dir.string() << "\n";
                std::cout << "Note: Copy models to " << cache_model_dir.string() << " for user cache access\n\n";
            }
        }

        // Configure model paths
        eddy::parakeet::ModelPaths paths{
            .preprocessor = {.path = (model_dir / "parakeet_melspectogram.xml").string()},
            .encoder = {.path = (model_dir / "parakeet_encoder.xml").string()},
            .decoder = {.path = (model_dir / "parakeet_decoder.xml").string()},
            .joint = {.path = (model_dir / "parakeet_joint.xml").string()},
            .tokenizer_json = (model_dir / "parakeet_vocab.json").string()
        };

        // Configure runtime (v2 uses blank_token_id=1024)
        eddy::parakeet::RuntimeConfig cfg{
            .device = device,
            .blank_token_id = 1024,
            .duration_bins = {0, 1, 2, 3, 4}
        };

        // Load models
        std::cout << "Loading Parakeet models ... ";
        std::cout.flush();
        auto model = eddy::parakeet::make_openvino_parakeet(backend, paths, cfg);
        std::cout << "[OK]\n";

        // Warmup
        std::cout << "Warming up model ... ";
        std::cout.flush();
        auto parakeet_model = std::static_pointer_cast<eddy::parakeet::OpenVINOParakeet>(model);
        parakeet_model->warmup();
        std::cout << "[OK]\n\n";

        // Prepare audio segment
        eddy::parakeet::AudioSegment segment;
        segment.sample_rate = 16000;
        segment.pcm = audio_samples;

        // Run inference
        std::cout << std::string(70, '=') << "\n";
        std::cout << "TRANSCRIBING...\n";
        std::cout << std::string(70, '=') << "\n\n";

        eddy::parakeet::SegmentOptions options;
        auto start = std::chrono::high_resolution_clock::now();
        auto result = model->infer(segment, options);
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate metrics
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float audio_duration = audio_samples.size() / 16000.0f;
        float rtfx = audio_duration / (duration_ms / 1000.0f);

        // Display results
        std::cout << "Result:\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << result.text << "\n";
        std::cout << std::string(70, '-') << "\n\n";

        std::cout << "Metrics:\n";
        std::cout << "  Tokens:           " << result.token_ids.size() << "\n";
        std::cout << "  Confidence:       " << std::fixed << std::setprecision(1)
                  << (result.overall_confidence * 100.0f) << "%\n";
        std::cout << "  Processing time:  " << duration_ms << " ms\n";
        std::cout << "  Audio duration:   " << std::fixed << std::setprecision(2)
                  << audio_duration << " s\n";
        std::cout << "  Real-time factor: " << std::fixed << std::setprecision(1)
                  << rtfx << "x\n\n";

        // Show token timings (first 10 tokens as sample)
        if (!result.token_timings.empty()) {
            std::cout << "Token Timings (first 10):\n";
            const size_t num_to_show = std::min(size_t(10), result.token_timings.size());
            for (size_t i = 0; i < num_to_show; ++i) {
                const auto& timing = result.token_timings[i];
                // Convert frame_index to seconds (frame * 0.08)
                float time_seconds = timing.frame_index * 0.08f;
                std::cout << "  " << std::setw(3) << i+1 << ". "
                          << "t=" << std::fixed << std::setprecision(2) << std::setw(5) << time_seconds << "s "
                          << "conf=" << std::setprecision(1) << std::setw(4) << (timing.confidence * 100.0f) << "% "
                          << "token_id=" << timing.token_id << "\n";
            }
            if (result.token_timings.size() > num_to_show) {
                std::cout << "  ... and " << (result.token_timings.size() - num_to_show) << " more tokens\n";
            }
            std::cout << "\n";
        }

        // Performance assessment
        if (rtfx >= 10.0f) {
            std::cout << "✅ Performance: Excellent (>" << std::setprecision(0) << rtfx << "x real-time)\n";
        } else if (rtfx >= 1.0f) {
            std::cout << "✅ Performance: Good (processing faster than real-time)\n";
        } else {
            std::cout << "⚠️  Performance: Below real-time (consider GPU device or optimizations)\n";
        }

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "SUCCESS\n";
        std::cout << std::string(70, '=') << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        std::cerr << "Troubleshooting:\n";
        std::cerr << "  1. Ensure audio file is 16kHz WAV format\n";
        std::cerr << "  2. Check models are in: " << eddy::get_model_assets_dir("parakeet-v2").string() << "\n";
        std::cerr << "     or in: models/parakeet/\n";
        std::cerr << "  3. Verify OpenVINO runtime is properly installed\n";
        std::cerr << "  4. Try --device CPU if GPU fails\n";
        return 1;
    }
}
