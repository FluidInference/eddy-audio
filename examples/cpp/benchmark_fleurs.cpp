// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0
//
// FLEURS Multilingual ASR Benchmark (C++ Native Implementation)
//
// Based on FluidAudio Swift implementation:
//   FluidAudio/Sources/FluidAudioCLI/Commands/ASR/FleursBenchmark.swift
//
// This native C++ benchmark processes FLEURS dataset entirely in C++ for maximum performance.
// It expects FLEURS data to be pre-downloaded by benchmark_fleurs.py.
//
// Usage:
//   benchmark_fleurs.exe <fleurs_cache_dir> --languages en_us,fr_fr --samples 10 --device NPU --output results.json

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
#include <map>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

// Language mapping (matches Python SUPPORTED_LANGUAGES)
const std::map<std::string, std::string> SUPPORTED_LANGUAGES = {
    {"en_us", "English (US)"},
    {"es_419", "Spanish (Spain)"},
    {"it_it", "Italian (Italy)"},
    {"fr_fr", "French (France)"},
    {"de_de", "German (Germany)"},
    {"ru_ru", "Russian (Russia)"},
    {"nl_nl", "Dutch (Netherlands)"},
    {"pl_pl", "Polish (Poland)"},
    {"uk_ua", "Ukrainian (Ukraine)"},
    {"sk_sk", "Slovak (Slovakia)"},
    {"cs_cz", "Czech (Czech Republic)"},
    {"bg_bg", "Bulgarian (Bulgaria)"},
    {"hr_hr", "Croatian (Croatia)"},
    {"ro_ro", "Romanian (Romania)"},
    {"fi_fi", "Finnish (Finland)"},
    {"hu_hu", "Hungarian (Hungary)"},
    {"sv_se", "Swedish (Sweden)"},
    {"et_ee", "Estonian (Estonia)"},
    {"da_dk", "Danish (Denmark)"},
    {"lt_lt", "Lithuanian (Lithuania)"},
    {"el_gr", "Greek (Greece)"},
    {"mt_mt", "Maltese (Malta)"},
    {"lv_lv", "Latvian (Latvia)"},
    {"sl_si", "Slovenian (Slovenia)"}
};

struct FLEURSSample {
    std::string sample_id;
    std::string audio_path;
    std::string transcription;
    std::string language;
};

struct LanguageResults {
    std::string language;
    std::string language_name;
    double wer;
    double cer;
    double rtfx;
    int samples_processed;
    int samples_skipped;
    double total_duration;
    double processing_time;
};

struct BenchmarkConfig {
    std::string cache_dir;
    std::vector<std::string> languages;
    int max_samples_per_lang;
    std::string output_file;
    std::string device;
    bool debug;
};

// Simple text normalization (lowercase, remove punctuation, normalize whitespace)
// Simplified version - for production, use Whisper normalizer
std::string normalize_text(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    bool last_was_space = false;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            result += std::tolower(static_cast<unsigned char>(c));
            last_was_space = false;
        } else if (!last_was_space && !result.empty()) {
            result += ' ';
            last_was_space = true;
        }
    }

    // Trim trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

// Calculate Levenshtein distance for WER/CER
int levenshtein_distance(const std::vector<std::string>& ref, const std::vector<std::string>& hyp) {
    const size_t m = ref.size();
    const size_t n = hyp.size();

    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (size_t i = 0; i <= m; ++i) dp[i][0] = i;
    for (size_t j = 0; j <= n; ++j) dp[0][j] = j;

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (ref[i-1] == hyp[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
            }
        }
    }

    return dp[m][n];
}

// Split text into words
std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }
    return words;
}

// Calculate WER (Word Error Rate)
double calculate_wer(const std::string& reference, const std::string& hypothesis) {
    auto ref_norm = normalize_text(reference);
    auto hyp_norm = normalize_text(hypothesis);

    auto ref_words = split_words(ref_norm);
    auto hyp_words = split_words(hyp_norm);

    if (ref_words.empty()) {
        return hyp_words.empty() ? 0.0 : 1.0;
    }

    int distance = levenshtein_distance(ref_words, hyp_words);
    return static_cast<double>(distance) / ref_words.size();
}

// Calculate CER (Character Error Rate)
double calculate_cer(const std::string& reference, const std::string& hypothesis) {
    auto ref_norm = normalize_text(reference);
    auto hyp_norm = normalize_text(hypothesis);

    // Remove spaces for character-level comparison
    std::string ref_chars, hyp_chars;
    for (char c : ref_norm) if (c != ' ') ref_chars += c;
    for (char c : hyp_norm) if (c != ' ') hyp_chars += c;

    if (ref_chars.empty()) {
        return hyp_chars.empty() ? 0.0 : 1.0;
    }

    // Convert to vectors of single-char strings
    std::vector<std::string> ref_vec, hyp_vec;
    for (char c : ref_chars) ref_vec.push_back(std::string(1, c));
    for (char c : hyp_chars) hyp_vec.push_back(std::string(1, c));

    int distance = levenshtein_distance(ref_vec, hyp_vec);
    return static_cast<double>(distance) / ref_chars.size();
}

// Load FLEURS samples for a language
std::vector<FLEURSSample> load_language_samples(
    const std::string& cache_dir,
    const std::string& language,
    int max_samples
) {
    std::vector<FLEURSSample> samples;

    fs::path lang_dir = fs::path(cache_dir) / language;
    if (!fs::exists(lang_dir)) {
        std::cerr << "Warning: Language directory not found: " << lang_dir << "\n";
        return samples;
    }

    // Load transcriptions from .trans.txt file (LibriSpeech format)
    fs::path trans_file = lang_dir / (language + ".trans.txt");
    std::map<std::string, std::string> transcriptions;

    if (fs::exists(trans_file)) {
        std::ifstream file(trans_file);
        std::string line;
        while (std::getline(file, line)) {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string sample_id = line.substr(0, space_pos);
                std::string text = line.substr(space_pos + 1);
                transcriptions[sample_id] = text;
            }
        }
    }

    // Load audio files
    std::vector<fs::path> audio_files;
    for (const auto& entry : fs::directory_iterator(lang_dir)) {
        if (entry.path().extension() == ".wav") {
            audio_files.push_back(entry.path());
        }
    }

    // Sort for consistent ordering
    std::sort(audio_files.begin(), audio_files.end());

    // Limit samples if specified
    if (max_samples > 0 && audio_files.size() > static_cast<size_t>(max_samples)) {
        audio_files.resize(max_samples);
    }

    // Create samples
    for (const auto& audio_path : audio_files) {
        std::string sample_id = audio_path.stem().string();

        FLEURSSample sample;
        sample.sample_id = sample_id;
        sample.audio_path = audio_path.string();
        sample.language = language;

        if (transcriptions.count(sample_id)) {
            sample.transcription = transcriptions[sample_id];
        }

        samples.push_back(sample);
    }

    return samples;
}

// Process samples for a language
LanguageResults process_language_samples(
    const std::vector<FLEURSSample>& samples,
    const std::string& language,
    std::shared_ptr<eddy::parakeet::OpenVINOParakeet> model,
    bool debug_enabled
) {
    LanguageResults results;
    results.language = language;
    results.language_name = SUPPORTED_LANGUAGES.at(language);
    results.wer = 0.0;
    results.cer = 0.0;
    results.rtfx = 0.0;
    results.samples_processed = 0;
    results.samples_skipped = 0;
    results.total_duration = 0.0;
    results.processing_time = 0.0;

    double total_wer = 0.0;
    double total_cer = 0.0;

    for (const auto& sample : samples) {
        try {
            // Load audio
            auto audio_samples = eddy::audio::read_wav(sample.audio_path);
            double audio_duration = audio_samples.size() / 16000.0;

            if (debug_enabled) {
                std::cout << "  Processing: " << sample.sample_id << "\n";
                std::cout << "    Duration: " << std::fixed << std::setprecision(2)
                          << audio_duration << "s\n";
            }

            // Prepare audio segment
            eddy::parakeet::AudioSegment segment;
            segment.sample_rate = 16000;
            segment.pcm = audio_samples;

            // Run inference
            eddy::parakeet::SegmentOptions options;
            auto start = std::chrono::high_resolution_clock::now();
            auto inference_result = model->infer(segment, options);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            double processing_time_sec = duration_ms / 1000.0;

            // Calculate metrics
            if (!sample.transcription.empty()) {
                double wer = calculate_wer(sample.transcription, inference_result.text);
                double cer = calculate_cer(sample.transcription, inference_result.text);

                total_wer += wer;
                total_cer += cer;

                if (debug_enabled) {
                    std::cout << "    Hypothesis: " << inference_result.text << "\n";
                    std::cout << "    Reference:  " << sample.transcription << "\n";
                    std::cout << "    WER: " << std::fixed << std::setprecision(1) << (wer * 100) << "%\n";
                }
            }

            results.total_duration += audio_duration;
            results.processing_time += processing_time_sec;
            results.samples_processed++;

        } catch (const std::exception& e) {
            std::cerr << "Warning: Error processing " << sample.sample_id << ": " << e.what() << "\n";
            results.samples_skipped++;
        }
    }

    // Calculate averages
    if (results.samples_processed > 0) {
        results.wer = total_wer / results.samples_processed;
        results.cer = total_cer / results.samples_processed;
        results.rtfx = (results.processing_time > 0)
            ? results.total_duration / results.processing_time
            : 0.0;
    }

    return results;
}

// Save results to JSON
void save_results_json(
    const std::vector<LanguageResults>& results,
    const BenchmarkConfig& config,
    const std::string& output_file
) {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");

    file << "{\n";
    file << "  \"benchmark\": \"FLEURS Multilingual ASR (C++ Native)\",\n";
    file << "  \"timestamp\": \"" << timestamp.str() << "\",\n";
    file << "  \"config\": {\n";
    file << "    \"languages\": [";
    for (size_t i = 0; i < config.languages.size(); ++i) {
        file << "\"" << config.languages[i] << "\"";
        if (i + 1 < config.languages.size()) file << ", ";
    }
    file << "],\n";
    file << "    \"samplesPerLanguage\": " << config.max_samples_per_lang << ",\n";
    file << "    \"device\": \"" << config.device << "\"\n";
    file << "  },\n";
    file << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        file << "    {\n";
        file << "      \"language\": \"" << r.language << "\",\n";
        file << "      \"languageName\": \"" << r.language_name << "\",\n";
        file << "      \"wer\": " << r.wer << ",\n";
        file << "      \"cer\": " << r.cer << ",\n";
        file << "      \"rtfx\": " << r.rtfx << ",\n";
        file << "      \"samplesProcessed\": " << r.samples_processed << ",\n";
        file << "      \"samplesSkipped\": " << r.samples_skipped << ",\n";
        file << "      \"totalDuration\": " << r.total_duration << ",\n";
        file << "      \"processingTime\": " << r.processing_time << "\n";
        file << "    }";
        if (i + 1 < results.size()) file << ",";
        file << "\n";
    }

    file << "  ],\n";
    file << "  \"summary\": {\n";

    double avg_wer = 0.0, avg_cer = 0.0, avg_rtfx = 0.0;
    double total_duration = 0.0, total_processing = 0.0;
    int total_samples = 0, total_skipped = 0;

    for (const auto& r : results) {
        avg_wer += r.wer;
        avg_cer += r.cer;
        avg_rtfx += r.rtfx;
        total_duration += r.total_duration;
        total_processing += r.processing_time;
        total_samples += r.samples_processed;
        total_skipped += r.samples_skipped;
    }

    if (!results.empty()) {
        avg_wer /= results.size();
        avg_cer /= results.size();
        avg_rtfx /= results.size();
    }

    file << "    \"averageWER\": " << avg_wer << ",\n";
    file << "    \"averageCER\": " << avg_cer << ",\n";
    file << "    \"averageRTFx\": " << avg_rtfx << ",\n";
    file << "    \"totalSamples\": " << total_samples << ",\n";
    file << "    \"totalSkipped\": " << total_skipped << ",\n";
    file << "    \"totalDuration\": " << total_duration << ",\n";
    file << "    \"totalProcessingTime\": " << total_processing << "\n";
    file << "  }\n";
    file << "}\n";

    file.close();
}

void print_usage(const char* program_name) {
    std::cout << "FLEURS Multilingual ASR Benchmark (C++ Native)\n\n";
    std::cout << "Usage: " << program_name << " <cache_dir> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --languages <list>  Comma-separated language codes (default: en_us)\n";
    std::cout << "  --samples <n>       Max samples per language (default: 10, use 0 for all)\n";
    std::cout << "  --device <device>   OpenVINO device (default: CPU)\n";
    std::cout << "  --output <file>     Output JSON file (default: fleurs_cpp_results.json)\n";
    std::cout << "  --debug             Enable debug output\n";
    std::cout << "  --help              Show this help\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " %LOCALAPPDATA%/eddy/datasets/FLEURS --languages en_us,fr_fr --samples 10 --device NPU\n";
}

int main(int argc, char* argv[]) {
    // Force unbuffered output
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    BenchmarkConfig config;
    config.cache_dir = argv[1];
    config.languages = {"en_us"};  // default
    config.max_samples_per_lang = 10;
    config.output_file = "fleurs_cpp_results.json";
    config.device = "CPU";
    config.debug = false;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--languages" && i + 1 < argc) {
            config.languages.clear();
            std::string langs = argv[++i];
            std::istringstream ss(langs);
            std::string lang;
            while (std::getline(ss, lang, ',')) {
                config.languages.push_back(lang);
            }
        } else if (arg == "--samples" && i + 1 < argc) {
            config.max_samples_per_lang = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            config.device = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--debug") {
            config.debug = true;
        }
    }

    std::cout << "=== FLEURS Multilingual ASR Benchmark (C++ Native) ===\n\n";
    std::cout << "Cache directory: " << config.cache_dir << "\n";
    std::cout << "Languages: ";
    for (size_t i = 0; i < config.languages.size(); ++i) {
        std::cout << config.languages[i];
        if (i + 1 < config.languages.size()) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "Samples per language: " << (config.max_samples_per_lang == 0 ? "all" : std::to_string(config.max_samples_per_lang)) << "\n";
    std::cout << "Device: " << config.device << "\n";
    std::cout << "Output: " << config.output_file << "\n\n";

    try {
        // Initialize OpenVINO backend
        std::cout << "Initializing OpenVINO backend (" << config.device << ") ... ";
        std::cout.flush();

        auto compiled_cache_dir = eddy::get_model_dir("parakeet-v3").string();
        eddy::OpenVINOOptions ov_opts;
        ov_opts.device = config.device;
        ov_opts.cache_dir = compiled_cache_dir;
        auto backend = std::make_shared<eddy::OpenVINOBackend>(ov_opts);
        std::cout << "[OK]\n";

        // Load Parakeet v3 models
        auto cache_model_dir = eddy::get_model_assets_dir("parakeet-v3");
        std::string fetch_err;
        if (!eddy::model_utils::check_models_available(cache_model_dir, &fetch_err)) {
            if (!fetch_err.empty()) std::cout << "[INFO] " << fetch_err << "\n";
        }

        std::filesystem::path model_dir = cache_model_dir;

        eddy::parakeet::ModelPaths paths{
            .preprocessor = {.path = (model_dir / "parakeet_melspectogram.xml").string()},
            .encoder = {.path = (model_dir / "parakeet_encoder.xml").string()},
            .decoder = {.path = (model_dir / "parakeet_decoder.xml").string()},
            .joint = {.path = (model_dir / "parakeet_joint.xml").string()},
            .tokenizer_json = (model_dir / "parakeet_vocab.json").string()
        };

        // Configure for Parakeet v3
        eddy::parakeet::RuntimeConfig cfg{
            .device = config.device,
            .blank_token_id = 8192,  // v3
            .duration_bins = {0, 1, 2, 3, 4}
        };

        std::cout << "Loading Parakeet v3 models ... ";
        std::cout.flush();
        auto model = eddy::parakeet::make_openvino_parakeet(backend, paths, cfg);
        std::cout << "[OK]\n";

        std::cout << "Warming up model ... ";
        std::cout.flush();
        auto parakeet_model = std::static_pointer_cast<eddy::parakeet::OpenVINOParakeet>(model);
        parakeet_model->warmup();
        std::cout << "[OK]\n\n";

        // Process each language
        std::vector<LanguageResults> all_results;

        for (const auto& language : config.languages) {
            if (SUPPORTED_LANGUAGES.find(language) == SUPPORTED_LANGUAGES.end()) {
                std::cerr << "Warning: Unsupported language: " << language << "\n";
                continue;
            }

            std::cout << "Processing " << SUPPORTED_LANGUAGES.at(language) << " (" << language << ")...\n";

            auto samples = load_language_samples(config.cache_dir, language, config.max_samples_per_lang);
            if (samples.empty()) {
                std::cerr << "Warning: No samples found for " << language << "\n";
                continue;
            }

            std::cout << "  Loaded " << samples.size() << " samples\n";

            auto results = process_language_samples(samples, language, parakeet_model, config.debug);
            all_results.push_back(results);

            std::cout << "  " << language << ": WER=" << std::fixed << std::setprecision(1)
                      << (results.wer * 100) << "%, CER=" << (results.cer * 100)
                      << "%, RTFx=" << std::setprecision(1) << results.rtfx << "x"
                      << " (" << results.samples_processed << " processed)\n\n";
        }

        // Save results
        save_results_json(all_results, config, config.output_file);
        std::cout << "Results saved to: " << config.output_file << "\n";

        // Print summary
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BENCHMARK SUMMARY\n";
        std::cout << std::string(80, '=') << "\n\n";

        for (const auto& r : all_results) {
            std::cout << std::left << std::setw(25) << r.language_name
                      << " | WER=" << std::fixed << std::setprecision(1) << std::setw(5) << (r.wer * 100) << "%"
                      << " | CER=" << std::setw(5) << (r.cer * 100) << "%"
                      << " | RTFx=" << std::setprecision(1) << std::setw(5) << r.rtfx << "x"
                      << " | Samples=" << r.samples_processed << "\n";
        }

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "SUCCESS\n";
        std::cout << std::string(80, '=') << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n\n";
        std::cerr << "Troubleshooting:\n";
        std::cerr << "  1. Ensure FLEURS data is downloaded (use benchmark_fleurs.py first)\n";
        std::cerr << "  2. Check cache directory exists: " << config.cache_dir << "\n";
        std::cerr << "  3. Verify Parakeet v3 models are available\n";
        std::cerr << "  4. Try --device CPU if " << config.device << " fails\n";
        return 1;
    }
}
