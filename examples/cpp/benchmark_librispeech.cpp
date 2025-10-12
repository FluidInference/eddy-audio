// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/core/cache.hpp"
#include "eddy/models/parakeet/parakeet.hpp"
#include "eddy/models/parakeet/parakeet_openvino.hpp"
#include "eddy/pipelines/audio_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// Text Normalization and WER/CER Calculation
// ============================================================================

std::string normalize_text(const std::string& text) {
    // Lowercase
    std::string result;
    for (char c : text) {
        result += std::tolower(c);
    }

    // Remove punctuation
    std::string no_punct;
    for (char c : result) {
        if (std::isalnum(c) || std::isspace(c)) {
            no_punct += c;
        }
    }

    // Normalize whitespace
    std::string normalized;
    bool in_space = false;
    for (char c : no_punct) {
        if (std::isspace(c)) {
            if (!in_space && !normalized.empty()) {
                normalized += ' ';
                in_space = true;
            }
        } else {
            normalized += c;
            in_space = false;
        }
    }

    // Trim trailing space
    if (!normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }

    return normalized;
}

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::string word;
    for (char c : text) {
        if (std::isspace(c)) {
            if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        } else {
            word += c;
        }
    }
    if (!word.empty()) {
        words.push_back(word);
    }
    return words;
}

struct WERMetrics {
    float wer;
    int substitutions;
    int deletions;
    int insertions;
    int total_words;
    std::string normalized_hypothesis;
    std::string normalized_reference;
};

WERMetrics calculate_wer(const std::string& hypothesis, const std::string& reference) {
    auto hyp_norm = normalize_text(hypothesis);
    auto ref_norm = normalize_text(reference);

    auto hyp_words = split_words(hyp_norm);
    auto ref_words = split_words(ref_norm);

    int m = ref_words.size();
    int n = hyp_words.size();

    // Dynamic programming for edit distance
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

    for (int i = 0; i <= m; ++i) dp[i][0] = i;
    for (int j = 0; j <= n; ++j) dp[0][j] = j;

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (ref_words[i-1] == hyp_words[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({
                    dp[i-1][j],     // deletion
                    dp[i][j-1],     // insertion
                    dp[i-1][j-1]    // substitution
                });
            }
        }
    }

    // Backtrack to count operations
    int i = m, j = n;
    int subs = 0, dels = 0, ins = 0;

    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && ref_words[i-1] == hyp_words[j-1]) {
            --i; --j;
        } else if (i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1] + 1) {
            ++subs;
            --i; --j;
        } else if (i > 0 && dp[i][j] == dp[i-1][j] + 1) {
            ++dels;
            --i;
        } else if (j > 0 && dp[i][j] == dp[i][j-1] + 1) {
            ++ins;
            --j;
        } else {
            break;
        }
    }

    float wer = m > 0 ? static_cast<float>(subs + dels + ins) / m : 0.0f;

    return {wer, subs, dels, ins, m, hyp_norm, ref_norm};
}

float calculate_cer(const std::string& hypothesis, const std::string& reference) {
    auto hyp_norm = normalize_text(hypothesis);
    auto ref_norm = normalize_text(reference);

    // Remove spaces for character-level comparison
    std::string hyp_chars, ref_chars;
    for (char c : hyp_norm) if (!std::isspace(c)) hyp_chars += c;
    for (char c : ref_norm) if (!std::isspace(c)) ref_chars += c;

    int m = ref_chars.size();
    int n = hyp_chars.size();

    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

    for (int i = 0; i <= m; ++i) dp[i][0] = i;
    for (int j = 0; j <= n; ++j) dp[0][j] = j;

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (ref_chars[i-1] == hyp_chars[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
            }
        }
    }

    return m > 0 ? static_cast<float>(dp[m][n]) / m : 0.0f;
}

// ============================================================================
// LibriSpeech Dataset Loading
// ============================================================================

struct TestFile {
    std::string file_id;
    fs::path audio_path;
    std::string reference;
};

std::vector<TestFile> load_librispeech_transcripts(const fs::path& dataset_dir, int max_files = -1) {
    std::vector<TestFile> files;

    // Find all .trans.txt files
    for (const auto& entry : fs::recursive_directory_iterator(dataset_dir)) {
        if (entry.path().extension() != ".txt") continue;
        if (entry.path().filename().string().find(".trans.txt") == std::string::npos) continue;

        std::ifstream file(entry.path());
        if (!file.is_open()) continue;

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            // Parse: "file-id TRANSCRIPT TEXT"
            auto space_pos = line.find(' ');
            if (space_pos == std::string::npos) continue;

            std::string file_id = line.substr(0, space_pos);
            std::string transcript = line.substr(space_pos + 1);

            // Find corresponding .flac file
            fs::path audio_path = entry.path().parent_path() / (file_id + ".flac");
            if (!fs::exists(audio_path)) continue;

            files.push_back({file_id, audio_path, transcript});

            if (max_files > 0 && files.size() >= static_cast<size_t>(max_files)) {
                return files;
            }
        }
    }

    // Sort by file_id for consistency
    std::sort(files.begin(), files.end(), [](const TestFile& a, const TestFile& b) {
        return a.file_id < b.file_id;
    });

    return files;
}

// ============================================================================
// FLAC to WAV Conversion (using ffmpeg)
// ============================================================================

fs::path convert_flac_to_wav(const fs::path& flac_path) {
    fs::path wav_path = flac_path;
    wav_path.replace_extension(".wav");

    // Check if already converted
    if (fs::exists(wav_path)) {
        return wav_path;
    }

    // Convert using ffmpeg
    std::string cmd = "ffmpeg -i \"" + flac_path.string() + "\" "
                      "-ar 16000 -ac 1 -y -loglevel error \"" + wav_path.string() + "\"";

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("ffmpeg conversion failed for: " + flac_path.string());
    }

    return wav_path;
}

// ============================================================================
// Benchmark Result Storage
// ============================================================================

struct BenchmarkResult {
    std::string file_id;
    std::string audio_path;
    std::string hypothesis;
    std::string reference;
    float wer;
    float cer;
    double processing_time_ms;
    double audio_duration_sec;
    int substitutions;
    int deletions;
    int insertions;
    int total_words;
};

void save_results_json(const std::string& output_file, const std::vector<BenchmarkResult>& results) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    // Calculate summary stats
    double total_wer = 0.0, total_cer = 0.0;
    double total_audio_duration = 0.0, total_processing_time = 0.0;

    std::vector<float> wer_values;
    for (const auto& r : results) {
        total_wer += r.wer;
        total_cer += r.cer;
        total_audio_duration += r.audio_duration_sec;
        total_processing_time += r.processing_time_ms / 1000.0;
        wer_values.push_back(r.wer);
    }

    float avg_wer = results.empty() ? 0.0f : total_wer / results.size();
    float avg_cer = results.empty() ? 0.0f : total_cer / results.size();

    // Calculate median WER
    std::sort(wer_values.begin(), wer_values.end());
    float median_wer = wer_values.empty() ? 0.0f : wer_values[wer_values.size() / 2];

    float overall_rtfx = total_processing_time > 0 ? total_audio_duration / total_processing_time : 0.0f;

    // Write JSON
    out << "{\n";
    out << "  \"summary\": {\n";
    out << "    \"files_processed\": " << results.size() << ",\n";
    out << "    \"average_wer\": " << avg_wer << ",\n";
    out << "    \"median_wer\": " << median_wer << ",\n";
    out << "    \"average_cer\": " << avg_cer << ",\n";
    out << "    \"overall_rtfx\": " << overall_rtfx << ",\n";
    out << "    \"total_audio_duration\": " << total_audio_duration << ",\n";
    out << "    \"total_processing_time\": " << total_processing_time << "\n";
    out << "  },\n";
    out << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"file_id\": \"" << r.file_id << "\",\n";
        out << "      \"audio_path\": \"" << r.audio_path << "\",\n";
        out << "      \"hypothesis\": \"" << r.hypothesis << "\",\n";
        out << "      \"reference\": \"" << r.reference << "\",\n";
        out << "      \"wer\": " << r.wer << ",\n";
        out << "      \"cer\": " << r.cer << ",\n";
        out << "      \"processing_time_ms\": " << r.processing_time_ms << ",\n";
        out << "      \"audio_duration_sec\": " << r.audio_duration_sec << ",\n";
        out << "      \"substitutions\": " << r.substitutions << ",\n";
        out << "      \"deletions\": " << r.deletions << ",\n";
        out << "      \"insertions\": " << r.insertions << ",\n";
        out << "      \"total_words\": " << r.total_words << "\n";
        out << "    }";
        if (i < results.size() - 1) out << ",";
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    // Parse arguments
    int max_files = 25;
    std::string device = "CPU";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max-files" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "all") {
                max_files = -1;
            } else {
                max_files = std::stoi(val);
            }
        } else if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --max-files <N|all>  Number of files to process (default: 25)\n";
            std::cout << "  --device <device>    OpenVINO device: CPU, GPU, AUTO (default: CPU)\n";
            std::cout << "  --help              Show this help\n";
            return 0;
        }
    }

    std::cout << std::string(80, '=') << "\n";
    std::cout << "eddy LibriSpeech ASR Benchmark (C++ Native)\n";
    std::cout << std::string(80, '=') << "\n\n";

    try {
        // Find dataset directory
        fs::path dataset_dir = fs::path(std::getenv("LOCALAPPDATA")) / "eddy" / "datasets" / "LibriSpeech" / "test-clean";

        if (!fs::exists(dataset_dir)) {
            std::cerr << "ERROR: LibriSpeech test-clean not found at: " << dataset_dir << "\n";
            std::cerr << "Please run the Python benchmark first to download the dataset:\n";
            std::cerr << "  python benchmark_librispeech.py --max-files 5\n";
            return 1;
        }

        // Load test files
        std::cout << "Loading test files";
        if (max_files > 0) {
            std::cout << " (max: " << max_files << ")";
        }
        std::cout << "...\n";

        auto test_files = load_librispeech_transcripts(dataset_dir, max_files);
        std::cout << "[OK] Found " << test_files.size() << " files\n\n";

        // Initialize OpenVINO backend
        std::cout << "Initializing OpenVINO backend (" << device << ") ... ";
        std::cout.flush();
        auto backend = std::make_shared<eddy::OpenVINOBackend>(
            eddy::OpenVINOOptions{.device = device}
        );
        std::cout << "[OK]\n";

        // Load models
        auto cache_model_dir = eddy::get_model_files_dir("parakeet-v2");
        fs::path model_dir;

        if (fs::exists(cache_model_dir / "parakeet_encoder.xml")) {
            model_dir = cache_model_dir;
            std::cout << "Using cached models\n";
        } else {
            model_dir = "models/parakeet";
            std::cout << "Using local models\n";
        }

        eddy::parakeet::ModelPaths paths{
            .preprocessor = {.path = (model_dir / "parakeet_melspectogram.xml").string()},
            .encoder = {.path = (model_dir / "parakeet_encoder.xml").string()},
            .decoder = {.path = (model_dir / "parakeet_decoder.xml").string()},
            .joint = {.path = (model_dir / "parakeet_joint.xml").string()},
            .tokenizer_json = (model_dir / "parakeet_vocab.json").string()
        };

        eddy::parakeet::RuntimeConfig cfg{
            .device = device,
            .blank_token_id = 1024,
            .duration_bins = {0, 1, 2, 3, 4}
        };

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

        // Run benchmark
        std::cout << "Processing " << test_files.size() << " files...\n";
        std::cout << std::string(80, '-') << "\n";

        std::vector<BenchmarkResult> results;
        double total_audio_duration = 0.0;
        double total_processing_time = 0.0;
        auto benchmark_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < test_files.size(); ++i) {
            const auto& test_file = test_files[i];

            try {
                // Convert FLAC to WAV
                auto wav_path = convert_flac_to_wav(test_file.audio_path);

                // Load audio
                auto audio_samples = eddy::audio::read_wav(wav_path.string());
                double audio_duration = audio_samples.size() / 16000.0;

                // Transcribe
                eddy::parakeet::AudioSegment segment;
                segment.sample_rate = 16000;
                segment.pcm = audio_samples;

                eddy::parakeet::SegmentOptions options;

                auto start = std::chrono::high_resolution_clock::now();
                auto result = model->infer(segment, options);
                auto end = std::chrono::high_resolution_clock::now();

                double proc_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                // Calculate metrics
                auto wer_metrics = calculate_wer(result.text, test_file.reference);
                float cer = calculate_cer(result.text, test_file.reference);

                // Store result
                results.push_back({
                    test_file.file_id,
                    test_file.audio_path.string(),
                    result.text,
                    test_file.reference,
                    wer_metrics.wer,
                    cer,
                    proc_time_ms,
                    audio_duration,
                    wer_metrics.substitutions,
                    wer_metrics.deletions,
                    wer_metrics.insertions,
                    wer_metrics.total_words
                });

                total_audio_duration += audio_duration;
                total_processing_time += proc_time_ms / 1000.0;

                // Calculate running averages
                double running_wer = 0.0;
                for (const auto& r : results) running_wer += r.wer;
                running_wer /= results.size();

                double running_rtfx = total_audio_duration / total_processing_time;

                // Calculate ETA
                auto elapsed = std::chrono::high_resolution_clock::now() - benchmark_start;
                double elapsed_sec = std::chrono::duration<double>(elapsed).count();
                double avg_time_per_file = elapsed_sec / (i + 1);
                double eta_sec = avg_time_per_file * (test_files.size() - i - 1);
                int eta_min = static_cast<int>(eta_sec / 60);
                int eta_sec_rem = static_cast<int>(eta_sec) % 60;

                double rtfx = audio_duration / (proc_time_ms / 1000.0);

                // Progress report
                std::cout << "[" << (i + 1) << "/" << test_files.size() << "] "
                          << test_file.file_id << ": "
                          << "WER=" << std::fixed << std::setprecision(1) << (wer_metrics.wer * 100) << "% "
                          << "RTFx=" << std::setprecision(1) << rtfx << "x "
                          << "(avg WER=" << std::setprecision(2) << (running_wer * 100) << "% "
                          << "RTFx=" << std::setprecision(1) << running_rtfx << "x) "
                          << "ETA=" << eta_min << "m" << eta_sec_rem << "s\n";

                // Show detailed error if WER > 10%
                if (wer_metrics.wer > 0.10) {
                    std::cout << "  REF: " << wer_metrics.normalized_reference << "\n";
                    std::cout << "  HYP: " << wer_metrics.normalized_hypothesis << "\n";
                }

            } catch (const std::exception& e) {
                std::cerr << "[" << (i + 1) << "/" << test_files.size() << "] "
                          << test_file.file_id << ": ERROR - " << e.what() << "\n";
            }
        }

        // Print summary
        double avg_wer = 0.0, avg_cer = 0.0;
        std::vector<float> wer_values;
        for (const auto& r : results) {
            avg_wer += r.wer;
            avg_cer += r.cer;
            wer_values.push_back(r.wer);
        }
        avg_wer /= results.size();
        avg_cer /= results.size();

        std::sort(wer_values.begin(), wer_values.end());
        float median_wer = wer_values[wer_values.size() / 2];
        float overall_rtfx = total_audio_duration / total_processing_time;

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BENCHMARK RESULTS\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << "Files processed:      " << results.size() << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Average WER:          " << (avg_wer * 100) << "%\n";
        std::cout << "Median WER:           " << (median_wer * 100) << "%\n";
        std::cout << "Average CER:          " << (avg_cer * 100) << "%\n";
        std::cout << std::setprecision(1);
        std::cout << "Overall RTFx:         " << overall_rtfx << "x\n";
        std::cout << std::setprecision(1);
        std::cout << "Total audio:          " << total_audio_duration << "s\n";
        std::cout << "Total processing:     " << total_processing_time << "s\n";
        std::cout << std::string(80, '=') << "\n\n";

        // Comparison
        std::cout << "Expected (FluidAudio v2): 2.2% WER, 141x RTFx\n";
        std::cout << std::setprecision(1);
        std::cout << "Your eddy performance:    " << (avg_wer * 100) << "% WER, " << overall_rtfx << "x RTFx\n\n";

        if (avg_wer < 0.05) {
            std::cout << "[EXCELLENT] Accuracy is excellent!\n";
        } else if (avg_wer < 0.10) {
            std::cout << "[ACCEPTABLE] Accuracy is acceptable\n";
        } else {
            std::cout << "[POOR] Accuracy needs investigation\n";
        }

        // Save JSON
        std::string output_file = "eddy_benchmark_results_cpp.json";
        save_results_json(output_file, results);
        std::cout << "\n[OK] Results saved to " << output_file << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}
