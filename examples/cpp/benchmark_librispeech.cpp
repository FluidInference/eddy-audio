// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/core/app_dir.hpp"
#include "eddy/models/parakeet-v2/parakeet.hpp"
#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "eddy/utils/ensure_models.hpp"
#include "eddy/utils/audio_utils.hpp"

#include "text_normalizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct Args {
  std::string device = "CPU";
  std::string model = []{
    const char* env = std::getenv("EDDY_PARAKEET_MODEL");
    return env && *env ? std::string(env) : std::string("parakeet-v2");
  }();
  size_t max_files = 25;
  bool debug = false;
};

static void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [--device CPU|GPU|NPU|AUTO] [--model parakeet-v2|parakeet-v3] [--max-files N] [--debug]\n";
}

static std::vector<std::string> split_words(const std::string& s) {
  std::istringstream iss(s);
  std::vector<std::string> out;
  std::string w;
  while (iss >> w) out.push_back(w);
  return out;
}

static int word_levenshtein(const std::vector<std::string>& ref, const std::vector<std::string>& hyp) {
  const int n = static_cast<int>(ref.size());
  const int m = static_cast<int>(hyp.size());
  std::vector<int> dp(m + 1);
  for (int j = 0; j <= m; ++j) dp[j] = j;
  for (int i = 1; i <= n; ++i) {
    int prev = dp[0];
    dp[0] = i;
    for (int j = 1; j <= m; ++j) {
      int tmp = dp[j];
      int cost = (ref[i - 1] == hyp[j - 1]) ? 0 : 1;
      dp[j] = std::min({ dp[j] + 1, dp[j - 1] + 1, prev + cost });
      prev = tmp;
    }
  }
  return dp[m];
}

static std::map<std::string, std::string> load_transcripts(const fs::path& trans_file) {
  std::map<std::string, std::string> t;
  std::ifstream in(trans_file);
  if (!in.good()) return t;
  std::string line;
  while (std::getline(in, line)) {
    auto pos = line.find(' ');
    if (pos == std::string::npos) continue;
    std::string id = line.substr(0, pos);
    std::string text = line.substr(pos + 1);
    t.emplace(std::move(id), std::move(text));
  }
  return t;
}

static fs::path default_testclean_dir() {
#if defined(_WIN32)
  char* local = std::getenv("LOCALAPPDATA");
  if (local && *local) {
    return fs::path(local) / "eddy" / "datasets" / "LibriSpeech" / "test-clean";
  }
#endif
  return fs::path();
}

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    else if (a == "--device" && i + 1 < argc) { args.device = argv[++i]; }
    else if (a == "--model" && i + 1 < argc) { args.model = argv[++i]; }
    else if (a == "--max-files" && i + 1 < argc) { args.max_files = static_cast<size_t>(std::stoul(argv[++i])); }
    else if (a == "--debug") { args.debug = true; }
  }

  if (args.debug) {
#if defined(_WIN32)
    _putenv_s("EDDY_DEBUG", "1");
#else
    setenv("EDDY_DEBUG", "1", 1);
#endif
  }

  // Backend (compiled model cache per model)
  eddy::OpenVINOOptions ov_opts;
  ov_opts.device = args.device;
  ov_opts.cache_dir = eddy::get_model_dir(args.model).string();
  auto backend = std::make_shared<eddy::OpenVINOBackend>(ov_opts);

  // Resolve model files (prefer cache)
  auto assets = eddy::get_model_assets_dir(args.model);
  std::string err;
  (void)eddy::parakeet::check_models_available(
    assets, &err,
    (args.model == "parakeet-v3") ? eddy::model_configs::PARAKEET_V3_FILES
                                   : eddy::model_configs::PARAKEET_STANDARD_FILES);

  auto exists_nonempty = [](const fs::path& p) {
    std::error_code ec; auto sz = fs::file_size(p, ec); return !ec && sz > 0; };
  fs::path model_dir = exists_nonempty(assets / "parakeet_encoder.xml") ? assets : fs::path("models/parakeet");

  eddy::parakeet::ModelPaths paths{
    .preprocessor = {.path = (model_dir / "parakeet_melspectogram.xml").string()},
    .encoder = {.path = (model_dir / "parakeet_encoder.xml").string()},
    .decoder = {.path = (model_dir / "parakeet_decoder.xml").string()},
    .joint = {.path = (model_dir / "parakeet_joint.xml").string()},
    .tokenizer_json = (args.model == "parakeet-v3") ? (model_dir / "parakeet_v3_vocab.json").string()
                                                     : (model_dir / "parakeet_vocab.json").string()
  };

  eddy::parakeet::RuntimeConfig cfg{
    .device = args.device,
    .blank_token_id = (args.model == "parakeet-v3") ? 8192 : 1024,
    .duration_bins = {0,1,2,3,4}
  };

  std::cout << "Initializing OpenVINO backend (" << args.device << ") ... [OK]\n";
  std::cout << "Using cached models at: " << model_dir.string() << "\n";
  std::cout << "Loading Parakeet models ... ";
  auto model = eddy::parakeet::make_openvino_parakeet(backend, paths, cfg);
  std::cout << "[OK]\n";
  std::static_pointer_cast<eddy::parakeet::OpenVINOParakeet>(model)->warmup();

  // Dataset discovery (Windows local cache)
  fs::path test_dir = default_testclean_dir();
  if (test_dir.empty() || !fs::exists(test_dir)) {
    std::cerr << "ERROR: LibriSpeech test-clean not found. Expected at %LOCALAPPDATA%/eddy/datasets/LibriSpeech/test-clean\n";
    return 1;
  }
  std::cout << "Using dataset at: \"" << test_dir.string() << "\"\n\n";

  // Build file list
  struct Sample { fs::path path; std::string id; std::string ref; };
  std::vector<Sample> samples;
  for (auto& spk : fs::directory_iterator(test_dir)) if (spk.is_directory()) {
    for (auto& chap : fs::directory_iterator(spk.path())) if (chap.is_directory()) {
      auto trans = load_transcripts(chap.path() / (spk.path().filename().string() + "-" + chap.path().filename().string() + ".trans.txt"));
      for (auto& f : fs::directory_iterator(chap.path())) {
        if (!f.is_regular_file()) continue;
        auto ext = f.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".flac" || ext == ".wav") {
          std::string id = f.path().stem().string();
          auto it = trans.find(id);
          if (it != trans.end()) samples.push_back({f.path(), id, it->second});
        }
      }
    }
  }
  std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b){ return a.id < b.id; });
  if (samples.empty()) { std::cerr << "ERROR: No audio files found in dataset" << std::endl; return 1; }

  size_t limit = std::min(args.max_files, samples.size());
  std::cout << "Processing " << limit << " files...\n";
  std::cout << std::string(80, '-') << "\n";

  eddy::TextNormalizer normalizer;
  // Optional English variants mapping (if available)
  normalizer.load_variants_json("FluidAudio/Sources/FluidAudioCLI/Utils/english.json");

  double total_audio = 0.0;
  double total_proc = 0.0;
  int total_edits = 0;
  int total_words = 0;


  for (size_t i = 0; i < limit; ++i) {
    const auto& s = samples[i];
    auto t0 = std::chrono::steady_clock::now();

    // Read audio via libsndfile (resample to 16k)
    auto pcm = eddy::audio::read_wav(s.path.string());

    eddy::parakeet::AudioSegment seg;
    seg.sample_rate = 16000;
    seg.pcm = pcm;

    auto st = std::chrono::steady_clock::now();
    auto res = model->infer(seg, {});
    auto et = std::chrono::steady_clock::now();

    double dur_s = static_cast<double>(pcm.size()) / 16000.0;
    double proc_s = std::chrono::duration<double>(et - st).count();
    total_audio += dur_s;
    total_proc += proc_s;

    // Normalize and compute WER
    std::string ref_norm = normalizer.normalize(s.ref);
    std::string hyp_norm = normalizer.normalize(res.text);
    auto ref_words = split_words(ref_norm);
    auto hyp_words = split_words(hyp_norm);
    int edits = word_levenshtein(ref_words, hyp_words);
    int words = static_cast<int>(ref_words.size());
    total_edits += edits;
    total_words += words;
    double wer = words > 0 ? (100.0 * edits / words) : 0.0;

    double rtfx = proc_s > 0.0 ? (dur_s / proc_s) : 0.0;
    std::cout << "[" << (i+1) << "/" << limit << "] " << s.id << ": WER=" << std::fixed << std::setprecision(1)
              << wer << "% RTFx=" << std::setprecision(1) << rtfx
              << " (avg WER=" << std::setprecision(2) << (total_words ? (100.0 * total_edits / total_words) : 0.0)
              << "% RTFx=" << std::setprecision(1) << (total_proc > 0 ? (total_audio / total_proc) : 0.0) << ")\n";

  }

  double overall_wer = total_words ? (100.0 * total_edits / total_words) : 0.0;
  double overall_rtfx = total_proc > 0.0 ? (total_audio / total_proc) : 0.0;

  std::cout << "\n" << std::string(80, '=') << "\n";
  std::cout << "BENCHMARK RESULTS" << "\n";
  std::cout << std::string(80, '=') << "\n";
  std::cout << "Files processed:      " << limit << "\n";
  std::cout << "Average WER:          " << std::fixed << std::setprecision(2) << overall_wer << "%\n";
  std::cout << "Overall RTFx:         " << std::setprecision(1) << overall_rtfx << "x\n";
  std::cout << "Total audio:          " << std::setprecision(1) << total_audio << "s\n";
  std::cout << "Total processing:     " << std::setprecision(1) << total_proc << "s\n";
  std::cout << std::string(80, '=') << "\n\n";

  std::cout << "(No JSON saved; run Python benchmark for detailed output file)\n";
  return 0;
}
