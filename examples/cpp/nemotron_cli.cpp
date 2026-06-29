// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0
//
// CLI for the NVIDIA Nemotron cache-aware streaming ASR backend: the
// 3.5-ASR-Streaming-Multilingual 0.6B model and the English speech-streaming
// 0.6B model (selected via --model).

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/core/app_dir.hpp"
#include "eddy/models/nemotron/nemotron.hpp"
#include "eddy/utils/audio_utils.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " <audio.wav> [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --device <device>   OpenVINO device (default: CPU). CPU, AUTO, NPU\n";
  std::cout << "  --lang <code>       Language: en-US, zh-CN, ... or auto (default: auto).\n";
  std::cout << "                      Ignored by the English speech-streaming model.\n";
  std::cout << "  --model <name>      Model variant (selects the cache dir):\n";
  std::cout << "                        nemotron-streaming[-int8]         multilingual (40+ langs)\n";
  std::cout << "                        nemotron-speech-streaming[-int8]  English, no language prompt\n";
  std::cout << "                      Default: nemotron-streaming (FP16).\n";
  std::cout << "  --model-dir <dir>   Directory with nemotron_*.xml/bin + metadata.json\n";
  std::cout << "                      (overrides --model; default: cache for the --model variant)\n";
  std::cout << "  --help              Show this help\n";
}

int main(int argc, char* argv[]) {
  std::cout.setf(std::ios::unitbuf);
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string audio_file, device = "CPU", lang = "auto", model_dir_arg,
              model_name = "nemotron-streaming";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    // A flag that needs a value but is the last arg must error, not fall through
    // to the positional branch (which would swallow the flag as the audio path).
    auto take_value = [&](const char* flag, std::string& dst) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << flag << " requires an argument\n";
        return false;
      }
      dst = argv[++i];
      return true;
    };
    if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (a == "--device") {
      if (!take_value("--device", device)) return 1;
    } else if (a == "--lang") {
      if (!take_value("--lang", lang)) return 1;
    } else if (a == "--model") {
      if (!take_value("--model", model_name)) return 1;
    } else if (a == "--model-dir") {
      if (!take_value("--model-dir", model_dir_arg)) return 1;
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "Error: unknown option " << a << "\n\n";
      print_usage(argv[0]);
      return 1;
    } else {
      audio_file = a;
    }
  }
  if (audio_file.empty()) {
    std::cerr << "Error: no audio file specified\n\n";
    print_usage(argv[0]);
    return 1;
  }

  std::cout << "=== Nemotron ASR Streaming CLI ===\n\n";

  try {
    auto pcm = eddy::audio::read_wav(audio_file);
    const float audio_seconds = static_cast<float>(pcm.size()) / 16000.0f;
    std::cout << "Audio: " << audio_file << "  (" << std::fixed << std::setprecision(2)
              << audio_seconds << "s)\n";

    std::filesystem::path model_dir =
        model_dir_arg.empty() ? eddy::get_model_assets_dir(model_name)
                              : std::filesystem::path(model_dir_arg);
    std::cout << "Model:  " << model_name << "\n";
    std::cout << "Models: " << model_dir.string() << "\n";

    eddy::OpenVINOOptions ov_opts;
    ov_opts.device = device;
    ov_opts.cache_dir = eddy::get_model_dir(model_name).string();
    auto backend = std::make_shared<eddy::OpenVINOBackend>(ov_opts);

    eddy::nemotron::ModelPaths paths{
        .preprocessor = (model_dir / "nemotron_preprocessor.xml").string(),
        .encoder = (model_dir / "nemotron_encoder.xml").string(),
        .decoder = (model_dir / "nemotron_decoder.xml").string(),
        .joint = (model_dir / "nemotron_joint.xml").string(),
        .vocab_json = (model_dir / "nemotron_vocab.json").string(),
        .metadata_json = (model_dir / "metadata.json").string(),
    };
    eddy::nemotron::Config cfg;
    cfg.device = device;
    cfg.language = lang;

    eddy::nemotron::OpenVINONemotron model(backend, paths, cfg);
    std::cout << "Compiling + warming up (" << device << ") ... ";
    model.warmup();
    std::cout << "[OK]\n\n";

    std::cout << std::string(70, '=') << "\nTRANSCRIBING...\n" << std::string(70, '=') << "\n\n";
    const auto result = model.transcribe(pcm);

    const float rtfx = result.latency_ms > 0.0
                           ? audio_seconds / static_cast<float>(result.latency_ms / 1000.0)
                           : 0.0f;

    std::cout << "Result:\n" << std::string(70, '-') << "\n";
    std::cout << result.text << "\n" << std::string(70, '-') << "\n\n";
    std::cout << "prompt_id_used:  " << result.prompt_id_used << "\n";
    std::cout << "detected_lang:   " << (result.detected_language.empty() ? "(none)" : result.detected_language) << "\n";
    std::cout << "tokens:          " << result.token_ids.size() << "\n";
    std::cout << "processing time: " << std::fixed << std::setprecision(0) << result.latency_ms << " ms\n";
    std::cout << "real-time factor:" << std::fixed << std::setprecision(1) << rtfx << "x\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\n[ERROR] " << e.what() << "\n";
    return 1;
  }
}
