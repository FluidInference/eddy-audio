// OpenVINO utility functions for model compilation and configuration

#include "eddy/utils/openvino_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace eddy::parakeet {

namespace {

// Cached environment variable checks (computed once per program execution)

struct CompileConfig {
  ov::AnyMap config;
  bool initialized = false;
};

// Build OpenVINO compile config from environment variables (cached)
ov::AnyMap make_compile_cfg_from_env() {
  static CompileConfig cached = []() {
    CompileConfig result;
    ov::AnyMap& cfg = result.config;

    if (const char* perf = std::getenv("EDDY_OV_PERF")) {
      std::string v(perf);
      for (auto& c : v) c = static_cast<char>(::toupper(c));
      if (v == "LATENCY") cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
      else if (v == "THROUGHPUT") cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    }
    if (const char* nr = std::getenv("EDDY_OV_NUM_REQUESTS")) {
      try {
        int n = std::max(1, std::stoi(nr));
        cfg[ov::hint::num_requests.name()] = n;
      } catch (const std::exception& e) {
        std::cerr << "[WARN] Invalid EDDY_OV_NUM_REQUESTS value '" << nr << "', using default\n";
      }
    }
    if (const char* th = std::getenv("EDDY_OV_THREADS")) {
      try {
        int n = std::max(1, std::stoi(th));
        cfg[ov::inference_num_threads.name()] = n;
      } catch (const std::exception& e) {
        std::cerr << "[WARN] Invalid EDDY_OV_THREADS value '" << th << "', using default\n";
      }
    }
    // Removed precision hint: rely on device defaults and model precision.
    result.initialized = true;
    return result;
  }();

  return cached.config;
}

// Check if debug logging is enabled (cached for performance)
bool is_debug_enabled() {
  static bool cached = (std::getenv("EDDY_DEBUG") != nullptr);
  return cached;
}

// Convert string to uppercase for case-insensitive comparison
std::string to_upper(const std::string& s) {
  std::string result = s;
  for (auto& c : result) {
    c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));
  }
  return result;
}

}  // anonymous namespace

ov::CompiledModel compile_component(ov::Core& core, const ModelFile& file, const std::string& device) {
  if (file.path.empty()) {
    throw std::invalid_argument("Parakeet component path is empty");
  }
  if (device.empty()) {
    throw std::invalid_argument("Device string is empty");
  }

  auto cfg = make_compile_cfg_from_env();

  if (file.compiled) {
    std::ifstream blob_stream(file.path, std::ios::binary);
    if (!blob_stream.good()) {
      throw std::runtime_error("Failed to open compiled blob: " + file.path);
    }
    if (!cfg.empty()) return core.import_model(blob_stream, device, cfg);
    return core.import_model(blob_stream, device);
  }

  if (!cfg.empty()) return core.compile_model(file.path, device, cfg);
  return core.compile_model(file.path, device);
}

ov::CompiledModel compile_with_npu_fallback(ov::Core& core,
                                            const ModelFile& file,
                                            const std::string& device,
                                            const char* component_name) {
  // Case-insensitive device comparison
  const bool target_npu = (to_upper(device) == "NPU");

  if (!target_npu) {
    return compile_component(core, file, device);
  }

  try {
    return compile_component(core, file, "NPU");
  } catch (const std::exception& e) {
    if (is_debug_enabled()) {
      std::cerr << "[DEBUG] NPU compile failed";
      if (component_name) {
        std::cerr << " for " << component_name;
      }
      std::cerr << ": " << e.what() << ", falling back to CPU\n";
    }
    return compile_component(core, file, "CPU");
  }
}

}  // namespace eddy::parakeet
