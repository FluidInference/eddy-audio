// OpenVINO utility functions for model compilation and configuration

#pragma once

#include "eddy/models/parakeet-v2/parakeet.hpp"
#include <openvino/openvino.hpp>
#include <string>

namespace eddy::parakeet {

// Compile an OpenVINO model from XML/BIN or compiled blob
// Handles both IR format (.xml + .bin) and compiled blobs
// Respects EDDY_OPENVINO_* environment variables for configuration
[[nodiscard]] ov::CompiledModel compile_component(ov::Core& core, const ModelFile& file, const std::string& device);

// Compile with NPU fallback to CPU
// If target device is NPU and compilation fails, automatically falls back to CPU
// Useful for models that may not be supported on all NPU versions
[[nodiscard]] ov::CompiledModel compile_with_npu_fallback(ov::Core& core,
                                                           const ModelFile& file,
                                                           const std::string& device,
                                                           const char* component_name = nullptr);

}  // namespace eddy::parakeet
