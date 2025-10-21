// OpenVINO utility functions for model compilation and configuration

#pragma once

#include "eddy/models/parakeet-v2/parakeet.hpp"
#include <openvino/openvino.hpp>
#include <string>

namespace eddy::parakeet {

// Compile an OpenVINO model from XML/BIN or compiled blob
// Handles both IR format (.xml + .bin) and compiled blobs
// Respects EDDY_OPENVINO_* environment variables for configuration
ov::CompiledModel compile_component(ov::Core& core, const ModelFile& file, const std::string& device);

}  // namespace eddy::parakeet
