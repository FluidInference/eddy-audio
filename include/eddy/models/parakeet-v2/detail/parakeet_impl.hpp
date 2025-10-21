#pragma once

#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"
#include "eddy/models/parakeet-v2/tokenizer.hpp"

#include <openvino/openvino.hpp>
#include <mutex>
#include <string>

namespace eddy::parakeet {

// PRIVATE IMPLEMENTATION HEADER - DO NOT INSTALL OR INCLUDE IN PUBLIC API
//
// This file contains internal implementation details for OpenVINOParakeet.
// It is kept in detail/ to hide implementation from users:
// - OpenVINO types (ov::CompiledModel, ov::InferRequest)
// - Internal state management (mutexes, port indices)
// - Implementation can change without breaking API
//
// Only parakeet_*.cpp files should include this header.

// Implementation struct for OpenVINOParakeet (Pimpl idiom)
// Also used directly by helper functions as ParakeetImpl alias
struct ParakeetImpl {
  std::shared_ptr<eddy::OpenVINOBackend> backend;
  ModelPaths model_paths;
  RuntimeConfig runtime_cfg;

  Tokenizer tokenizer;

  ov::CompiledModel preproc_model;
  ov::InferRequest preproc_request;

  ov::CompiledModel encoder_model;
  ov::InferRequest encoder_request;

  ov::CompiledModel decoder_model;
  ov::InferRequest decoder_request;

  ov::CompiledModel joint_model;
  ov::InferRequest joint_request;

  size_t encoder_expected_frames = 0;
  size_t encoder_hidden_size = 0;
  size_t decoder_hidden_size = 0;
  size_t joint_output_size = 0;

  std::once_flag compile_once;
  std::mutex request_guard;

  // Resolved encoder ports
  EncoderPorts encoder_ports;

  // Output indices for encoder outputs (robust retrieval)
  size_t encoder_output_index = 0;   // [1, hidden, time]
  size_t encoder_length_index = 1;   // [1]
};

// OpenVINOParakeet::Impl simply inherits from ParakeetImpl
// This satisfies the forward declaration while allowing helper functions to use ParakeetImpl directly
struct OpenVINOParakeet::Impl : ParakeetImpl {};

}  // namespace eddy::parakeet
