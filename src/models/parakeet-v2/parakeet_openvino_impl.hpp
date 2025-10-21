#pragma once

#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "eddy/models/parakeet-v2/parakeet_encoder.hpp"
#include "eddy/models/parakeet-v2/tokenizer.hpp"

#include <openvino/openvino.hpp>
#include <mutex>
#include <string>

namespace eddy::parakeet {

// Internal implementation struct for OpenVINOParakeet
// This header is for internal use only by the parakeet implementation files
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

  // Preferred port names (configurable via metadata JSON)
  std::string enc_mel_name = "melspectogram";
  std::string enc_len_name = "melspectogram_length";
  std::string enc_out_name = "encoder_output";
  std::string enc_len_out_name = "encoder_output_length";
};

// Alias for OpenVINOParakeet::Impl
struct OpenVINOParakeet::Impl : ParakeetImpl {};

}  // namespace eddy::parakeet
