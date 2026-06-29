// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0
//
// NVIDIA Nemotron-3.5-ASR-Streaming-Multilingual 0.6B backend.
//
// Unlike the Parakeet TDT path (stateless encoder + overlapping-chunk
// dedup + token/duration heads), Nemotron is a *cache-aware streaming*
// FastConformer-RNNT: the encoder carries cache tensors across chunks and
// takes an int `prompt_id` for language conditioning. Decoding is plain
// RNNT (no duration head). This warrants a separate module rather than
// overloading the Parakeet pipeline.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace eddy {
class OpenVINOBackend;
}

namespace eddy::nemotron {

/// Paths to the exported OpenVINO IR plus tokenizer/metadata.
/// File layout matches export_openvino.py / the HF model repo.
struct ModelPaths {
  std::string preprocessor;  // nemotron_preprocessor.xml  (audio -> mel)
  std::string encoder;       // nemotron_encoder.xml       (mel + caches + prompt_id -> encoded + caches)
  std::string decoder;       // nemotron_decoder.xml       (token + lstm state -> dec_out + state)
  std::string joint;         // nemotron_joint.xml         (enc_step + dec_step -> logits)
  std::string vocab_json;    // nemotron_vocab.json        (id -> piece)
  std::string metadata_json; // metadata.json              (shapes, blank_idx, prompt_dictionary, ...)
};

struct Config {
  // CPU is the default (safe, tested path; matches the eddy_c C API default and
  // the CLI). Set "AUTO"/"NPU"/"GPU" to target other OpenVINO devices.
  std::string device = "CPU";  // OpenVINO device for encoder/decoder/joint (preprocessor always CPU)
  /// Language for prompt conditioning. Accepts dictionary keys ("en-US"),
  /// 2-letter codes ("en" -> first "en-*"), or "auto" (model self-detects).
  std::string language = "auto";
  size_t max_symbols_per_frame = 10;  // RNNT inner-loop safety cap
};

struct TranscriptionResult {
  std::string text;                 // lang-tag tokens stripped
  std::string detected_language;    // first <xx-XX> tag emitted, if any (empty otherwise)
  int prompt_id_used = 0;
  std::vector<int> token_ids;       // raw emitted token ids (pre-strip)
  double latency_ms = 0.0;
};

/// Streaming Nemotron ASR over OpenVINO. Construct, then transcribe whole
/// PCM buffers (internally chunked with cache-aware state continuity).
class OpenVINONemotron {
public:
  OpenVINONemotron(std::shared_ptr<eddy::OpenVINOBackend> backend, ModelPaths paths, Config config);
  ~OpenVINONemotron();

  OpenVINONemotron(const OpenVINONemotron&) = delete;
  OpenVINONemotron& operator=(const OpenVINONemotron&) = delete;

  /// Compile models + load tokenizer/metadata (lazy; called by transcribe()).
  void warmup();

  /// Transcribe 16 kHz mono float32 PCM in [-1, 1].
  TranscriptionResult transcribe(const std::vector<float>& pcm_16k_mono);

  /// Resolve a language string to its integer prompt id using the model's
  /// prompt_dictionary (falls back to "auto").
  [[nodiscard]] int resolve_prompt_id(const std::string& language) const;

  struct Impl;

private:
  // const: only mutates *impl_ (reachable through the unique_ptr in a const
  // method) and is std::call_once-guarded, so resolve_prompt_id() (const) can
  // lazily compile without a const_cast.
  void ensure_compiled() const;
  std::unique_ptr<Impl> impl_;
};

}  // namespace eddy::nemotron
