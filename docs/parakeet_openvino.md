# Parakeet-TDT v2 OpenVINO Integration (Draft)

This document outlines how to experiment with the Parakeet-TDT 0.6B v2 model using the OpenVINO backend scaffolding in `eddy`.

## Prerequisites
- OpenVINO Runtime 2025.x (September 2025 release or newer) available on your build machine.
- CMake 3.22+
- A C++20-capable compiler (MSVC 2022, Clang 16, GCC 13 or newer).
- Access to the Parakeet OpenVINO artifacts published at `https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-ov` and the shared vocabulary from the CoreML export.

## Fetching the Model
The OpenVINO export is split into four graphs plus a shared vocabulary. Download the XML/BIN pairs (or `.blob` binaries if you pre-compile on device) and the SentencePiece vocabulary:

```
huggingface-cli download FluidInference/parakeet-tdt-0.6b-v2-ov parakeet_melspectogram.xml --local-dir models/parakeet
huggingface-cli download FluidInference/parakeet-tdt-0.6b-v2-ov parakeet_melspectogram.bin --local-dir models/parakeet
huggingface-cli download FluidInference/parakeet-tdt-0.6b-v2-ov parakeet_encoder.xml --local-dir models/parakeet
huggingface-cli download FluidInference/parakeet-tdt-0.6b-v2-ov parakeet_encoder.bin --local-dir models/parakeet
huggingface-cli download FluidInference/parakeet-tdt-0.6b-v2-ov parakeet_decoder.xml --local-dir models/parakeet
huggingface-cli download FluidInference/parakeet-tdt-0.6b-v2-ov parakeet_decoder.bin --local-dir models/parakeet
huggingface-cli download FluidInference/parakeet-tdt-0.6b-v2-ov parakeet_joint.xml --local-dir models/parakeet
huggingface-cli download FluidInference/parakeet-tdt-0.6b-v2-ov parakeet_joint.bin --local-dir models/parakeet
huggingface-cli download FluidInference/parakeet-tdt-0.6b-v2-coreml parakeet_vocab.json --local-dir models/parakeet
```

If you generate compiled blobs ahead of time, set the relevant `ModelFile::compiled` flag when wiring the model paths.

## Building the SDK Skeleton
```
cmake -S . -B build -DOpenVINO_DIR=/opt/intel/openvino/runtime/cmake
cmake --build build
```

The build produces the static library `build/libeddy.a` (platform-specific extension applies).

## Running a Smoke Test
The OpenVINO backend now performs mel extraction, encoder/decoder passes, greedy TDT decoding, and SentencePiece detokenisation. A minimal synchronous transcription looks like this:

```cpp
#include "eddy/backends/openvino_backend.hpp"
#include "eddy/models/parakeet/parakeet_openvino.hpp"

int main() {
  auto backend = std::make_shared<eddy::OpenVINOBackend>(eddy::OpenVINOOptions{.device = "AUTO"});

  eddy::parakeet::ModelPaths paths{
      .preprocessor = {.path = "models/parakeet/parakeet_melspectogram.xml"},
      .encoder = {.path = "models/parakeet/parakeet_encoder.xml"},
      .decoder = {.path = "models/parakeet/parakeet_decoder.xml"},
      .joint = {.path = "models/parakeet/parakeet_joint.xml"},
      .tokenizer_json = "models/parakeet/parakeet_vocab.json",
  };

  eddy::parakeet::RuntimeConfig cfg{
      .device = "GPU",
      .blank_token_id = 1024,
      .duration_bins = {0, 1, 2, 3, 4},
  };

  auto model = eddy::parakeet::make_openvino_parakeet(backend, paths, cfg);
  model->warmup();

  eddy::parakeet::AudioSegment segment;
  segment.sample_rate = 16000;
  segment.pcm.resize(16000 * 5, 0.0f);  // 5 seconds of silence as placeholder input

  auto result = model->infer(segment, {});
  // result.text now contains the decoded transcript (empty for silence input).
}
```

## Next Steps
- Add resampling and streaming helpers so callers can feed non-16 kHz audio and rolling windows without bespoke preprocessing code.
- Extend the greedy decoder to expose token timestamps and confidences, bringing it closer to the full TDT beam search used in production FluidAudio builds.
- Add automated integration tests with short speech clips to guard model compatibility across OpenVINO runtime releases.
