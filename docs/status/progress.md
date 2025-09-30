# eddy Progress Tracker

This document captures what is already in place and what still needs to ship to satisfy the original goals for the cross-platform edge inference SDK (Intel, Qualcomm, AMD NPUs) and multi-language bindings.

## What’s Done

### Core SDK Framework
- C++20 project scaffold with `eddy::Runtime`, a generic backend interface, and the initial OpenVINO backend implementation.
- Parakeet model abstractions (`ModelPaths`, runtime configuration, segment/result types) shared between implementations.
- Build system (CMake) installs headers/lib and locates OpenVINO runtime.

### Parakeet (OpenVINO) Path
- Wiring for four OpenVINO components (melspectrogram, encoder, decoder, joint) with per-stage `InferRequest`s.
- SentencePiece tokenizer loader (using shared `parakeet_vocab.json`) and greedy detokenisation to text.
- End-to-end synchronous inference flow for 16 kHz PCM input (mel extraction → encoder → decoder → joint → tokens → text).
- Documentation updated to outline asset download steps and smoke-test sample.

### Tooling & Third-Party
- Vendored `nlohmann/json.hpp` header so the tokenizer/parsing path builds without extra dependencies.

## What’s Still Outstanding

### Parakeet Model (OpenVINO)
- Streaming window management (context stitching, timestamp/confidence emission, segmentation similar to FluidAudio pipeline).
- Duration-aware fast-forward logic (full TDT inner-loop + force blank handling) and configurable beam search.
- Audio frontend enhancements: on-device resampling, voice activity trimming, normalization.
- Integration tests and audio fixtures to catch regression across OpenVINO releases/devices.

### Qualcomm (QNN / QISDK) Support
- Stand up Qualcomm backend abstraction (QNN runtime init, context/device selection, memory management).
- Convert/export Parakeet (and other target models) into QNN-compatible graphs or leverage existing QISDK bundles.
- Implement Parakeet inference path via QISDK with feature parity to the OpenVINO backend (streaming, tokenizer reuse).
- Validate across Snapdragon NPU SKUs and document setup (firmware requirements, SDK version).

### AMD NPU / ROCm Edge Targets
- Identify AMD NPU runtime or ROCm inference API appropriate for edge deployment.
- Mirror backend interface (model load/execution paths) and ensure Parakeet graph availability/compatibility.
- Performance profiling & fallback behaviour (CPU vs. NPU) per AMD device class.

### Additional Models
- Speaker diarization (Pyannote segmentation & embeddings): port preprocessing, diarizer runtime, OpenVINO/QISDK deployments.
- Voice Activity Detection (VAD): lightweight model integration for both Intel and Qualcomm paths.
- Kokoro TTS: define inference API, asset packaging, backend support, audio post-processing.
- Moondream2 VLM: handle multi-modal inputs, image preprocessing, text decoding; ensure backend compatibility or CPU fallback.
- NVIDIA Parakeet TDT v3 parity (multilingual) and any model variants needed for Whisper-style fallback.

### SDK Surface & Language Bridges
- Define stable C API surface for core SDK (model loading, session/inference lifecycle, streaming callbacks).
- Generate bindings for C#, Kotlin (JVM + Android), Rust (crate with safe wrappers), and TypeScript (Node/WebAssembly or native addon).
- Package/distribution strategy for desktop (macOS/Windows/Linux) and mobile targets (iOS/Android) including model asset delivery.

### Runtime & Deployment Concerns
- Configuration system (YAML/JSON) to declare model assets, cache locations, device preferences per platform.
- Telemetry/logging hooks suitable for partner integrations while remaining opt-in/offline friendly.
- Warmup/caching support (compiled blobs, startup latency reduction) per backend/device.
- CI pipelines with matrix builds (Intel CPU/GPU, simulated Qualcomm targets, AMD where feasible) and automated regression tests.
- Licensing/compliance review for redistributed model assets and third-party dependencies.

### Documentation & Samples
- Comprehensive developer guide covering SDK architecture, backend requirements, and troubleshooting for each vendor runtime.
- Sample applications per language binding (desktop CLI, mobile demo app, web harness).
- Benchmarking guide comparing inference latency/accuracy across devices and models.

---

_Last updated: 2025-09-28_
