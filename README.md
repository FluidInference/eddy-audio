# eddy (Work In Progress)

eddy is a C++ inference library designed for native runtimes and multi-vendor edge NPUs, exposing a consistent C++ API plus language bindings for app developers (C#; more to follow). The current milestone focuses on the OpenVINO 2025.x backend for the Parakeet-TDT speech model family while we bring additional runtimes online.

## Platform Support

- Supported: Windows and Linux.
- Not supported: Apple platforms (macOS/iOS). For Apple, use FluidAudio (FA).

## Repository Layout

- `include/` – public headers for the runtime, backend abstractions, and model bridges.
- `src/` – backend/runtime implementations and model-specific glue code.
- `docs/` – design notes and usage guides.
- `benchmarks/` – Python scripts for LibriSpeech ASR benchmarking (see [benchmarks/README.md](benchmarks/README.md)).

## Dependencies

### Required
- **OpenVINO** (2025.x) - AI inference runtime
- **libsndfile** - Audio file I/O (WAV, FLAC, OGG, etc.)
- **libsamplerate** - High-quality audio resampling

### Optional
- **OpenVINO GenAI** - For Whisper support

### Installing with vcpkg (recommended)

```bash
# Install vcpkg dependencies (uses vcpkg.json manifest)
vcpkg install

# Or manually install specific packages
vcpkg install openvino libsndfile libsamplerate
```

## Building

```bash
# Configure with vcpkg toolchain
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake

# Or specify OpenVINO manually if not using vcpkg
cmake -S . -B build -DOpenVINO_DIR=/opt/intel/openvino/runtime/cmake

# Build
cmake --build build --config Release
```

The build emits the static target `eddy` with all required dependencies.

### Optional Whisper Support

```bash
cmake -S . -B build -DEDDY_ENABLE_WHISPER=ON -DOpenVINOGenAI_DIR="<path-to-genai-cmake>"
```

## Models (auto-download on first run)

Models will automatically download from `FluidInference/parakeet-tdt-0.6b-v2-ov` on first use.

Cached at: `%LOCALAPPDATA%\eddy\models\parakeet-v2\files\`

Manual download: Run `hf_fetch_models.exe` or visit <https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-ov>

To disable auto-download: set `EDDY_DISABLE_AUTO_FETCH=1`

## Parakeet OpenVINO Prototype

Refer to `docs/parakeet_openvino.md` for instructions on pulling the exported model from Hugging Face and running a smoke test through the new `OpenVINOParakeet` wrapper.

## Roadmap Snapshot

- Flesh out Parakeet preprocessing (feature pipeline, tokenizer, decoder).
- Add telemetry and zero-copy buffers per backend.
- Introduce unit and integration tests (GoogleTest) with small audio fixtures.
- Extend the backend layer to Qualcomm QNN and AMD MIGraphX once the OpenVINO path is validated.
