# eddy

**eddy** is a high-performance audio-based AI library for Windows and Linux optimized for Intel NPUs and CPUs; eddy powers private, offline automatic speech recognition (ASR) with support for multiple hardware accelerators planned.

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![GitHub Stars](https://img.shields.io/github/stars/FluidInference/eddy?style=flat&logo=github)](https://github.com/FluidInference/eddy)

## Overview

- **Private**: Fully on-device inference, no network calls after model download
- **Multilingual**: 24 European languages supported (Parakeet V3)
- **Cross-platform**: Windows 10/11, Linux (Ubuntu 20.04+)
- **NPU Optimized**: Designed for Intel NPUs and CPUs with support for additional accelerators (Qualcomm, AMD) planned
- **Apple devices** (macOS/iOS): [FluidAudio](https://github.com/FluidInference/FluidAudio)

## Quick Start

### Python

```bash
# Clone repository
git clone https://github.com/FluidInference/eddy.git
cd eddy

# Build C++ library
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# Run FLEURS multilingual benchmark
python benchmark_fleurs.py --languages en_us --samples 10 --device NPU

# Or run LibriSpeech benchmark
cd benchmarks
uv run python benchmark.py --max-files 10 --device NPU
```

Models auto-download on first run from HuggingFace. See [benchmark_fleurs.py](benchmark_fleurs.py) and [benchmarks/benchmark.py](benchmarks/benchmark.py) for Python usage via ctypes.

<details>
<summary><b>C++ Build & Usage</b></summary>

For advanced users who want to build from source:

```bash
git clone https://github.com/FluidInference/eddy.git
cd eddy

# Build with vcpkg (handles dependencies automatically)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

**Usage:**

```bash
# Transcribe with Parakeet V2
build/examples/cpp/Release/parakeet_cli.exe audio.wav --model parakeet-v2

# Transcribe with NPU acceleration
build/examples/cpp/Release/parakeet_cli.exe audio.wav --model parakeet-v2 --device NPU

# Transcribe with Parakeet V3 (multilingual)
build/examples/cpp/Release/parakeet_cli.exe audio.wav --model parakeet-v3 --device NPU

# Benchmark on LibriSpeech
build/examples/cpp/Release/benchmark_librispeech.exe --max-files 100 --device NPU

# Benchmark on FLEURS
build/examples/cpp/Release/benchmark_fleurs.exe "%LOCALAPPDATA%\eddy\datasets\FLEURS" --device NPU
```

</details>

## Models & Performance

Benchmarked on Intel Core Ultra 7 155H (Meteor Lake) with Intel AI Boost NPU.

### Parakeet V2 (English)
- **Languages**: English only
- **WER**: 2.76% (LibriSpeech test-clean)
- **Speed**: 38× RTFx (NPU), 5-8× RTFx (CPU)
- **Size**: 600MB
- **Download**: [parakeet-tdt-0.6b-v2-ov](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-ov)

### Parakeet V3 (Multilingual)
- **Languages**: 24 European languages
- **WER**: 6.09% English, 17.0% average across all languages (FLEURS)
- **Speed**: 41× RTFx (NPU), 5-8× RTFx (CPU)
- **Size**: 1.1GB
- **Download**: [parakeet-tdt-1.1b-v3-ov](https://huggingface.co/FluidInference/parakeet-tdt-1.1b-v3-ov)

**Supported Languages**: English, Spanish, Italian, French, German, Dutch, Russian, Polish, Ukrainian, Slovak, Bulgarian, Finnish, Romanian, Croatian, Czech, Swedish, Estonian, Hungarian, Lithuanian, Danish, Maltese, Slovenian, Latvian, Greek

> **RTFx** = Real-Time Factor. 41× means 10 minutes of audio transcribed in ~15 seconds.

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed performance metrics.

## Architecture

eddy uses a 4-model **FastConformer-RNNT** pipeline:

1. **Mel Spectrogram** - Converts raw audio → 80 mel-frequency bins
2. **Encoder** (FastConformer) - Processes acoustic features, outputs embeddings
3. **Decoder** (LSTM) - Prediction network with language model
4. **Joint Network** - Combines encoder + decoder, predicts tokens

**Key Features**:
- LSTM state continuity across audio chunks
- Token deduplication via 2D search algorithm
- Batch chunking: 10s windows with 3s overlap
- Per-token timestamps (80ms granularity) and confidence scores
- Greedy decoding for low-latency inference

## Dependencies

### Required
- **OpenVINO** (2025.x) - AI inference runtime
- **libsndfile** - Audio file I/O (WAV, FLAC, OGG)
- **libsamplerate** - High-quality audio resampling

### Installing with vcpkg (recommended)

```bash
# Install from vcpkg.json manifest
vcpkg install

# Or install manually
vcpkg install openvino libsndfile libsamplerate
```

## Building

```bash
# Configure with vcpkg toolchain
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake

# Or specify OpenVINO manually
cmake -S . -B build -DOpenVINO_DIR=/opt/intel/openvino/runtime/cmake

# Build (Release mode recommended)
cmake --build build --config Release
```

The build produces:
- **Static library**: `eddy` (linkable C++ library)
- **CLI tool**: `parakeet_cli.exe` (transcription utility)
- **Benchmarks**: `benchmark_librispeech.exe`, `benchmark_fleurs.exe`
- **Model fetcher**: `hf_fetch_models.exe` (manual model download)

## Model Cache

Models auto-download on first run and are cached at:

- **Windows**: `%LOCALAPPDATA%\eddy\models\`
  - V2: `%LOCALAPPDATA%\eddy\models\parakeet-v2\files\`
  - V3: `%LOCALAPPDATA%\eddy\models\parakeet-v3\files\`
- **Linux**: `~/.cache/eddy/models/`

To disable auto-download: `set EDDY_DISABLE_AUTO_FETCH=1`

Manual download: `build/examples/cpp/Release/hf_fetch_models.exe --model parakeet-v3`

## C++ API

```cpp
#include "eddy/parakeet_inference.h"

int main() {
    // Initialize Parakeet v3 with NPU
    auto asr = eddy::ParakeetASR::create("parakeet-v3", "NPU");

    // Transcribe audio file
    auto result = asr->transcribe("audio.wav");
    std::cout << "Text: " << result.text << std::endl;
    std::cout << "RTFx: " << result.rtfx << "×" << std::endl;

    return 0;
}
```

## Roadmap

- Parakeet V2/V3 OpenVINO inference ✓
- NPU/GPU/CPU multi-device support ✓
- LibriSpeech and FLEURS benchmarks ✓
- Python bindings (C API complete, wrapper in progress)
- Voice Activity Detection (VAD) preprocessing
- C# bindings for .NET applications
- Qualcomm QNN backend (Snapdragon NPU)
- AMD Ryzen AI Software backend
- Additional audio model support

## Troubleshooting

<details>
<summary><b>NPU Not Detected</b></summary>

Check for Intel Core Ultra (Meteor Lake or newer):
```bash
build/examples/cpp/Release/parakeet_cli.exe --list-devices
```

</details>

<details>
<summary><b>Slow Performance</b></summary>

- Ensure OpenVINO 2025.x is installed
- Try `--device NPU` for NPU acceleration (optimized for Intel Core Ultra)
- See the Performance section above for expected speed on each device

</details>

<details>
<summary><b>Model Configuration Issues</b></summary>

Ensure you're using the correct model configuration:
- V2: `blank_token_id = 1024`
- V3: `blank_token_id = 8192`

```bash
# Verify model version
build/examples/cpp/Release/parakeet_cli.exe --version
```

</details>

## Citation

```bibtex
@misc{eddy-2025,
  title={eddy: High-Performance ASR with OpenVINO and Parakeet TDT},
  author={FluidInference Team},
  year={2025},
  url={https://github.com/FluidInference/eddy}
}

@inproceedings{nvidia-parakeet-tdt,
  title={Parakeet-TDT: Token Duration Transducer for ASR},
  author={NVIDIA NeMo Team},
  year={2024},
  url={https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2}
}
```

## License

**Apache 2.0** - See [LICENSE](LICENSE) for details.

Third-party model licenses may vary. See [THIRDPARTY_LICENSES](THIRDPARTY_LICENSES.md) for details on Parakeet TDT models (CC-BY-4.0) and other dependencies.

## Links

- **Discord**: [Join Community](https://discord.gg/WNsvaCtmDe)
- **Documentation**: [docs/](documentation/)
- **Parakeet V2**: [HuggingFace Model Card](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-ov)
- **Parakeet V3**: [HuggingFace Model Card](https://huggingface.co/FluidInference/parakeet-tdt-1.1b-v3-ov)
- **Base Models**: [NVIDIA NeMo Parakeet TDT](https://huggingface.co/collections/nvidia/parakeet-tdt-family-6733b7a0df18b25e7689b7b0)

## Acknowledgments

- **NVIDIA NeMo Team**: Parakeet TDT architecture and base models
- **Intel OpenVINO**: Cross-platform inference runtime and NPU support
- **Benchmark Datasets**: LibriSpeech (OpenSLR), FLEURS (Google Research)
