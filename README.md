# eddy

**eddy** is a high-performance C++ inference library for automatic speech recognition (ASR), optimized for Intel NPUs, GPUs, and CPUs on Windows and Linux. Built on OpenVINO 2025.x, eddy powers private, offline transcription with the Parakeet TDT model family.

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![GitHub Stars](https://img.shields.io/github/stars/FluidInference/eddy?style=flat&logo=github)](https://github.com/FluidInference/eddy)

## Highlights

- **🚀 Fast**: 40-45× real-time on Intel NPU (10 min audio in ~15 seconds)
- **🔒 Private**: Fully on-device inference, no network calls after model download
- **🌍 Multilingual**: Parakeet V3 supports 24 European languages
- **🎯 Accurate**: V2 achieves 2.76% WER on LibriSpeech test-clean (English)
- **💻 Cross-platform**: Windows 10/11, Linux (Ubuntu 20.04+)
- **⚡ NPU Accelerated**: Optimized for Intel Core Ultra (Meteor Lake, Lunar Lake)

## Platform Support

- **Supported**: Windows and Linux
- **Not supported**: Apple platforms (macOS/iOS) - use [FluidAudio](https://github.com/FluidInference/FluidAudio) instead

## Quick Start

### Installation

```bash
git clone https://github.com/FluidInference/eddy.git
cd eddy

# Build with vcpkg (handles dependencies automatically)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

### Usage

```bash
# Transcribe with Parakeet V2 (English, best accuracy)
build/examples/cpp/Release/parakeet_cli.exe audio.wav --model parakeet-v2

# Use NPU for 6-10× speedup
build/examples/cpp/Release/parakeet_cli.exe audio.wav --model parakeet-v2 --device NPU

# Transcribe with Parakeet V3 (24 languages)
build/examples/cpp/Release/parakeet_cli.exe audio.wav --model parakeet-v3 --device NPU

# Benchmark on LibriSpeech test-clean (English)
build/examples/cpp/Release/benchmark_librispeech.exe --max-files 100 --device NPU

# Benchmark on FLEURS (multilingual, 24 languages)
build/examples/cpp/Release/benchmark_fleurs.exe "%LOCALAPPDATA%\eddy\datasets\FLEURS" --device NPU
```

Models auto-download on first run from HuggingFace.

## Models

| Model | Languages | WER | Speed (NPU) | Size | HuggingFace |
|-------|-----------|-----|-------------|------|-------------|
| **Parakeet V2** | English only | 2.76% | 41× RTFx | 600MB | [v2-ov](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-ov) |
| **Parakeet V3** | 24 European | 6.09% EN<br>16.98% avg | 41× RTFx | 1.1GB | [v3-ov](https://huggingface.co/FluidInference/parakeet-tdt-1.1b-v3-ov) |

**V3 Languages**: 🇬🇧 English • 🇪🇸 Spanish • 🇮🇹 Italian • 🇫🇷 French • 🇩🇪 German • 🇳🇱 Dutch • 🇷🇺 Russian • 🇵🇱 Polish • 🇺🇦 Ukrainian • 🇸🇰 Slovak • 🇧🇬 Bulgarian • 🇫🇮 Finnish • 🇷🇴 Romanian • 🇭🇷 Croatian • 🇨🇿 Czech • 🇸🇪 Swedish • 🇪🇪 Estonian • 🇭🇺 Hungarian • 🇱🇹 Lithuanian • 🇩🇰 Danish • 🇲🇹 Maltese • 🇸🇮 Slovenian • 🇱🇻 Latvian • 🇬🇷 Greek

## Performance

### Device Comparison

| Device | RTFx | Power | Best For |
|--------|------|-------|----------|
| **Intel NPU** | 40-45× | Lowest | Laptops (Core Ultra) |
| **GPU** | 15-25× | Medium | Desktops with discrete GPU |
| **CPU** | 5-8× | Higher | Compatibility |

> **RTFx** = Real-Time Factor. 41× means 10 minutes of audio transcribed in ~15 seconds.

### Benchmarks

**Parakeet V2 - LibriSpeech test-clean (English)**
- WER: 2.76% | CER: 1.13% | RTFx: 41.36× (Intel NPU)
- Tested on: 100 samples, Intel Core Ultra 7 258V

**Parakeet V3 - FLEURS (24 Languages)**
- Average WER: 16.98% | Average CER: 5.39% | RTFx: 41.1×
- Best: Italian (4.30%), Spanish (5.44%), English (6.09%)

See [FLEURS_BENCHMARK.md](FLEURS_BENCHMARK.md) for detailed multilingual results.

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

## Repository Layout

- [include/](include/) - Public C++ API headers
- [src/](src/) - Backend implementations and model inference code
- [examples/cpp/](examples/cpp/) - CLI tools and benchmarks
- [documentation/](documentation/) - Design docs and guides
- [FLEURS_BENCHMARK.md](FLEURS_BENCHMARK.md) - Multilingual benchmark details
- [PARAKEET_V2_MODEL_CARD.md](PARAKEET_V2_MODEL_CARD.md) - V2 model documentation
- [README_V3_HUGGINGFACE.md](README_V3_HUGGINGFACE.md) - V3 model documentation

## Dependencies

### Required
- **OpenVINO** (2025.x) - AI inference runtime
- **libsndfile** - Audio file I/O (WAV, FLAC, OGG)
- **libsamplerate** - High-quality audio resampling

### Optional
- **OpenVINO GenAI** - For Whisper model support (enabled by default, disable with `-DEDDY_ENABLE_WHISPER=OFF`)

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

## Python API (Coming Soon)

```python
from eddy import ParakeetASR

# Initialize with NPU acceleration
asr = ParakeetASR(model="parakeet-v3", device="NPU")

# Transcribe audio file
result = asr.transcribe("audio.wav")
print(f"Text: {result['text']}")
print(f"WER: {result['wer']:.2f}%, RTFx: {result['rtfx']:.1f}×")
```

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

## Hardware Requirements

### Minimum
- **CPU**: Intel Core (Gen 10+), AMD Ryzen
- **RAM**: 8GB
- **Storage**: 2GB for models

### Recommended (NPU Acceleration)
- **CPU**: Intel Core Ultra (Meteor Lake, Lunar Lake)
- **NPU**: Intel AI Boost (4th gen, 48 TOPS)
- **RAM**: 16GB
- **OS**: Windows 11 or Linux with NPU drivers

## Roadmap

- ✅ Parakeet V2/V3 OpenVINO inference
- ✅ NPU/GPU/CPU multi-device support
- ✅ LibriSpeech and FLEURS benchmarks
- ⏳ Streaming inference (buffered real-time transcription)
- ⏳ Python bindings (C API complete, wrapper in progress)
- ⏳ Voice Activity Detection (VAD) preprocessing
- 🔮 C# bindings for .NET applications
- 🔮 Qualcomm QNN backend (Snapdragon NPU)
- 🔮 AMD MIGraphX backend (Ryzen AI)

## Troubleshooting

### High WER (>90%)

Ensure you're using the correct model:
- V2: `blank_token_id = 1024` (English only)
- V3: `blank_token_id = 8192` (multilingual)

```bash
# Verify model version
build/examples/cpp/Release/parakeet_cli.exe --version
```

### NPU Not Detected

Check for Intel Core Ultra (Meteor Lake or newer):
```bash
build/examples/cpp/Release/parakeet_cli.exe --list-devices
```

### Slow Performance

- Use `--device NPU` for 6-10× speedup over CPU
- Ensure OpenVINO 2025.x is installed
- On CPU, 5-8× RTFx is expected

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

**CC-BY-4.0** - See [LICENSE](LICENSE) for details.

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
