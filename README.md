# eddy

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![GitHub Stars](https://img.shields.io/github/stars/FluidInference/eddy?style=flat&logo=github)](https://github.com/FluidInference/eddy)

**C++ inference library for multi-vendor edge NPUs.** Current focus: OpenVINO 2025.x backend for Parakeet-TDT and Whisper models. Additional runtimes (Qualcomm QNN, AMD Ryzen AI Software) coming soon.

For Apple platforms (macOS/iOS), use [FluidAudio](https://github.com/FluidInference/FluidAudio).

**Model Cards:**
- [Parakeet V2 (English)](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-ov)
- [Parakeet V3 (Multilingual)](https://huggingface.co/FluidInference/parakeet-tdt-1.1b-v3-ov)
- [Whisper large-v3-turbo](https://huggingface.co/FluidInference/whisper-large-v3-turbo-fp16-ov-npu)

## Building

```bash
# Configure with vcpkg toolchain
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake

# Or specify OpenVINO manually if not using vcpkg
cmake -S . -B build -DOpenVINO_DIR=/opt/intel/openvino/runtime/cmake

# Build (Release mode recommended)
cmake --build build --config Release
```

The build produces:
- **Static library**: `eddy` (linkable C++ library)
- **CLI tools**: `parakeet_cli.exe`, `whisper_example.exe` (examples)
- **Benchmarks**: `benchmark_librispeech.exe`, `benchmark_fleurs.exe`

### Optional: Whisper Support

Whisper requires OpenVINO GenAI (not included by default):

```bash
cmake -S . -B build -DEDDY_ENABLE_WHISPER=ON -DOpenVINOGenAI_DIR="<path-to-genai-cmake>"
```

## Usage

### Basic Transcription

**Parakeet V2** (English only):
```bash
build/examples/cpp/Release/parakeet_cli.exe audio.wav --model parakeet-v2 --device NPU
```

**Parakeet V3** (Multilingual - 24 languages):
```bash
# English (default)
build/examples/cpp/Release/parakeet_cli.exe audio.wav --model parakeet-v3 --device NPU

# Spanish
build/examples/cpp/Release/parakeet_cli.exe audio_es.wav --model parakeet-v3 --language es --device NPU

# French
build/examples/cpp/Release/parakeet_cli.exe audio_fr.wav --model parakeet-v3 --language fr --device NPU
```

**Whisper** (if built with `EDDY_ENABLE_WHISPER=ON`):
```bash
build/examples/cpp/Release/whisper_example.exe path/to/whisper-model audio.wav NPU
```

### Device Selection

```bash
# NPU (best performance on Intel Core Ultra)
--device NPU

# CPU (fallback)
--device CPU
```

Models auto-download from HuggingFace on first run. See [C++ API documentation](docs/CPP_API.md) for library integration.

## Models & Performance

Benchmarked on Intel Core Ultra 7 155H (Meteor Lake) with Intel AI Boost NPU. RTFx values are averaged across LibriSpeech test-clean dataset.

| Model | Languages | NPU Speed (avg) | CPU Speed (avg) | Size |
|-------|-----------|-----------------|-----------------|------|
| **Parakeet V2** | English | **38× RTFx** | 8× RTFx | 600MB |
| **Parakeet V3** | 24 languages | **41× RTFx** | 8× RTFx | 1.1GB |
| **Whisper large-v3-turbo** | 99 languages | **16× RTFx** | 0.44× RTFx | 1.6GB |

> **RTFx** = Real-Time Factor (higher is faster). 38× means processing is 38× faster than real-time playback - 10 minutes of audio transcribed in ~16 seconds.

### Performance Comparison: eddy (OpenVINO) vs PyTorch

Benchmarked on Intel Core Ultra 7 155H (Meteor Lake):

| Model | eddy NPU | eddy CPU | PyTorch GPU (Arc 140V) | PyTorch CPU | eddy NPU Speedup (vs PyTorch GPU) |
|-------|----------|----------|------------------------|-------------|-----------------------------------|
| **Parakeet V2** | 38× RTFx | 8× RTFx | 8.4× RTFx¹ | 2× RTFx² | **4.5× faster** |
| **Parakeet V3** | 41× RTFx | 8× RTFx | 8.4× RTFx¹ | 2.5× RTFx² | **4.9× faster** |
| **Whisper large-v3-turbo** | 16× RTFx | 0.44× RTFx | 5.5× RTFx | 0.90× RTFx | **2.9× faster** |

¹ Benchmarked using NeMo parakeet-tdt_ctc-110m as proxy (similar architecture)
² Estimated based on NeMo reference implementations

*eddy's NPU implementation provides 3-5× acceleration over PyTorch GPU and 8-20× over PyTorch CPU on Intel Core Ultra 7 155H.*

**Parakeet V3 Languages**: English, Spanish, Italian, French, German, Dutch, Russian, Polish, Ukrainian, Slovak, Bulgarian, Finnish, Romanian, Croatian, Czech, Swedish, Estonian, Hungarian, Lithuanian, Danish, Maltese, Slovenian, Latvian, Greek

**Benchmarks**: See [BENCHMARK.md](BENCHMARK.md) for detailed results and instructions.

## Roadmap

- Voice Activity Detection (VAD)
- C# bindings for .NET applications
- Qualcomm QNN backend (Snapdragon NPU)
- AMD Ryzen AI Software backend
- Additional audio model support

## Support & Resources

### Troubleshooting

#### NPU Not Detected

**Windows:**
Check for Intel Core Ultra (Meteor Lake or newer):
```bash
build/examples/cpp/Release/parakeet_cli.exe --list-devices
```

**Linux:**
NPU support requires the Intel NPU driver.

> **Note:** Linux NPU support has not been tested yet.

**Requirements:**
- Ubuntu 22.04+ with kernel 6.6+
- Intel Core Ultra (Meteor Lake) or newer processor

For installation instructions, see the official Intel NPU driver documentation:
- **Installation Guide**: [github.com/intel/linux-npu-driver](https://github.com/intel/linux-npu-driver)
- **Latest Releases**: [github.com/intel/linux-npu-driver/releases](https://github.com/intel/linux-npu-driver/releases)

#### Slow Performance

- Ensure OpenVINO 2025.x is installed
- Try `--device NPU` for NPU acceleration (optimized for Intel Core Ultra)
- See the Performance section above for expected speed on each device

#### Model Configuration Issues

Ensure you're using the correct model configuration:
- V2: `blank_token_id = 1024`
- V3: `blank_token_id = 8192`

```bash
# Verify model version
build/examples/cpp/Release/parakeet_cli.exe --version
```

### Citation

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

### License

**Apache 2.0** - See [LICENSE](LICENSE) for details.

Third-party model licenses may vary. See [ThirdPartyLicenses/](ThirdPartyLicenses/) for details on Parakeet TDT models (CC-BY-4.0) and other dependencies.

### Acknowledgments

- **NVIDIA NeMo Team**: Parakeet TDT architecture and base models
- **Intel OpenVINO**: Cross-platform inference runtime and NPU support
- **Benchmark Datasets**: LibriSpeech (OpenSLR), FLEURS (Google Research)
