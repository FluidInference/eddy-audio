# eddy Documentation

eddy is a high-performance, embeddable inference SDK built for native runtimes. It provides speech recognition, transcription, and audio processing capabilities with a modular backend layer — OpenVINO GenAI is the first supported target, with additional runtimes in development.

## Table of Contents

- [Getting Started](getting-started.md) - Quick start guide and installation
- [Build Instructions](build-instructions.md) - How to build from source
- [Performance Benchmarks](performance-benchmarks.md) - NPU vs CPU performance stats
- [C++ API Reference](api/cpp-api.md) - Core C++ API documentation
- [C API Reference](api/c-api.md) - C FFI layer for language bindings
- [C# Bindings](bindings/csharp-bindings.md) - .NET integration guide
- [Examples](examples.md) - Code examples and usage patterns
- [Architecture](architecture.md) - System design and architecture overview

## Quick Links

### Features
- Whisper Speech Recognition — OpenAI's Whisper large-v3-turbo model
- Intel NPU Acceleration — 6x faster inference than CPU
- Model Caching — 55x faster startup after first run
- Multi-Device Support — NPU, CPU, AUTO
- Language Bindings — C# (Rust, Kotlin, Flutter planned)
- Timestamp Support — Word/segment-level timestamps
- Multi-Language — Support for 99+ languages

### Supported Platforms
- Windows 11 (Intel Core Ultra with NPU)
- Windows 10 (CPU fallback)
- Linux (planned)

### Requirements
Current OpenVINO backend prerequisites:
- OpenVINO 2025.3.0+
- OpenVINO GenAI 2025.3.0+
- CMake 3.22+
- MSVC 2022 (Windows) or GCC 9+ (Linux)

## Getting Started

```cpp
#include <eddy/pipelines/whisper_pipeline.hpp>

eddy::WhisperConfig config;
config.model_path = "models/whisper-large-v3-turbo";
config.device = "NPU";
config.language = "en";

eddy::WhisperPipeline pipeline(config);
auto result = pipeline.transcribe("audio.wav");

std::cout << "Text: " << result.text << std::endl;
std::cout << "Confidence: " << (result.confidence * 100) << "%" << std::endl;
```

## Community and Support

- Issues: https://github.com/yourusername/eddy/issues
- Discussions: https://github.com/yourusername/eddy/discussions

## License

Apache License 2.0 - See ../LICENSE for details.

