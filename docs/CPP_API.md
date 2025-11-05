# C++ API Documentation

This document describes how to integrate eddy as a library in your C++ applications.

## Basic Usage

### Parakeet V2 (English Only)

```cpp
#include "eddy/parakeet_inference.h"

int main() {
    // Initialize Parakeet V2 with NPU
    auto asr = eddy::ParakeetASR::create("parakeet-v2", "NPU");

    // Transcribe English audio file
    auto result = asr->transcribe("audio.wav");

    // Access results
    std::cout << "Text: " << result.text << std::endl;
    std::cout << "RTFx: " << result.rtfx << "×" << std::endl;

    return 0;
}
```

### Parakeet V3 (Multilingual)

```cpp
#include "eddy/parakeet_inference.h"

int main() {
    // Initialize Parakeet V3 with NPU
    auto asr = eddy::ParakeetASR::create("parakeet-v3", "NPU");

    // Transcribe English audio (default)
    auto result_en = asr->transcribe("audio_en.wav");
    std::cout << "English: " << result_en.text << std::endl;

    // Transcribe Spanish audio
    auto result_es = asr->transcribe("audio_es.wav", "es");
    std::cout << "Spanish: " << result_es.text << std::endl;

    // Transcribe French audio
    auto result_fr = asr->transcribe("audio_fr.wav", "fr");
    std::cout << "French: " << result_fr.text << std::endl;

    return 0;
}
```

**Supported Languages (24):** English (default), Spanish, Italian, French, German, Dutch, Russian, Polish, Ukrainian, Slovak, Bulgarian, Finnish, Romanian, Croatian, Czech, Swedish, Estonian, Hungarian, Lithuanian, Danish, Maltese, Slovenian, Latvian, Greek

**Language Codes:** `en`, `es`, `it`, `fr`, `de`, `nl`, `ru`, `pl`, `uk`, `sk`, `bg`, `fi`, `ro`, `hr`, `cs`, `sv`, `et`, `hu`, `lt`, `da`, `mt`, `sl`, `lv`, `el`

### Whisper ASR

```cpp
#include "eddy/whisper_inference.h"

int main() {
    // Initialize Whisper with model path and device
    auto asr = eddy::WhisperASR::create(
        "path/to/whisper-model",
        "NPU"
    );

    // Transcribe audio file
    auto result = asr->transcribe("audio.wav");

    std::cout << "Text: " << result.text << std::endl;

    return 0;
}
```

## API Reference

### ParakeetASR Class

#### Static Methods

##### `create(model_name, device)`
Creates a new Parakeet ASR instance.

**Parameters:**
- `model_name` (string): Model identifier - `"parakeet-v2"` or `"parakeet-v3"`
- `device` (string): Target device - `"NPU"`, `"CPU"`, or `"GPU"`

**Returns:** `std::unique_ptr<ParakeetASR>`

#### Instance Methods

##### `transcribe(audio_path, language = "en")`
Transcribes an audio file.

**Parameters:**
- `audio_path` (string): Path to audio file (WAV, FLAC, OGG, etc.)
- `language` (string, optional): Language code for Parakeet V3 (default: `"en"`)
  - V2: Only supports English (parameter ignored)
  - V3: Supports 24 languages - see language codes above

**Returns:** `TranscriptionResult`

**TranscriptionResult fields:**
- `text` (string): Transcribed text
- `rtfx` (double): Real-time factor (speed metric)
- `wer` (double): Word error rate (if reference available)
- `tokens` (vector): Individual token information
- `confidence` (double): Overall confidence score

**Example:**
```cpp
// V2: English only
auto result = asr->transcribe("audio.wav");

// V3: Specify language
auto result_es = asr->transcribe("audio.wav", "es");  // Spanish
auto result_fr = asr->transcribe("audio.wav", "fr");  // French
```

### WhisperASR Class

#### Static Methods

##### `create(model_path, device)`
Creates a new Whisper ASR instance.

**Parameters:**
- `model_path` (string): Path to OpenVINO Whisper model directory
- `device` (string): Target device - `"NPU"`, `"CPU"`, or `"GPU"`

**Returns:** `std::unique_ptr<WhisperASR>`

#### Instance Methods

##### `transcribe(audio_path)`
Transcribes an audio file.

**Parameters:**
- `audio_path` (string): Path to audio file

**Returns:** `TranscriptionResult`

## Building and Linking

### CMakeLists.txt Example

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyApp)

# Find eddy package
find_package(eddy REQUIRED)

# Create your executable
add_executable(my_app main.cpp)

# Link against eddy
target_link_libraries(my_app PRIVATE eddy::eddy)
```

### Compiler Requirements

- **C++17** or later
- **CMake 3.20+**
- **OpenVINO 2025.0+** installed

### Supported Platforms

- **Windows**: MSVC 2019+, MinGW-w64
- **Linux**: GCC 9+, Clang 10+

## Advanced Usage

### Custom Model Cache Directory

Set environment variable before running:

```bash
# Windows
set EDDY_CACHE_DIR=C:\path\to\cache

# Linux
export EDDY_CACHE_DIR=/path/to/cache
```

### Disable Auto-Download

```bash
# Windows
set EDDY_DISABLE_AUTO_FETCH=1

# Linux
export EDDY_DISABLE_AUTO_FETCH=1
```

### List Available Devices

```cpp
#include "eddy/device_utils.h"

int main() {
    auto devices = eddy::list_available_devices();
    for (const auto& device : devices) {
        std::cout << "Device: " << device << std::endl;
    }
    return 0;
}
```

Or via CLI:

```bash
build/examples/cpp/Release/parakeet_cli.exe --list-devices
```

## Error Handling

All methods may throw exceptions on error:

```cpp
#include "eddy/parakeet_inference.h"
#include <exception>

int main() {
    try {
        auto asr = eddy::ParakeetASR::create("parakeet-v3", "NPU");
        auto result = asr->transcribe("audio.wav");
        std::cout << result.text << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

## Performance Tips

1. **Use Release builds** - Debug builds are significantly slower
2. **NPU for best performance** - On Intel Core Ultra processors
3. **Batch processing** - Initialize once, transcribe multiple files
4. **Model caching** - Models auto-download and cache on first use

## Examples

See the [examples/cpp](../examples/cpp) directory for complete working examples:

- `parakeet_cli.cpp` - Command-line transcription tool
- `whisper_example.cpp` - Whisper integration example
- `benchmark_librispeech.cpp` - Benchmark on LibriSpeech dataset
- `benchmark_fleurs.cpp` - Multilingual benchmark on FLEURS

## Support

For issues or questions:
- **GitHub Issues**: [github.com/FluidInference/eddy/issues](https://github.com/FluidInference/eddy/issues)
- **Discord**: [discord.gg/WNsvaCtmDe](https://discord.gg/WNsvaCtmDe)
