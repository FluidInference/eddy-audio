# Getting Started with eddy

This guide will help you get up and running with eddy for speech recognition.
It focuses on the OpenVINO backend while additional native runtime guides are prepared.

## Prerequisites

### Hardware Requirements
- Intel Core Ultra processor with NPU (recommended)
- Alternatively: Any x86_64 CPU (will use CPU inference)
- 8GB+ RAM
- 2GB+ free disk space for models

### Software Requirements
- Windows 11 (or Windows 10 with NPU driver support)
- Python 3.9+ (for OpenVINO installation)
- CMake 3.22 or later
- Visual Studio 2022 with C++ Desktop Development workload

## Installation

### Step 1: Install OpenVINO

```bash
pip install openvino==2025.3.0.dev20250709
pip install openvino-genai==2025.3.0.0.dev20250709
```

Verify installation:
```bash
python -c "import openvino; print(openvino.__version__)"
```

### Step 2: Clone and Build OpenVINO GenAI

```bash
cd ..
git clone https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai

cmake -S . -B build \
  -DOpenVINO_DIR="C:/Users/YOUR_USERNAME/AppData/Roaming/Python/Python313/site-packages/openvino/cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build build --config Release -j 4
```

**Note**: First build takes 10-15 minutes.

### Step 3: Download Whisper Model

```bash
cd eddy
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='FluidInference/whisper-large-v3-turbo-int8-ov-npu',
    local_dir='models/whisper-large-v3-turbo'
)
"
```

Model size: ~700MB

### Step 4: Build eddy

```bash
cmake -S . -B build \
  -DOpenVINO_DIR="C:/Users/YOUR_USERNAME/AppData/Roaming/Python/Python313/site-packages/openvino/cmake" \
  -DOpenVINOGenAI_DIR="../openvino.genai/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEDDY_BUILD_EXAMPLES=ON \
  -DBUILD_TESTING=OFF

cmake --build build --config Release -j 4
```

### Step 5: Copy Runtime Dependencies

```bash
cp ../openvino.genai/build/openvino_genai/*.dll build/examples/cpp/Release/
cp C:/Users/YOUR_USERNAME/AppData/Roaming/Python/Python313/site-packages/openvino/libs/*.dll build/examples/cpp/Release/
```

## Quick Test

### Create Test Audio

```bash
ffmpeg -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 -ac 1 test_audio/test.wav
```

Or use Windows TTS:
```powershell
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('test_audio/speech.wav')
$synth.Speak('Hello world, this is a test.')
$synth.Dispose()
```

Convert to 16kHz:
```bash
ffmpeg -i test_audio/speech.wav -ar 16000 -ac 1 test_audio/speech_16khz.wav
```

### Run Example

```bash
build/examples/cpp/Release/whisper_example.exe models/whisper-large-v3-turbo test_audio/speech_16khz.wav NPU en
```

**First run**: Expect 2-5 minutes for NPU compilation (one-time)
**Subsequent runs**: ~3 seconds load + <1 second inference

## Expected Output

```
=== eddy Whisper Example ===
Model: models/whisper-large-v3-turbo
Audio: test_audio/speech_16khz.wav
Device: NPU
Language: en

Creating Whisper pipeline...
[eddy] Model loaded in 2855 ms
[eddy] Whisper pipeline ready!

Transcribing audio...
[eddy] Transcription complete in 609 ms

=== Transcription Result ===
Text: Hello world, this is a test.
Confidence: 100.00%
Inference Time: 609.00 ms
```

## Next Steps

- [Build Instructions](build-instructions.md) - Detailed build configuration
- [C++ API Reference](api/cpp-api.md) - Learn the C++ API
- [Performance Benchmarks](performance-benchmarks.md) - See detailed performance stats
- [Examples](examples.md) - More code examples

## Troubleshooting

### "Error loading DLL"
Copy all required DLLs to the executable directory:
```bash
cp ../openvino.genai/build/openvino_genai/*.dll build/examples/cpp/Release/
cp C:/Users/YOUR_USERNAME/AppData/Roaming/Python/Python313/site-packages/openvino/libs/*.dll build/examples/cpp/Release/
```

### "NPU device not found"
Try CPU fallback:
```bash
build/examples/cpp/Release/whisper_example.exe models/whisper-large-v3-turbo test_audio/speech_16khz.wav CPU en
```

### "Model compilation taking too long"
First NPU compilation takes 2-5 minutes. Ensure:
- Cache directory is writable (`./cache`)
- Sufficient disk space (2GB+)
- Not running other NPU workloads

### "WAV file must be 16kHz"
All audio must be 16kHz mono or stereo. Convert with:
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
```
