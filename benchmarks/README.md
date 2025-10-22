# eddy LibriSpeech Benchmark

Fast benchmarking using C API (ctypes) - Python orchestrates, C++ does inference.

---

## Quick Start

```bash
# Go to benchmarks directory
cd benchmarks

# Install Python dependencies (one-time)
uv sync

# Run benchmark (automatically rebuilds C++ library)
uv run benchmark.py

# Full benchmark (2620 files)
uv run benchmark.py --max-files 2620

# Skip rebuild if you haven't changed C++ code
uv run benchmark.py --no-rebuild
```

**Note:** The benchmark script automatically rebuilds the C++ library to ensure you're testing the latest code. Use `--no-rebuild` to skip this step if you haven't modified the C++ code.

---

## How It Works

```
Workflow:
1. Python rebuilds C++ library (cmake --build)
2. Python loads eddy_c.dll via ctypes
3. Python loads dataset (HuggingFace datasets)
4. C++ performs inference (fast!)
5. Python normalizes text (Whisper)
6. Python calculates WER (jiwer)

Clean Architecture:
├─ Python: Build orchestration, dataset loading
├─ C++: Inference only (via eddy_c.dll)
├─ Python: Text normalization (Whisper)
└─ Python: WER calculation (jiwer)

No subprocess overhead - direct C API calls via ctypes!
```

**Key Benefits:**
- ✅ **Clean separation** - C++ for inference, Python for everything else
- ✅ **No subprocess overhead** - Direct C API calls
- ✅ **Industry standard** - Whisper normalization + jiwer
- ✅ **Flexible** - Easy to modify Python orchestration
- ✅ **Fast** - Models loaded once, reused for all files

---

## Usage

```bash
# Default (50 files, ~2 minutes, auto-rebuild)
uv run benchmark.py

# Full benchmark (2620 files)
uv run benchmark.py --max-files 2620

# Skip rebuild (when C++ code hasn't changed)
uv run benchmark.py --no-rebuild

# Use NPU (Neural Processing Unit)
uv run benchmark.py --max-files 100 --device NPU

# Custom library path (implies --no-rebuild)
uv run benchmark.py --lib build/Release/eddy_c.dll

# Custom output file
uv run benchmark.py --output my_results.json
```

---

## Example Output

```
Using library: build\Release\eddy_c.dll
Loading dataset: librispeech_asr/clean test.clean
Loading model on device: CPU

Running inference on 50 files...
================================================================================
[10/50] WER: 3.45%  RTFx: 5.2x  Last: 0.00%
[20/50] WER: 3.67%  RTFx: 5.3x  Last: 5.26%
[30/50] WER: 3.52%  RTFx: 5.4x  Last: 0.00%
[40/50] WER: 3.61%  RTFx: 5.3x  Last: 4.17%
[50/50] WER: 3.63%  RTFx: 5.4x  Last: 0.00%

================================================================================
BENCHMARK SUMMARY
================================================================================
Files processed:      50
Overall WER:          3.63%
Median WER:           0.00%
Overall RTFx:         5.4x
Total audio:          289.3s
Total processing:     53.6s
Benchmark elapsed:    68.2s
Results saved to:     eddy_benchmark_results.json
================================================================================
```

---

## Architecture: ctypes C API

**Why ctypes approach?**

Previous approach:
- Python spawns C++ subprocess for each operation
- Subprocess overhead ~50ms per file
- Complex JSON parsing between processes

**New approach:**
- Python loads `eddy_c.dll` directly via ctypes
- Direct function calls (no subprocess)
- C++ handles **inference only**
- Python handles dataset, normalization, WER

**Code example:**

```python
# Load library
lib = ctypes.CDLL("build/Release/eddy_c.dll")

# Create model (once)
handle = lib.eddy_parakeet_create(config)

# Process all files
for audio_file in dataset:
    result = lib.eddy_parakeet_infer_buffer(handle, audio_data)
    wer = calculate_wer(result.text, reference)

# Cleanup
lib.eddy_parakeet_destroy(handle)
```

Clean and simple! C++ does what it's good at (fast inference), Python does the rest.

---

## Dataset

LibriSpeech test-clean dataset (2620 files, ~5.4 hours audio) is automatically downloaded via HuggingFace `datasets` library on first run.

No manual download needed!

---

## Output Format

```json
{
  "config": {
    "device": "CPU",
    "model_dir": "cache",
    "dataset": "librispeech_asr/clean",
    "split": "test.clean",
    "num_files": 50
  },
  "metrics": {
    "overall_wer": 3.63,
    "median_wer": 0.0,
    "overall_rtfx": 5.4,
    "total_audio_duration_sec": 289.3,
    "total_processing_time_sec": 53.6,
    "benchmark_elapsed_sec": 68.2
  },
  "per_file_results": [
    {
      "file_id": "1089-134686-0000",
      "reference": "HE HOPED THERE WOULD BE...",
      "hypothesis": "he hoped there would be...",
      "wer": 0.0,
      "audio_duration_sec": 5.47,
      "processing_time_sec": 0.98,
      "rtfx": 5.58,
      "confidence": 0.95
    }
  ]
}
```

---

## Text Normalization

Uses **OpenAI Whisper's English normalizer** - the industry standard for ASR benchmarking.

Normalization includes:
- **Lowercase conversion** - `He hoped` → `he hoped`
- **Punctuation removal** - `dinner,` → `dinner`
- **Number standardization** - `1st` → `first`, `1,000` → `one thousand`
- **Whitespace normalization** - Multiple spaces → single space
- **Filler word removal** - Remove special tokens

Same normalization used by:
- OpenAI Whisper
- HuggingFace Open ASR Leaderboard
- FluidAudio benchmarks
- Most competitive ASR systems

---

## Dependencies

Managed via `uv` (fast Python package manager):

```bash
cd benchmarks
uv sync  # Installs everything from pyproject.toml
```

**Required packages:**
- `datasets` - HuggingFace datasets (for LibriSpeech)
- `jiwer` - Word Error Rate calculation
- `whisper-normalizer` - OpenAI Whisper text normalization
- `numpy` - Array handling for audio data

All lightweight, standard libraries!

---

## Performance

**Benchmark results (Intel CPU):**

| Files | WER | RTFx | Time |
|-------|-----|------|------|
| 50 | 3.63% | 5.4x | ~70s |
| 100 | 3.65% | 5.3x | ~2.5min |
| 2620 | ~3.5% | ~5.0x | ~2 hours |

**RTFx = Real-Time Factor** (how many times faster than real-time)
- 5.4x means 1 second of audio processed in 0.185 seconds

---

## Troubleshooting

**Library not found:**
```bash
# Build eddy_c library first
cmake --build build --config Release --target eddy_c

# Or specify path manually
uv run benchmark.py --lib path/to/eddy_c.dll
```

**Missing dependencies:**
```bash
cd benchmarks
uv sync
```

**Dataset download fails:**
- Check internet connection
- Ensure ~2GB free disk space
- The `datasets` library handles download automatically

**CUDA/NPU errors:**
- Make sure OpenVINO is set up for your device
- Try `--device CPU` as fallback

---

## Files

- **`benchmark.py`** - Main benchmark script (ctypes approach)
- **`pyproject.toml`** - Python dependencies (uv format)
- **`uv.lock`** - Locked dependency versions
- **`README.md`** - This file

---

## Why This Approach?

**Separation of concerns:**
- 🎯 C++ does **inference** (what it's good at)
- 🐍 Python does **orchestration** (what it's good at)

**Benefits:**
- ✅ No subprocess overhead
- ✅ Direct C API calls (fast)
- ✅ Easy to extend in Python
- ✅ Industry-standard text normalization
- ✅ Models loaded once (efficient)
- ✅ Clean, maintainable code

**Comparison to previous approach:**

| Aspect | Old (subprocess) | New (ctypes) |
|--------|-----------------|--------------|
| Architecture | C++ standalone | Python + C API |
| Dataset loading | C++ curl + tar | Python datasets library |
| Model loading | Per-subprocess | Once per benchmark |
| Overhead | ~50ms per file | None (direct calls) |
| Text norm | Basic C++ | Whisper (industry standard) |
| Flexibility | Limited | High (Python) |

The ctypes approach is **cleaner, faster, and more maintainable**! 🎯
