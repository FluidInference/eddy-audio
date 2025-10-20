# eddy LibriSpeech Benchmark

Fast benchmarking using C++ for inference + Python for WER calculation with Whisper normalization.

---

## Quick Start

```bash
# From project root, build C++ benchmark (one-time)
cmake --build build --config Release --target benchmark_librispeech

# Go to benchmarks directory
cd benchmarks

# Install Python dependencies
uv pip install whisper-normalizer jiwer

# Run benchmark
python benchmark.py --max-files 25
```

---

## How It Works

```
Step 1: C++ Benchmark
├─ Load models ONCE
├─ Process all files (fast)
└─ Output: transcriptions + raw WER

Step 2: Python WER Calculation
├─ Read C++ transcriptions
├─ Normalize text (Whisper normalizer)
├─ Recalculate WER with normalization
└─ Output: final results + comparison

Total: ~49 minutes for 2620 files (vs 2 hours with subprocess approach)
```

**Key Benefits:**
- ✅ **Fast** - Models loaded once (2-3x faster)
- ✅ **Accurate** - Whisper normalization (industry standard)
- ✅ **Comparison** - Shows impact of normalization
- ✅ **Best of both worlds** - C++ speed + Python text processing

---

## Usage

```bash
# Default (25 files, ~1 minute)
python benchmark.py

# Full benchmark (2620 files, ~49 minutes)
python benchmark.py --max-files 2620

# Use GPU
python benchmark.py --max-files 2620 --device GPU

# Use NPU (Neural Processing Unit)
python benchmark.py --max-files 2620 --device NPU

# Custom output file
python benchmark.py --max-files 100 --output my_results.json
```

---

## Example Output

```
================================================================================
eddy Fast Benchmark (C++ + Python)
================================================================================

Step 1: C++ inference (fast, models loaded once)
Step 2: Python WER calculation (Whisper normalization)

================================================================================
STEP 1: Running C++ benchmark...
================================================================================

[Processing 100 files...]

C++ Results (no text normalization):
  Average WER: 4.17%
  Median WER:  0.00%
  Overall RTFx: 2.4x

================================================================================
STEP 2: Calculating WER with Python normalization...
================================================================================

Python Results (with Whisper normalization):
  Average WER: 3.63%
  Median WER:  0.00%

================================================================================
COMPARISON: Impact of Text Normalization
================================================================================
C++ WER (no normalization):      4.17%
Python WER (Whisper normalization): 3.63%
Improvement:                      0.54% (13.0% better)

Text normalization (OpenAI Whisper standard):
  - Lowercase conversion
  - Punctuation removal
  - Number standardization
  - Whitespace normalization
  - Remove filler words and special tokens

================================================================================
Final results saved to: eddy_benchmark_results.json
================================================================================
```

---

## Dataset Location

LibriSpeech test-clean (~350 MB) auto-downloads to cache:

- **Windows:** `%LOCALAPPDATA%\eddy\datasets\LibriSpeech\test-clean\`
- **Linux/Mac:** `~/.cache/eddy/datasets/LibriSpeech\test-clean\`

---

## Manual Workflow (Advanced)

The benchmark script handles both C++ inference and Python WER calculation automatically.

If you need to run the C++ benchmark separately:

```bash
# From benchmarks/ directory
../build/examples/cpp/Release/benchmark_librispeech.exe \
    --max-files 2620 \
    --device CPU

# This outputs: eddy_benchmark_results_cpp.json
```

Then use the Python script's WER recalculation (already integrated in benchmark.py).

---

## Output Format

```json
{
  "summary": {
    "files_processed": 100,
    "average_wer_percent": 4.17,    // C++ (no normalization)
    "avg_wer": 3.63,                // Python (with normalization)
    "median_wer": 0.0,
    "overall_rtfx": 2.4,
    "device": "CPU"
  },
  "results": [
    {
      "file_id": "1089-134686-0000",
      "reference": "HE HOPED THERE WOULD BE...",
      "hypothesis": "He hoped there would be...",
      "wer": 0.0,
      "substitutions": 0,
      "deletions": 0,
      "insertions": 0,
      "hits": 28,
      "reference_normalized": "he hoped there would be...",
      "hypothesis_normalized": "he hoped there would be..."
    }
  ]
}
```

---

## Text Normalization

OpenAI Whisper normalization (industry standard):

- **Lowercase conversion** - `He hoped` → `he hoped`
- **Punctuation removal** - `dinner,` → `dinner`
- **Number standardization** - `1st` → `first`, `1,000` → `one thousand`
- **Whitespace normalization** - Multiple spaces → single space
- **Filler word removal** - Remove special tokens and common fillers

This ensures **comparable results** to OpenAI Whisper, FluidAudio, and the HuggingFace Open ASR Leaderboard.

---

## Performance Comparison

| Metric | Value | Notes |
|--------|-------|-------|
| Speed (2620 files) | ~49 minutes | Models loaded once |
| Speed improvement | 2-3x faster | vs subprocess approach |
| WER improvement | ~13% better | With normalization |
| Devices | CPU/GPU/NPU/AUTO | OpenVINO backends |

**Example (100 files):**
- C++ WER: 4.17%
- Python WER: 3.63% (13% improvement)
- Processing time: ~6 minutes
- RTFx: 2.4x on CPU

---

## Dependencies

```bash
# Two lightweight packages needed
uv pip install whisper-normalizer jiwer
```

- **whisper-normalizer**: OpenAI Whisper's text normalization (industry standard)
- **jiwer**: Word Error Rate calculation

No longer need:
- ❌ tqdm (progress bars) - C++ shows progress
- ❌ soundfile (audio processing) - C++ handles this
- ❌ datasets (HuggingFace) - direct download instead

---

## Troubleshooting

**C++ benchmark not found:**
```bash
cmake --build build --config Release --target benchmark_librispeech
```

**Missing dependencies:**
```bash
uv pip install whisper-normalizer jiwer
```

**Dataset not downloading:**
- Check internet connection
- Verify disk space (~350 MB needed)
- Dataset auto-downloads on first run

**JSON parsing errors:**
- The script now auto-fixes Windows path escaping issues
- If problems persist, check `eddy_benchmark_results_cpp.json` format

---

## Files

- **`benchmark.py`** - Main benchmark script (C++ inference + Python WER)
- **`pyproject.toml`** - Python dependencies (whisper-normalizer + jiwer)
- **`README.md`** - This file

---

## Why This Approach?

**C++ for inference:**
- ⚡ Fast (models loaded once)
- 🎯 Efficient (no subprocess overhead)
- 📊 Native performance

**Python for WER:**
- 📚 Rich ecosystem (Whisper normalizer + jiwer)
- ✅ Industry standard normalization (OpenAI Whisper)
- 🔄 Easy to modify and experiment

**Best of both worlds!** 🎯
