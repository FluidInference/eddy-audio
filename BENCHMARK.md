# Benchmark Guide

This guide explains how to run benchmarks to evaluate model performance on your hardware.

## Quick Start

### Install Dependencies

```bash
pip install openvino-genai soundfile numpy
```

Or using uv:

```bash
uv pip install openvino-genai soundfile numpy
```

### Run Benchmarks

#### All Models Comparison

Compare Parakeet V2, V3, and Whisper on your hardware:

```bash
uv run python benchmarks/benchmark_whisper_ov.py
```

#### FLEURS Multilingual Benchmark

Test on specific languages with the FLEURS dataset:

```bash
# English only, 10 samples, NPU device
uv run python benchmarks/benchmark_fleurs.py --languages en_us --samples 10 --device NPU

# Multiple languages, 25 samples each
uv run python benchmarks/benchmark_fleurs.py --languages en_us es_419 fr_fr --samples 25 --device CPU

# All available languages
uv run python benchmarks/benchmark_fleurs.py --all-languages --samples 5 --device NPU
```

**FLEURS Options:**
- `--languages`: Specific language codes (e.g., `en_us`, `es_419`, `fr_fr`)
- `--all-languages`: Test all 24 supported languages
- `--samples`: Number of audio samples per language (default: 10)
- `--device`: Target device - `NPU`, `CPU`, or `GPU`

#### LibriSpeech Benchmark (C++)

For detailed accuracy testing on LibriSpeech test-clean:

```bash
# Build the benchmark
cmake --build build --config Release --target benchmark_librispeech

# Run on 25 files
build/examples/cpp/Release/benchmark_librispeech.exe --max-files 25

# Run on all files (2620 total)
build/examples/cpp/Release/benchmark_librispeech.exe
```

## Benchmark Metrics

### RTFx (Real-Time Factor)

Measures processing speed relative to audio duration:
- **RTFx = 1.0**: Processes at real-time speed (1 min audio = 1 min processing)
- **RTFx > 1.0**: Faster than real-time (RTFx = 10 means 1 min audio in 6 seconds)
- **RTFx < 1.0**: Slower than real-time

### WER (Word Error Rate)

Measures transcription accuracy:
- **Lower is better**
- Calculated as: `(Substitutions + Deletions + Insertions) / Total Words × 100`
- Industry standard metric for ASR evaluation

### Confidence Score

Per-token confidence from the model:
- **Range**: 0.0 to 1.0 (higher is better)
- Useful for filtering uncertain predictions

## Benchmark Results

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed performance data on Intel Core Ultra 7 155H.

## Dataset Information

### LibriSpeech

- **Source**: [OpenSLR](http://www.openslr.org/12)
- **License**: CC-BY-4.0
- **Language**: English only
- **Test-clean subset**: 2,620 samples, ~5.4 hours
- **Use case**: High-quality English ASR evaluation

### FLEURS

- **Source**: [Google Research](https://huggingface.co/datasets/google/fleurs)
- **License**: CC-BY-4.0
- **Languages**: 102 languages (eddy supports 24)
- **Use case**: Multilingual ASR evaluation

## Supported Languages (Parakeet V3)

English, Spanish, Italian, French, German, Dutch, Russian, Polish, Ukrainian, Slovak, Bulgarian, Finnish, Romanian, Croatian, Czech, Swedish, Estonian, Hungarian, Lithuanian, Danish, Maltese, Slovenian, Latvian, Greek

**Language Codes for FLEURS:**
- `en_us` - English
- `es_419` - Spanish
- `it_it` - Italian
- `fr_fr` - French
- `de_de` - German
- `nl_nl` - Dutch
- `ru_ru` - Russian
- `pl_pl` - Polish
- `uk_ua` - Ukrainian
- `sk_sk` - Slovak
- `bg_bg` - Bulgarian
- `fi_fi` - Finnish
- `ro_ro` - Romanian
- `hr_hr` - Croatian
- `cs_cz` - Czech
- `sv_se` - Swedish
- `et_ee` - Estonian
- `hu_hu` - Hungarian
- `lt_lt` - Lithuanian
- `da_dk` - Danish
- `mt_mt` - Maltese
- `sl_si` - Slovenian
- `lv_lv` - Latvian
- `el_gr` - Greek

## Custom Benchmarks

### Python API Example

```python
from eddy import ParakeetASR
import time

# Initialize model
asr = ParakeetASR("parakeet-v3", device="NPU")

# Transcribe and measure performance
audio_file = "test.wav"
start_time = time.time()
result = asr.transcribe(audio_file)
elapsed = time.time() - start_time

print(f"Text: {result['text']}")
print(f"Time: {elapsed:.2f}s")
print(f"RTFx: {result['rtfx']:.2f}×")
```

### C++ API Example

See [docs/CPP_API.md](docs/CPP_API.md) for C++ integration examples.

## Hardware Recommendations

### Best Performance: Intel NPU

- **Devices**: Intel Core Ultra (Meteor Lake or newer)
- **Expected RTFx**: 30-40× for Parakeet, 15-20× for Whisper
- **Power efficiency**: Best for battery-powered devices

### CPU Fallback

- **Expected RTFx**: 5-10× for Parakeet, 0.4-0.5× for Whisper
- **Works on**: Any modern x86-64 CPU
- **Use when**: NPU not available

### GPU (Experimental)

- **Expected RTFx**: Varies by GPU (integrated vs discrete)
- **Note**: Best results with discrete GPUs

## Troubleshooting

### Slow Performance

1. Verify OpenVINO 2025.x is installed
2. Check device availability: `parakeet_cli.exe --list-devices`
3. Use `--device NPU` for Intel Core Ultra processors
4. Ensure Release build (Debug is ~10× slower)

### Out of Memory

- Reduce batch size in benchmark scripts
- Use smaller model (V2 instead of V3, or Whisper base instead of large)
- Close other applications

### Dataset Download Issues

LibriSpeech and FLEURS datasets auto-download on first run. If download fails:

```bash
# Manual download
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz

# Or use HuggingFace datasets library
pip install datasets
python -c "from datasets import load_dataset; load_dataset('google/fleurs', 'en_us')"
```

## Contributing Benchmark Results

Share your results with the community:

1. Run benchmarks on your hardware
2. Note your CPU/GPU model and OS
3. Submit results via GitHub Issues or Discord
4. Help us understand performance across different platforms

## Support

- **GitHub Issues**: [github.com/FluidInference/eddy/issues](https://github.com/FluidInference/eddy/issues)
- **Discord**: [discord.gg/WNsvaCtmDe](https://discord.gg/WNsvaCtmDe)
