# Benchmark Results

Comprehensive benchmark results for eddy ASR on LibriSpeech test-clean and FLEURS multilingual datasets.

**Hardware**: Intel Core Ultra 7 155H (Meteor Lake) with Intel AI Boost NPU
**Software**: OpenVINO 2025.3.0
**Normalization**: OpenAI Whisper English normalizer

---

## LibriSpeech test-clean (English)

### Parakeet V2 (English-only, optimized)

| Metric | Value |
|--------|-------|
| **Dataset** | LibriSpeech test-clean |
| **Files processed** | 2,620 |
| **Average WER** | 2.87% |
| **Median WER** | 0.00% |
| **Average CER** | 1.07% |
| **Overall RTFx (NPU)** | 37.8× |
| **Total audio duration** | 19,452.5s (5.4 hours) |
| **Total processing time** | 514.7s |

**Comparison**:
- FluidAudio v2 (CoreML): 2.2% WER, 141× RTFx on M4 Pro
- eddy v2 (OpenVINO NPU): 2.87% WER, 37.8× RTFx on Intel Core Ultra 7 155H

### Parakeet V3 (Multilingual)

| Metric | Value |
|--------|-------|
| **Dataset** | LibriSpeech test-clean |
| **Model** | parakeet-v3 |
| **Device** | NPU |
| **Files processed** | 2,620 |
| **Average WER** | 3.7% |
| **Median WER** | 0.0% |
| **Average CER** | 1.9% |
| **Median CER** | 0.0% |
| **Median RTFx** | 23.5× |
| **Overall RTFx (NPU)** | 25.7× |
| **Total audio duration** | 19,452.5s (5.4 hours) |
| **Total processing time** | 756.4s |
| **Benchmark runtime** | 789.8s |

**Comparison**:
- FluidAudio v3 (CoreML, multilingual): 2.6% WER
- eddy v3 (OpenVINO NPU, multilingual): 3.7% WER

---

## FLEURS Multilingual Benchmark (24 Languages)

**Model**: Parakeet V3
**Device**: NPU
**Dataset**: FLEURS (Federated Learning Evaluation Representation United States)

| Language | WER | Ref WER | CER | RTFx | Samples |
|----------|-----|---------|-----|------|---------|
| **Italian (Italy)** | 4.3% | 3.0% | 2.1% | 43.6× | 350 |
| **Spanish (Spain)** | 5.4% | 3.5% | 2.8% | 43.1× | 350 |
| **English (US)** | 6.1% | 4.9% | 3.0% | 41.9× | 350 |
| **German (Germany)** | 7.4% | 5.0% | 2.9% | 42.8× | 350 |
| **French (France)** | 7.7% | 5.2% | 3.2% | 40.6× | 350 |
| **Dutch (Netherlands)** | 9.8% | 7.5% | 3.3% | 37.5× | 350 |
| **Russian (Russia)** | 9.9% | 5.5% | 2.5% | 39.7× | 350 |
| **Polish (Poland)** | 10.5% | 7.3% | 3.1% | 37.3× | 350 |
| **Ukrainian (Ukraine)** | 10.7% | 6.8% | 2.9% | 39.3× | 350 |
| **Slovak (Slovakia)** | 11.1% | 8.8% | 3.5% | 43.7× | 350 |
| **Bulgarian (Bulgaria)** | 16.8% | 12.6% | 4.7% | 41.7× | 350 |
| **Finnish (Finland)** | 16.8% | 13.2% | 3.7% | 41.5× | 918 |
| **Romanian (Romania)** | 17.5% | 12.4% | 5.9% | 38.9× | 883 |
| **Croatian (Croatia)** | 17.8% | 12.5% | 5.8% | 41.0× | 350 |
| **Czech (Czechia)** | 18.5% | 11.0% | 5.3% | 43.1× | 350 |
| **Swedish (Sweden)** | 18.9% | 15.1% | 5.6% | 41.5× | 759 |
| **Hungarian (Hungary)** | 20.7% | 15.7% | 6.4% | 41.1× | 905 |
| **Estonian (Estonia)** | 20.8% | 17.7% | 4.9% | 43.4× | 893 |
| **Lithuanian (Lithuania)** | 24.6% | 20.4% | 6.7% | 40.4× | 986 |
| **Danish (Denmark)** | 25.4% | 18.4% | 9.3% | 44.0× | 930 |
| **Maltese (Malta)** | 25.3% | 20.5% | 9.2% | 41.3× | 926 |
| **Slovenian (Slovenia)** | 28.1% | 24.0% | 9.4% | 38.7× | 834 |
| **Latvian (Latvia)** | 30.6% | 22.8% | 8.1% | 42.6× | 851 |
| **Greek (Greece)** | 42.7% | 20.7% | 15.0% | 37.2× | 650 |

### FLEURS Summary

| Metric | Value |
|--------|-------|
| **Average WER** | 17.0% |
| **Reference WER** | 12.7% |
| **Average CER** | 5.4% |
| **Average RTFx** | 41.1× |
| **Languages** | 24 |
| **Total samples** | ~15,000+ |

---

## Nemotron Streaming Multilingual 0.6B (FLEURS)

**Model**: `nemotron-streaming-int8` (weight-only INT8 encoder, FP16 decoder/joint)
**Device**: Intel NPU · **Software**: OpenVINO 2025.0 · **Decoding**: greedy, forced language
**Preprocessor**: native C++ log-mel featurizer (replaces the dynamic-shape OV preprocessor)
**Scoring**: FluidAudio methodology — WER for spaced languages, character-level CER for CJK
(`ja`/`zh`), Whisper-style punctuation/symbol stripping

| Language | Metric | eddy NPU (int8) | OpenVINO FP32 ref | FluidAudio CoreML ref | RTFx | Samples |
|----------|--------|----------------:|------------------:|----------------------:|-----:|--------:|
| English (US) | WER | 12.48 | 11.78 | 12.09 | 22.3× | 350 |
| Spanish (LatAm) | WER | 7.03 | 6.99 | 9.01 | 22.5× | 350 |
| French (France) | WER | 13.35 | 12.92 | 15.18 | 21.7× | 350 |
| Chinese (Mandarin) | CER | 20.18 | 21.05 | 24.54 | 23.7× | 945 |
| Japanese | CER | 15.46 | 15.12 | 16.86 | 23.3× | 650 |

**Audio-weighted RTFx**: ~22–24× on Intel NPU (≈2× the CPU figure of ~11×).

**Notes**:
- INT8-on-NPU accuracy matches the OpenVINO FP32 reference within noise and beats the
  FluidAudio CoreML reference on es/fr/zh/ja.
- **NPU enablement**: the OpenVINO NPU plugin miscompiles `BitwiseNot` on a boolean
  (integer complement → mask all-true → encoder collapses to ~0 → empty transcripts).
  eddy rewrites `BitwiseNot → LogicalNot` in the encoder IR before compiling (no-op on
  CPU/GPU); without it the NPU produces empty output for this model.
- Unlike Parakeet (overlapping-chunk + 2D dedup), Nemotron is cache-aware streaming RNNT
  with per-chunk `prompt_id` language conditioning.

---

## Performance Notes

### Best Performing Languages (WER < 10%)
1. Italian: 4.3%
2. Spanish: 5.4%
3. English: 6.1%
4. German: 7.4%
5. French: 7.7%
6. Dutch: 9.8%
7. Russian: 9.9%

### RTFx Consistency
- NPU performance is very consistent across languages (37-44× RTFx)
- Average RTFx: 41.1× across all 24 languages
- Minimal variance indicates efficient NPU utilization

### Accuracy vs Reference
- Our WER is ~4.3% higher than reference WER on average
- This delta is consistent across most languages
- Likely due to differences in:
  - Text normalization approach
  - Model quantization (int8 for NPU optimization)
  - Greedy vs beam search decoding

---

## Methodology

- **Text Normalization**: OpenAI Whisper English normalizer (industry standard)
- **WER Calculation**: jiwer library
- **Audio Format**: 16kHz mono WAV
- **Inference**: Batch processing with 10-second chunks, 3-second overlap
- **State Management**: LSTM state continuity across chunks
- **Deduplication**: 2D search algorithm at chunk boundaries

See [FLEURS_BENCHMARK.md](FLEURS_BENCHMARK.md) for detailed FLEURS benchmark methodology and implementation.
