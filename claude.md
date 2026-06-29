# Claude Code Assistant Guidelines

## Git Policy

**NEVER commit changes** - User handles all git operations. You may use `git status`, `git diff`, `git log` for inspection only.

## Build After Changes

Always build immediately after code changes:
```bash
cmake --build build --config Release --target parakeet_cli
```

## Quick Reference

**Executable:** `build/examples/cpp/Release/parakeet_cli.exe`
**Model Cache:** `%LOCALAPPDATA%\eddy\cache\models\parakeet-v2\`
**Performance:** 5-8x RTF on CPU, 1.27% WER on LibriSpeech

## Project Goal

Match FluidAudio's Parakeet v2 (Swift/CoreML) implementation in C++/OpenVINO:
- ✅ 4-Model Pipeline, Greedy Decoding, LSTM State Continuity
- ✅ Token Deduplication, Timestamps, Confidence Scores
- ⏳ Decoder Output Caching (next priority)

**Current Results:** 1.27% WER (better than FluidAudio's 2.2% baseline)

## Architecture

Two first-class ASR paths are supported:

**Batch (Parakeet)** — `eddy::parakeet`, stateless overlapping-chunk encoder:
- 10s chunks with 3s overlap
- 2D search deduplication at boundaries
- LSTM state continuity across chunks

**Streaming (Nemotron)** — `eddy::nemotron`, cache-aware streaming FastConformer-RNNT:
- Carries `cache_channel`/`cache_time`/`cache_len` across chunks (`att_context=[56,0]`)
- Integer `prompt_id` per-chunk language conditioning (40+ languages)
- Plain RNNT greedy decode (no TDT duration bins)

## Testing

```bash
# Quick test
build/examples/cpp/Release/parakeet_cli.exe "assets/audio/first_15s.wav"

# Benchmark (C++ native, 2.5x faster than Python)
build/examples/cpp/Release/benchmark_librispeech.exe --max-files 25
```

## Key Implementation Notes

- Release builds only (Debug requires VS runtime DLLs)
- Token timestamps: frame × 0.08 = seconds
- Confidence: softmax over joint network logits
- Deduplication: checks last 20 tokens vs first 15 tokens of next chunk
