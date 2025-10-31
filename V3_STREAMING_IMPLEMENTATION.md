# V3 Streaming Algorithm Implementation Plan

## Problem Summary
Parakeet v3 in Eddy is 10x slower than it should be because it processes at mel frame granularity (125 fps) instead of encoder frame granularity (12.5 fps).

## Root Cause
- **Current**: Processes 2000 mel frames per 16-second window
- **Should be**: Process ~200-250 encoder frames per window
- **Impact**: 10x unnecessary computation

## The Fix

### 1. Update Frame Window Configuration
**File**: `src/models/parakeet-v2/parakeet_openvino.cpp`
**Line**: 456

```cpp
// BEFORE (line 456):
size_t def_frames = (impl_->runtime_cfg.blank_token_id == 8192) ? 2000 : 192;

// AFTER:
size_t def_frames = (impl_->runtime_cfg.blank_token_id == 8192) ? 225 : 192;
// V3: ~225 encoder frames for ~18 seconds (at 12.5 fps)
// V2: 192 encoder frames for ~15 seconds (at 12.5 fps)
```

### 2. Verify Encoder Output Shape
The encoder should output encoder frames, not mel frames:
- Input: Mel spectrogram (125 fps)
- Output: Encoder frames (12.5 fps)
- Downsampling ratio: 10x

### 3. Update Chunking Logic (if needed)
**File**: `src/models/parakeet-v2/parakeet_openvino.cpp`
**Function**: `process_mel_chunked`

Ensure the chunking works with encoder frame counts:
```cpp
// Line 239: Check against encoder frames, not mel frames
if (mel.frames <= max_frames) {
    // This should compare encoder frame equivalent
    // mel.frames / 10 <= max_frames for v3
}
```

### 4. Test Configuration
```bash
# Set environment variable to debug
set EDDY_DEBUG=1
set EDDY_MAX_FRAMES=225

# Test with v3
build/examples/cpp/Release/parakeet_cli.exe assets/audio/first_15s.wav --model parakeet-v3 --device CPU

# Should see:
# [DEBUG] V3 encoder using dynamic shape, defaulting to 225 encoder frames
```

## Streaming Algorithm Components

### Buffered Streaming (Already Partially Implemented)
The TDT duration skipping is already in place in `parakeet_decoder.cpp`:
- Lines 496-505: Duration extraction
- Lines 552-553: Frame advancement by duration

### What's Missing:
1. **Correct frame granularity** (main issue)
2. **Streaming buffer management** (for real-time audio)
3. **State persistence across chunks** (partially done)

## Performance Expectations

### Before Fix:
- Processing 2000 mel frames
- 6.8x RTFx on NPU
- 5.1x RTFx on CPU

### After Fix:
- Processing 225 encoder frames
- Expected 50-70x RTFx on NPU
- Expected 40-50x RTFx on CPU

## Testing Plan

### 1. Unit Test
```cpp
// Verify frame count
assert(impl_->encoder_expected_frames == 225); // for v3
assert(encoder.time_steps <= 250); // encoder output frames
```

### 2. Performance Test
```bash
# Before fix
build/examples/cpp/Release/benchmark_librispeech.exe --model parakeet-v3 --max-files 10
# Current: ~5x RTFx

# After fix
build/examples/cpp/Release/benchmark_librispeech.exe --model parakeet-v3 --max-files 10
# Expected: ~40-50x RTFx
```

### 3. Accuracy Test
```bash
# WER should remain the same or improve slightly
cd benchmarks
python benchmark.py --model parakeet-v3 --dataset librispeech --max-files 100
# Current WER: 1.87%
# Expected WER: 1.5-2.0% (similar)
```

## Implementation Timeline

1. **Immediate** (5 minutes):
   - Change line 456 in `parakeet_openvino.cpp`
   - Rebuild and test

2. **Short-term** (1 hour):
   - Verify encoder output shapes
   - Add debug logging
   - Run benchmarks

3. **Long-term** (if needed):
   - Implement true streaming with audio buffer
   - Add VAD (Voice Activity Detection)
   - Optimize memory usage

## Key Insight
**The models are correct!** This is purely a code fix. The v3 model already:
- Has proper encoder downsampling
- Includes TDT duration predictions
- Supports frame skipping

We just need to iterate at the correct granularity (encoder frames, not mel frames).

## References
- FluidAudio implementation: `FluidAudio/Sources/FluidAudio/ASR/ChunkProcessor.swift`
- Frame rates: 12.5 fps (encoder) vs 125 fps (mel)
- Window size: 14.4-18 seconds = 180-225 encoder frames