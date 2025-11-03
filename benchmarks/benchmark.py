#!/usr/bin/env python3
"""
LibriSpeech benchmark using Eddy C API via ctypes.

This is the CLEAN approach:
- C++ library handles inference only
- Python handles dataset loading, orchestration, and WER calculation
- Direct C API calls via ctypes (no subprocess overhead)

Workflow:
1. Python rebuilds C++ library to ensure latest code
2. Python loads LibriSpeech dataset using HuggingFace datasets
3. Python calls C++ inference via eddy_c library
4. Python normalizes output using Whisper normalizer
5. Python calculates WER with jiwer

Usage:
    uv run benchmark.py --max-files 2620 --device CPU
    uv run benchmark.py --max-files 100 --device NPU
    uv run benchmark.py --no-rebuild  # Skip rebuild step
"""

import argparse
import ctypes as C
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Import dependencies (automatically managed by uv when running: uv run benchmark.py)
from datasets import load_dataset, Audio
import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

# Initialize Whisper's English text normalizer (industry standard)
english_normalizer = EnglishTextNormalizer()


# ----- C types matching include/eddy/eddy_c.h -----

class EddyParakeetConfig(C.Structure):
    _fields_ = [
        ("device", C.c_char_p),
        ("model_dir", C.c_char_p),
        ("blank_token_id", C.c_int),
    ]


class EddyParakeetResult(C.Structure):
    _fields_ = [
        ("text", C.c_char_p),
        ("token_ids", C.POINTER(C.c_int)),
        ("num_tokens", C.c_size_t),
        ("confidence", C.c_float),
        ("latency_ms", C.c_double),
    ]


def load_lib(path: str) -> C.CDLL:
    """Load eddy_c library and configure function signatures"""
    lib = C.CDLL(path)

    # Function signatures
    lib.eddy_parakeet_create.argtypes = [EddyParakeetConfig, C.POINTER(C.c_char_p)]
    lib.eddy_parakeet_create.restype = C.c_void_p

    lib.eddy_parakeet_destroy.argtypes = [C.c_void_p]
    lib.eddy_parakeet_destroy.restype = None

    lib.eddy_parakeet_infer_buffer.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_float),
        C.c_size_t,
        C.c_int,
        C.POINTER(EddyParakeetResult),
        C.POINTER(C.c_char_p),
    ]
    lib.eddy_parakeet_infer_buffer.restype = C.c_int

    lib.eddy_parakeet_free_result.argtypes = [C.POINTER(EddyParakeetResult)]
    lib.eddy_parakeet_free_result.restype = None

    lib.eddy_free_string.argtypes = [C.c_char_p]
    lib.eddy_free_string.restype = None

    return lib


def normalize_text(text: str) -> str:
    """
    Normalize text using OpenAI Whisper's normalizer.

    This is the industry standard used by:
    - OpenAI Whisper
    - Hugging Face Open ASR Leaderboard
    - FluidAudio/Parakeet benchmarks
    """
    return english_normalizer(text)


def normalize_text_cpp_style(text: str) -> str:
    """
    Normalize text to match C++ TextNormalizer behavior:
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    import string
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def calculate_wer(hypothesis: str, reference: str) -> Dict[str, Any]:
    """Calculate WER and CER metrics using jiwer with Whisper normalization"""
    hyp_norm = normalize_text(hypothesis)
    ref_norm = normalize_text(reference)

    # Calculate WER
    wer_score = jiwer.wer(ref_norm, hyp_norm)

    # Get detailed word measures
    output = jiwer.process_words(ref_norm, hyp_norm)

    # Calculate CER (Character Error Rate)
    cer_score = jiwer.cer(ref_norm, hyp_norm)

    return {
        "wer": wer_score * 100,  # as percentage
        "cer": cer_score * 100,  # as percentage
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
    }


def rebuild_cpp_library() -> None:
    """Rebuild the C++ eddy_c library to ensure latest code is used"""
    print("Rebuilding C++ library...")
    print("=" * 80)

    # Get project root directory (parent of benchmarks/)
    project_root = Path(__file__).parent.parent
    build_dir = project_root / "build"

    if not build_dir.exists():
        print(f"ERROR: Build directory not found: {build_dir}", file=sys.stderr)
        print("Run 'cmake -B build' from project root first", file=sys.stderr)
        sys.exit(1)

    # Determine build command based on platform
    import platform
    if platform.system() == "Windows":
        build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release", "--target", "eddy_c"]
    else:
        build_cmd = ["cmake", "--build", str(build_dir), "--target", "eddy_c"]

    try:
        result = subprocess.run(build_cmd, check=True, capture_output=True, text=True, cwd=str(project_root))
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print("Build successful!")
        print("=" * 80)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Build failed with exit code {e.returncode}", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: cmake not found. Make sure CMake is installed and in PATH", file=sys.stderr)
        sys.exit(1)


def find_eddy_c_lib() -> Path:
    """Auto-detect eddy_c library path"""
    # Get project root directory (parent of benchmarks/)
    project_root = Path(__file__).parent.parent

    # Common locations relative to project root
    candidates = [
        project_root / "build/Release/eddy_c.dll",  # Windows
        project_root / "build/Debug/eddy_c.dll",
        project_root / "build/libeddy_c.so",  # Linux
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find eddy_c library. Build it first:\n"
        f"  cmake --build {project_root}/build --config Release --target eddy_c"
    )


def main():
    ap = argparse.ArgumentParser(description="Parakeet LibriSpeech benchmark via C API")
    ap.add_argument("--lib", default=None, help="Path to eddy_c shared library (auto-detected if not specified)")
    ap.add_argument("--device", default="CPU", help="Device: CPU/GPU/NPU/AUTO (default: CPU)")
    ap.add_argument("--model-dir", default=None, help="Directory with parakeet model files (defaults to Eddy cache)")
    ap.add_argument("--model", choices=["parakeet-v2", "parakeet-v3"], default="parakeet-v2", help="Model version (default: parakeet-v2)")
    ap.add_argument("--max-files", type=str, default="50", help="Max files to evaluate (default: 50, use 'all' for full dataset)")
    ap.add_argument("--dataset-config", default="clean", help="HF datasets config (default: clean)")
    ap.add_argument("--split", default="test", help="HF datasets split (default: test)")
    ap.add_argument("--output", default="eddy_benchmark_results.json", help="Output JSON file")
    ap.add_argument("--no-rebuild", action="store_true", help="Skip rebuilding C++ library (use existing build)")
    args = ap.parse_args()

    # Handle 'all' for max-files
    if args.max_files.lower() == "all":
        args.max_files = float('inf')  # Will be capped by dataset size
    else:
        args.max_files = int(args.max_files)

    # Rebuild C++ library unless --no-rebuild is specified
    if not args.no_rebuild:
        rebuild_cpp_library()

    # Find library
    lib_path = Path(args.lib) if args.lib else find_eddy_c_lib()
    if not lib_path.exists():
        print(f"ERROR: Library not found: {lib_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Using library: {lib_path}")

    # Load dataset
    # First try loading from local LibriSpeech directory (avoids 100GB+ HF download)
    local_test_dir = Path.home() / ".cache" / "librispeech" / "LibriSpeech" / "test-clean"

    if local_test_dir.exists() and args.split == "test":
        print(f"Loading dataset from local directory: {local_test_dir}")
        # Load from local LibriSpeech format
        from datasets import Dataset

        # LibriSpeech structure: speaker_id/chapter_id/audio_files.flac + transcripts
        examples = []
        for speaker_dir in sorted(local_test_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue

                # Load transcript file
                trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                if not trans_file.exists():
                    continue

                transcripts = {}
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]

                # Load audio files
                for flac_file in sorted(chapter_dir.glob("*.flac")):
                    file_id = flac_file.stem
                    if file_id in transcripts:
                        examples.append({
                            "id": file_id,
                            "audio": str(flac_file),
                            "text": transcripts[file_id]
                        })

        print(f"Loaded {len(examples)} examples from local LibriSpeech")
        ds = Dataset.from_list(examples)
        # Don't cast to Audio type - we'll load files manually with soundfile
    else:
        # Fallback to HuggingFace (will download everything)
        print(f"Loading dataset: librispeech_asr/{args.dataset_config} {args.split}")
        print(f"Note: This downloads all splits (~15GB). For test-only, run:")
        print(f"  python /tmp/download_test_only.py")
        ds = load_dataset("librispeech_asr", args.dataset_config, split=args.split)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Load library and create model
    print(f"Loading model {args.model} on device: {args.device}")
    lib = load_lib(str(lib_path))
    err = C.c_char_p()

    # Set blank_token_id based on model version
    blank_token_id = 8192 if args.model == "parakeet-v3" else 1024

    # Auto-resolve model directory from cache if not explicitly provided
    # The C++ code will use get_model_assets_dir(model_name) based on blank_token_id
    # So we can pass None and let C++ handle it, OR explicitly construct the path
    model_dir_to_use = args.model_dir
    if not model_dir_to_use:
        # Let C++ auto-select based on blank_token_id (it infers model name from it)
        # This ensures we use the correct cache directory: ~/.cache/eddy/models/parakeet-v{2,3}/
        model_dir_to_use = None

    cfg = EddyParakeetConfig(
        device=args.device.encode("utf-8"),
        model_dir=model_dir_to_use.encode("utf-8") if model_dir_to_use else None,
        blank_token_id=blank_token_id,
    )
    handle = lib.eddy_parakeet_create(cfg, C.byref(err))
    if not handle:
        msg = err.value.decode("utf-8") if err.value else "unknown error"
        print(f"ERROR: Failed to create Parakeet model: {msg}", file=sys.stderr)
        sys.exit(1)
    if err.value:
        lib.eddy_free_string(err)

    print(f"\nRunning inference on {min(args.max_files, len(ds))} files...")
    print("=" * 80)

    results = []

    total_audio_duration = 0.0
    total_processing_time = 0.0
    start_time = time.time()

    n = min(args.max_files, len(ds))
    for i in range(n):
        ex = ds[i]
        file_id = ex.get("id", f"file_{i}")
        ref = ex.get("text", "")

        # Load audio manually since torchcodec isn't available
        audio_path = ex["audio"]
        if isinstance(audio_path, str):
            # Local file path - load with soundfile
            import soundfile as sf
            pcm, sr = sf.read(audio_path, dtype='float32')
            if sr != 16000:
                raise ValueError(
                    f"Expected 16kHz audio, got {sr}Hz for {file_id}. "
                    f"LibriSpeech should be 16kHz. Please check the audio file."
                )
        else:
            # HuggingFace dataset format
            pcm = np.asarray(audio_path["array"], dtype=np.float32)

        buf = np.ascontiguousarray(pcm)

        # Calculate audio duration
        audio_duration = len(pcm) / 16000.0

        # Run inference
        res = EddyParakeetResult()
        call_err = C.c_char_p()
        rc = lib.eddy_parakeet_infer_buffer(
            handle,
            buf.ctypes.data_as(C.POINTER(C.c_float)),
            C.c_size_t(buf.size),
            C.c_int(16000),
            C.byref(res),
            C.byref(call_err),
        )

        if rc != 0:
            msg = call_err.value.decode("utf-8") if call_err.value else f"error code {rc}"
            print(f"[{i+1}/{n}] ERROR: {msg}", file=sys.stderr)
            if call_err.value:
                lib.eddy_free_string(call_err)
            continue

        hyp = (res.text or b"").decode("utf-8", errors="ignore")

        # Calculate WER and CER with OpenAI Whisper normalization
        wer_metrics = calculate_wer(hyp, ref)

        total_audio_duration += audio_duration
        total_processing_time += res.latency_ms / 1000.0

        # Store result
        result = {
            "file_id": file_id,
            "reference": ref,
            "hypothesis": hyp,
            "wer": wer_metrics["wer"],
            "cer": wer_metrics["cer"],
            "audio_duration_sec": audio_duration,
            "processing_time_sec": res.latency_ms / 1000.0,
            "rtfx": audio_duration / (res.latency_ms / 1000.0),
            "confidence": res.confidence,
        }
        results.append(result)

        # Free resources
        lib.eddy_parakeet_free_result(C.byref(res))
        if call_err.value:
            lib.eddy_free_string(call_err)

        # Progress update
        if (i + 1) % 10 == 0 or i == n - 1:
            # Calculate current average WER (per-file average, not corpus-level)
            current_wer_values = [r["wer"] for r in results]
            current_avg_wer = sum(current_wer_values) / len(current_wer_values) if current_wer_values else 0.0
            current_rtfx = total_audio_duration / max(0.001, total_processing_time)
            print(f"[{i+1}/{n}] Avg WER: {current_avg_wer:.2f}%  RTFx: {current_rtfx:.1f}x  Last: {wer_metrics['wer']:.2f}%")

    lib.eddy_parakeet_destroy(handle)

    # Calculate final metrics
    elapsed_time = time.time() - start_time
    overall_rtfx = total_audio_duration / max(0.001, total_processing_time)

    # Compute per-file WER and CER statistics
    wer_values = [r["wer"] for r in results]
    cer_values = [r["cer"] for r in results]
    rtfx_values = [r["rtfx"] for r in results]

    wer_values.sort()
    cer_values.sort()
    rtfx_values.sort()

    median_wer = wer_values[len(wer_values) // 2] if wer_values else 0.0
    median_cer = cer_values[len(cer_values) // 2] if cer_values else 0.0
    median_rtfx = rtfx_values[len(rtfx_values) // 2] if rtfx_values else 0.0

    # Calculate average metrics
    average_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
    average_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0

    # Save results
    output_data = {
        "config": {
            "device": args.device,
            "model_dir": args.model_dir or "cache",
            "model": args.model,
            "blank_token_id": blank_token_id,
            "dataset": f"librispeech_asr/{args.dataset_config}",
            "split": args.split,
            "num_files": n,
        },
        "metrics": {
            "average_wer": average_wer,
            "median_wer": median_wer,
            "average_cer": average_cer,
            "median_cer": median_cer,
            "median_rtfx": median_rtfx,
            "overall_rtfx": overall_rtfx,
            "total_audio_duration_sec": total_audio_duration,
            "total_processing_time_sec": total_processing_time,
            "benchmark_elapsed_sec": elapsed_time,
            "normalization": "OpenAI Whisper English",
        },
        "per_file_results": results,
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary (FluidAudio-style format)
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"   Dataset: librispeech {args.split}-{args.dataset_config}")
    print(f"   Model: {args.model}")
    print(f"   Device: {args.device}")
    print(f"   Files processed: {n}")
    print(f"   Average WER: {average_wer:.1f}%")
    print(f"   Median WER: {median_wer:.1f}%")
    print(f"   Average CER: {average_cer:.1f}%")
    print(f"   Median CER: {median_cer:.1f}%")
    print(f"   Median RTFx: {median_rtfx:.1f}x")
    print(f"   Overall RTFx: {overall_rtfx:.1f}x ({total_audio_duration:.1f}s / {total_processing_time:.1f}s)")
    print(f"   Benchmark runtime: {elapsed_time:.1f}s")
    print(f"Results saved to: {output_path}")
    print(f"   Normalization: OpenAI Whisper English")
    print("=" * 80)
    print()
    print("=" * 80)
    print("REFERENCE BENCHMARKS - FluidAudio CoreML (M4 Pro, 2620 files)")
    print("=" * 80)
    print("Model      | Avg WER | Med WER | Avg CER | Med RTFx | Overall RTFx | Runtime")
    print("-----------+---------+---------+---------+----------+--------------+---------")
    print("v2 (EN)    |   2.2%  |   0.0%  |   0.7%  |  125.6x  |    141.2x    |  3m 25s")
    print("v3 (multi) |   2.6%  |   0.0%  |   1.1%  |  137.8x  |    153.4x    |  3m  2s")
    print("=" * 80)
    print("Note: v2 is more accurate for English, v3 is faster and supports multilingual")
    print("=" * 80)


if __name__ == "__main__":
    main()
