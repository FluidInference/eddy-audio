#!/usr/bin/env python3
"""
Minimal LibriSpeech benchmark runner using Eddy C API via ctypes.

Requirements:
  pip install datasets jiwer numpy

Example:
  python scripts/bench_parakeet_ctypes.py \
    --lib build\\Release\\eddy_c.dll --device CPU --max 25
"""

import argparse
import ctypes as C
import os
import re
import sys
import time
from typing import Tuple

import numpy as np

try:
    from datasets import load_dataset, Audio  # type: ignore
except Exception as e:
    load_dataset = None  # lazy error later


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
    lib = C.CDLL(path)

    # Signatures
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


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wer_stats(ref: str, hyp: str) -> Tuple[int, int]:
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    # Levenshtein distance
    R, H = len(r), len(h)
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        dp[i][0] = i
    for j in range(H + 1):
        dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # del
                dp[i][j - 1] + 1,  # ins
                dp[i - 1][j - 1] + cost,  # sub
            )
    return dp[R][H], max(1, R)


def main():
    ap = argparse.ArgumentParser(description="Parakeet LibriSpeech benchmark via C ABI")
    ap.add_argument("--lib", required=True, help="Path to eddy_c shared library (e.g., build/Release/eddy_c.dll)")
    ap.add_argument("--device", default="CPU", help="Device: CPU/GPU/NPU/AUTO (default: CPU)")
    ap.add_argument("--model-dir", default=None, help="Directory with parakeet model files (defaults to Eddy cache)")
    ap.add_argument("--max", type=int, default=50, help="Max files to evaluate")
    ap.add_argument("--dataset-config", default="clean", help="HF datasets config (e.g., clean, other)")
    ap.add_argument("--split", default="test.clean", help="HF datasets split (e.g., test.clean, validation.clean)")
    args = ap.parse_args()

    if load_dataset is None:
        print("Please 'pip install datasets' to run this benchmark.", file=sys.stderr)
        sys.exit(2)

    # Load and resample dataset to 16kHz
    print(f"Loading dataset: librispeech_asr/{args.dataset_config} {args.split}")
    ds = load_dataset("librispeech_asr", args.dataset_config, split=args.split)  # type: ignore
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))  # type: ignore

    # Load library and create model
    lib = load_lib(args.lib)
    err = C.c_char_p()
    cfg = EddyParakeetConfig(
        device=args.device.encode("utf-8"),
        model_dir=args.model_dir.encode("utf-8") if args.model_dir else None,
        blank_token_id=1024,
    )
    handle = lib.eddy_parakeet_create(cfg, C.byref(err))
    if not handle:
        msg = err.value.decode("utf-8") if err.value else "unknown error"
        print(f"Failed to create Parakeet model: {msg}", file=sys.stderr)
        sys.exit(1)
    if err.value:
        # Clear any informational messages
        lib.eddy_free_string(err)
        err = C.c_char_p()

    print("Running inference ...")
    total_edits = 0
    total_words = 0
    latencies = []
    n = min(args.max, len(ds))
    for i in range(n):
        ex = ds[i]
        ref = ex.get("text", "")
        audio = ex["audio"]
        pcm = np.asarray(audio["array"], dtype=np.float32)
        buf = np.ascontiguousarray(pcm)
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
            print(f"[{i+1}/{n}] Inference failed: {msg}", file=sys.stderr)
            if call_err.value:
                lib.eddy_free_string(call_err)
            continue

        hyp = (res.text or b"").decode("utf-8", errors="ignore")
        edits, words = wer_stats(ref, hyp)
        total_edits += edits
        total_words += words
        latencies.append(res.latency_ms)
        lib.eddy_parakeet_free_result(C.byref(res))
        if call_err.value:
            lib.eddy_free_string(call_err)

        print(f"[{i+1}/{n}] WER={edits/words:.3f}  latency={latencies[-1]:.1f} ms")

    lib.eddy_parakeet_destroy(handle)

    wer = total_edits / max(1, total_words)
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    print("---")
    print(f"Samples: {n}")
    print(f"WER: {wer:.3f}")
    print(f"Latency p50/p95: {p50:.1f}/{p95:.1f} ms")


if __name__ == "__main__":
    main()
