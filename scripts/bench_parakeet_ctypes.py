#!/usr/bin/env python3
"""
Python benchmark example using Eddy's C ABI (ctypes).

Requirements:
  pip install datasets jiwer

Usage:
  python scripts/bench_parakeet_ctypes.py --lib build/Release/eddy_c.dll --device CPU --max 25
"""

import argparse
import ctypes
import os
import sys
import time
from ctypes import c_char_p, c_int, c_size_t, c_double, c_float, c_void_p, POINTER, byref


def load_lib(path_hint: str | None) -> ctypes.CDLL:
    if path_hint and os.path.exists(path_hint):
        return ctypes.CDLL(path_hint)
    # Try common Windows build locations
    candidates = [
        os.path.join("build", "Release", "eddy_c.dll"),
        os.path.join("build", "Debug", "eddy_c.dll"),
        os.path.join("build", "bin", "eddy_c.dll"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return ctypes.CDLL(p)
    raise SystemExit("Could not find eddy_c shared library. Pass --lib <path>.")


def define_api(lib: ctypes.CDLL):
    # Mirror eddy_c.h for Parakeet
    class EddyParakeetConfig(ctypes.Structure):
        _fields_ = [
            ("device", c_char_p),
            ("model_dir", c_char_p),
            ("blank_token_id", c_int),
        ]

    class EddyParakeetResult(ctypes.Structure):
        _fields_ = [
            ("text", c_char_p),
            ("token_ids", POINTER(c_int)),
            ("num_tokens", c_size_t),
            ("confidence", c_float),
            ("latency_ms", c_double),
        ]

    lib.eddy_parakeet_create.argtypes = [EddyParakeetConfig, POINTER(c_char_p)]
    lib.eddy_parakeet_create.restype = c_void_p
    lib.eddy_parakeet_destroy.argtypes = [c_void_p]
    lib.eddy_parakeet_infer_buffer.argtypes = [c_void_p, POINTER(c_float), c_size_t, c_int, POINTER(EddyParakeetResult), POINTER(c_char_p)]
    lib.eddy_parakeet_infer_buffer.restype = c_int
    lib.eddy_parakeet_free_result.argtypes = [POINTER(EddyParakeetResult)]
    lib.eddy_parakeet_decode_tokens.argtypes = [c_void_p, POINTER(c_int), c_size_t]
    lib.eddy_parakeet_decode_tokens.restype = c_char_p

    return EddyParakeetConfig, EddyParakeetResult


def main():
    ap = argparse.ArgumentParser(description="Eddy Parakeet LibriSpeech benchmark (Python ctypes)")
    ap.add_argument("--lib", help="Path to eddy_c shared library (eddy_c.dll)")
    ap.add_argument("--device", default="CPU", help="Device: CPU/GPU/NPU/AUTO")
    ap.add_argument("--max", type=int, default=25, help="Max files")
    args = ap.parse_args()

    try:
        from datasets import load_dataset, Audio
        from jiwer import wer
        import numpy as np
    except Exception as e:
        print("Please install dependencies: pip install datasets jiwer numpy", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        return 2

    lib = load_lib(args.lib)
    EddyParakeetConfig, EddyParakeetResult = define_api(lib)

    # Create model (use cache model_dir by passing NULL)
    err = c_char_p()
    cfg = EddyParakeetConfig(device=args.device.encode(), model_dir=None, blank_token_id=1024)
    handle = lib.eddy_parakeet_create(cfg, byref(err))
    if not handle:
        raise SystemExit((err.value or b"unknown error").decode())

    # Load LibriSpeech test clean and decode to 16kHz mono
    ds = load_dataset("librispeech_asr", "clean", split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.select(range(min(args.max, len(ds))))

    total_wer = 0.0
    total_audio = 0.0
    total_time = 0.0

    for i, ex in enumerate(ds):
        ref = ex["text"]
        audio = ex["audio"]["array"].astype("float32")
        dur = audio.shape[0] / 16000.0
        pcm_ptr = audio.ctypes.data_as(POINTER(c_float))

        res = EddyParakeetResult()
        err = c_char_p()
        t0 = time.time()
        rc = lib.eddy_parakeet_infer_buffer(handle, pcm_ptr, audio.shape[0], 16000, byref(res), byref(err))
        t1 = time.time()
        if rc != 0:
            msg = (err.value or b"inference failed").decode()
            print(f"[{i+1}] ERROR: {msg}")
            continue

        hyp = (res.text or b"").decode()
        w = wer(ref, hyp)
        print(f"[{i+1}/{len(ds)}] WER={w*100:.1f}%  dur={dur:.1f}s  time={(t1-t0):.2f}s")

        total_wer += w
        total_audio += dur
        total_time += (t1 - t0)

        lib.eddy_parakeet_free_result(byref(res))

    if len(ds) > 0:
        avg_wer = (total_wer / len(ds)) * 100.0
        rtfx = total_audio / total_time if total_time > 0 else 0.0
        print(f"\nAvg WER: {avg_wer:.2f}%  Overall RTFx: {rtfx:.1f}x")

    lib.eddy_parakeet_destroy(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

