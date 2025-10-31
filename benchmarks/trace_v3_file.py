#!/usr/bin/env python3
"""
Trace Parakeet v3 decoding on a single audio file using the C API.

Prints hypothesis and leverages EDDY_V3_TRACE/EDDY_DEBUG logs for stepwise details.

Usage:
  uv run python trace_v3_file.py --file <path-to-flac-or-wav> --device NPU

Relies on eddy_c.dll built at build/Release/eddy_c.dll.
"""

import argparse
import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except Exception as e:
    print("soundfile missing. Run from benchmarks/ with uv: uv run python trace_v3_file.py ...", file=sys.stderr)
    raise


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


def load_lib(root: Path) -> C.CDLL:
    dll = root / "build/Release/eddy_c.dll"
    if not dll.exists():
        raise FileNotFoundError(f"Not found: {dll}")
    lib = C.CDLL(str(dll))

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


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    pcm, sr = sf.read(str(path), dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.float32)
    return np.ascontiguousarray(pcm, dtype=np.float32), int(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to FLAC/WAV to transcribe")
    ap.add_argument("--device", default="NPU", help="Device: CPU/GPU/NPU/AUTO (default: NPU)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    # Enable v3 model and trace logs
    os.environ["EDDY_PARAKEET_MODEL"] = "parakeet-v3"
    os.environ.setdefault("EDDY_DEBUG", "1")
    os.environ.setdefault("EDDY_V3_TRACE", "1")
    os.environ.setdefault("EDDY_V3_TRACE_LIMIT", "256")

    lib = load_lib(root)

    # Create model
    cfg = EddyParakeetConfig(
        device=args.device.encode("utf-8"),
        model_dir=None,
        blank_token_id=8192,
    )
    err = C.c_char_p()
    handle = lib.eddy_parakeet_create(cfg, C.byref(err))
    if not handle:
        raise RuntimeError((err.value or b"create failed").decode("utf-8"))
    if err.value:
        lib.eddy_free_string(err)

    # Load audio
    audio_path = Path(args.file)
    pcm, sr = read_audio(audio_path)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}")

    # Inference
    res = EddyParakeetResult()
    call_err = C.c_char_p()
    rc = lib.eddy_parakeet_infer_buffer(
        handle,
        pcm.ctypes.data_as(C.POINTER(C.c_float)),
        C.c_size_t(pcm.size),
        C.c_int(sr),
        C.byref(res),
        C.byref(call_err),
    )
    if rc != 0:
        msg = call_err.value.decode("utf-8") if call_err.value else f"error code {rc}"
        print(f"ERROR: {msg}", file=sys.stderr)
        if call_err.value:
            lib.eddy_free_string(call_err)
        sys.exit(1)

    text = (res.text or b"").decode("utf-8", errors="ignore")
    print("\n----- Hypothesis -----\n" + text + "\n----------------------\n")

    lib.eddy_parakeet_free_result(C.byref(res))
    if call_err.value:
        lib.eddy_free_string(call_err)
    lib.eddy_parakeet_destroy(handle)


if __name__ == "__main__":
    main()

