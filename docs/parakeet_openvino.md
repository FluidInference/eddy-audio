# Parakeet v3 Setup (OpenVINO, Windows/NPU)

This guide gets you to a working Parakeet v3 setup with Eddy’s C++ runtime, using OpenVINO on Windows. It covers two paths:

- Option A (recommended, fastest): Use pre‑converted OpenVINO IR models from Hugging Face
- Option B (advanced): Convert from NeMo → ONNX → OpenVINO IR locally

It also includes run commands, performance knobs, and troubleshooting notes.

---

## Prerequisites

- Windows 10/11 x64
- Visual Studio 2022 (for building Eddy)
- OpenVINO 2025.0 (Runtime and tools)
  - Initialize environment for each shell: `"C:\\Program Files (x86)\\Intel\\openvino_2025.0.0\\setupvars.bat"`
- Optional NPU driver/runtime if you plan to run on NPU

---

## Build Eddy (once)

```powershell
cd C:\Users\brand\code\eddy
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DEDDY_ENABLE_OPENVINO=ON
cmake --build build --config Release --target eddy parakeet_cli hf_fetch_models
```

---

## Option A — Use Pre‑Converted OpenVINO IRs (Recommended)

This is the quickest way to get v3 running.

1) Download IRs to Eddy’s cache (automatic or manual):

- Easiest: use the downloader

```powershell
build\examples\cpp\Release\hf_fetch_models.exe --model parakeet-v3
```

- Manual: download from Hugging Face into the cache folder

```
%LOCALAPPDATA%\eddy\models\parakeet-v3\files
  parakeet_melspectogram.xml/bin
  parakeet_encoder.xml/bin
  parakeet_decoder.xml/bin
  parakeet_joint.xml/bin
  parakeet_v3_vocab.json
```

2) (Optional) Keep a repo‑local copy for fallback:

```powershell
mkdir -Force models\parakeet
copy %LOCALAPPDATA%\eddy\models\parakeet-v3\files\* models\parakeet\
```

---

## Option B — Local Conversion (NeMo → ONNX → OpenVINO IR)

Only use if you need to regenerate IRs. This requires a large Python toolchain.

1) Open a shell in `notebooks\openvino\parakeet` and use an ephemeral env with `uv` (avoids project conflicts):

```powershell
Push-Location notebooks\openvino\parakeet
uv run --no-project `
  --with numpy --with torch --with torchvision --with torchaudio `
  --with librosa --with soundfile --with openvino `
  --with hydra-core --with lightning --with pytorch-lightning `
  --with cloudpickle --with fiddle --with lhotse `
  --with nemo_toolkit --with onnx --with einops `
  --with transformers --with sentencepiece --with pandas --with texterrors `
  python convert_to_openvino.py --output-dir models_v3_ov
Pop-Location
```

2) Copy the generated IRs into Eddy’s cache:

```powershell
copy notebooks\openvino\parakeet\models_v3_ov\* `
     %LOCALAPPDATA%\eddy\models\parakeet-v3\files\
```

Expected outputs (names must match):

```
parakeet_melspectogram.xml/bin
parakeet_encoder.xml/bin
parakeet_decoder.xml/bin
parakeet_joint.xml/bin
parakeet_v3_vocab.json
```

> Note: The converter script sets input/output names to what Eddy expects. If you roll your own export, keep the same names, shapes, and dtypes.

---

## Run Commands

Always set the v3 mel‑frame window:

- Dynamic IRs (default): `EDDY_MAX_FRAMES=2000` (≈16 s, 125 fps)
- Static IRs (your new exports): `EDDY_MAX_FRAMES=1875` (15 s exact)

```powershell
set EDDY_MAX_FRAMES=2000
```

NPU environment:

```powershell
"C:\Program Files (x86)\Intel\openvino_2025.0.0\setupvars.bat"
```

Single file (CLI):

```powershell
build\examples\cpp\Release\parakeet_cli.exe eddy\assets\audio\first_10_seconds.wav --device NPU --model parakeet-v3
```

Benchmark (100 files, NPU):

```powershell
build\examples\cpp\Release\benchmark_librispeech.exe --model parakeet-v3 --max-files 100 --device NPU
```

Python benchmark:

```powershell
cd benchmarks
uv run benchmark.py --device NPU --model parakeet-v3 --max-files 100
```

---

## Static-Length IRs (T=1875)

If you’ve exported static-length v3 IRs with a 15 s window (T=1875):

- Place files in either location:
  - Cache (preferred): `%LOCALAPPDATA%\eddy\models\parakeet-v3\files`
  - Repo fallback: `models\parakeet`
- Eddy auto-detects T from the encoder input shape. No extra config is required.
- You may still set `EDDY_MAX_FRAMES=1875` to make the window explicit.
- The mel preprocessor runs on CPU by default (avoids known NPU compiler issues on Concat). You can override via `EDDY_PREPROC_DEVICE`, but CPU is recommended.

Example:

```powershell
set EDDY_MAX_FRAMES=1875
"C:\Program Files (x86)\Intel\openvino_2025.0.0\setupvars.bat"
build\examples\cpp\Release\parakeet_cli.exe eddy\assets\audio\first_10_seconds.wav --device NPU --model parakeet-v3
```

---

## Performance Knobs (Optional)

These can improve throughput. Use selectively and validate WER.

- Throughput hints
  - `EDDY_OV_PERF=THROUGHPUT`
  - `EDDY_OV_NUM_REQUESTS=4`

- Window/context (trade a bit of WER for speed)
  - `EDDY_MAX_FRAMES=1200..1600` (v3 default 2000)
  - `EDDY_CONTEXT_FRAMES=64`

- Joint input binding (code change)
  - Pre‑bind joint inputs and memcpy per step (mirrors v2). Reduces driver overhead.

- Quantization (advanced)
  - INT8 via OpenVINO POT for decoder/joint. Validate WER carefully.

---

## Known Differences (v2 vs v3)

- Time base
  - v2: ~192 encoder frames/15–16 s (12.5 fps)
  - v3: ~2000 mel frames/15–16 s (125 fps)

- Typical NPU throughput
  - v2: ~30–50× RTFx
  - v3: ~5–8× RTFx (10× more steps + larger heads)

This gap is expected due to the different architectures.

---

## Troubleshooting

- “to_shape was called on a dynamic shape”
  - Ensure you bind per‑step inputs properly (the Eddy code does this for v3), or set `EDDY_MAX_FRAMES` to your window (2000 dynamic IRs, 1875 static IRs).

- “Failed to open Parakeet vocabulary”
  - Ensure `parakeet_v3_vocab.json` is alongside the IRs in the same folder.

- Low throughput on NPU
  - Try `EDDY_OV_PERF=THROUGHPUT`, `EDDY_OV_NUM_REQUESTS=4`, and a smaller window. For static IRs, keep `EDDY_MAX_FRAMES` aligned with your export (e.g., 1875; you can also re‑export 1600 if desired).

- IR location
  - Cache: `%LOCALAPPDATA%\eddy\models\parakeet-v3\files`
  - Repo fallback: `models\parakeet` (if present)

---

## Summary

- Use Option A (prebuilt IRs) to get running fast.
- Always set `EDDY_MAX_FRAMES=2000` for v3.
- For NPU speed, apply throughput hints and consider a smaller window; accept a small WER trade‑off.
- If you must regenerate IRs, use the Option B script with `uv run --no-project` to avoid project name conflicts and heavy environment leaks.
