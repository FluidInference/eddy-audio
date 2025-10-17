# AGENTS.md

Guidelines for working in this repo as an automated agent.

## After Editing Parakeet/OpenVINO Code

When you touch code under `src/models/parakeet/` or related headers, make sure to:

1) Clear compiled caches (KEEP model files)

- Windows PowerShell
  - This removes compiled blobs under the Parakeet model cache while preserving `files/` with the downloaded models.
  - Path: `%LOCALAPPDATA%\eddy\cache\models\parakeet-v2`

```
$base = Join-Path $env:LOCALAPPDATA 'eddy\cache\models\parakeet-v2'
if (Test-Path $base) {
  Get-ChildItem -LiteralPath $base -File -Force | Remove-Item -Force -ErrorAction SilentlyContinue
  Get-ChildItem -LiteralPath $base -Directory -Force |
    Where-Object { $_.Name -ne 'files' } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
```

- Bash (Linux/macOS) — adapt path per platform
```
BASE="$HOME/.cache/eddy/models/parakeet-v2"  # or $XDG_CACHE_HOME/eddy/models/parakeet-v2
find "$BASE" -maxdepth 1 -type f -print -delete
find "$BASE" -mindepth 1 -maxdepth 1 -type d ! -name files -print -exec rm -rf {} +
```

Do NOT delete the `files/` subfolder. That folder contains the model XML/BIN/JSON.

2) Rebuild the targets

```
cmake --build build --config Release --target eddy parakeet_cli benchmark_librispeech
```

3) First run after cache clear

- The first NPU/GPU run will recompile models and can take a few minutes.
- Subsequent runs will use the newly built code and the fresh compiled cache.

## Optional: Run Quick Checks

- Single file (NPU):
```
build\examples\cpp\Release\parakeet_cli.exe "<path-to-wav>" --device NPU
```

- Benchmark (NPU):
```
build\examples\cpp\Release\benchmark_librispeech.exe --max-files 50 --device NPU
```

## Environment Knobs (for chunking/dedup)

- `EDDY_CONTEXT_FRAMES` (default `20`): overlap uses `2 * context_frames`.
- `EDDY_BOUNDARY_SEARCH_FRAMES` (default `20`): boundary window for duplicate search.
- `EDDY_DISABLE_HOLDBACK=1`: disable right-context holdback; rely on overlap dedup only.

## Model Download (if missing)

- Python helper:
```
python scripts\fetch_parakeet_ov.py
```
- Or use the setup script to fetch models and rebuild:
```
powershell -ExecutionPolicy Bypass -File scripts\setup_parakeet_models.ps1
```
p.s. A longer, end‑to‑end run/troubleshooting guide (including PowerShell quoting, OpenVINO setupvars usage, JSON filters, and multi‑chunk logs) lives at:

- `docs/Benchmark-Troubleshooting.md`

### Agent Quick Notes

- Use `run_bench_npu.bat` to ensure NPU runs always load the OpenVINO environment first.
- If `OpenVINO_DIR` is needed, default to: `C:\\Program Files (x86)\\Intel\\openvino_2025.0.0\\runtime\\cmake`.
- For focused regression JSON, use `--min-wer 10` to include only high‑WER cases in the output file.

