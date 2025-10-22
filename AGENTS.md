# Repository Guidelines

## Project Structure & Module Organization

- Source in `src/` (core, backends, `models/parakeet/`, pipelines, streaming); public headers in `include/eddy/`.
- Examples in `examples/cpp/` (`parakeet_cli.cpp`, `benchmark_librispeech.cpp`, optional `whisper_example.cpp`).
- Scripts in `scripts/` (model fetch); docs in `docs/` (e.g., `docs/Benchmark-Troubleshooting.md`).
- Assets: sample WAVs in repo root; models live under per-user app data, not in git.

## Build, Test, and Development Commands

- Configure: `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DEDDY_ENABLE_OPENVINO=ON`
- Build tools: `cmake --build build --config Release --target eddy parakeet_cli benchmark_librispeech hf_fetch_models`
- Run CLI (NPU): `build\examples\cpp\Release\parakeet_cli.exe "<path-to-wav>" --device NPU`
- Benchmark (NPU): `build\examples\cpp\Release\benchmark_librispeech.exe --max-files 50 --device NPU`
- Tests (optional): configure with `-DBUILD_TESTING=ON`, then `ctest --test-dir build`
- Whisper (optional): add `-DEDDY_ENABLE_WHISPER=ON` and set `OpenVINOGenAI_DIR` if not auto-discovered.
- Python: always use `uv` commands, never use `pip` directly.

## Coding Style & Naming Conventions

- C++20; 2-space indentation; braces on the same line.
- Types/classes: PascalCase. Functions/variables/files: snake_case (e.g., `parakeet_openvino.cpp`).
- Keep headers under `include/eddy/...` mirrored by sources under `src/...`.
- Prefer small, focused functions; avoid inline comments unless clarifying non-obvious logic.

## Testing Guidelines

- Place unit/integration tests under `tests/` (enable with `-DBUILD_TESTING=ON`); name files `<area>_test.cpp`.
- Manual checks: use `parakeet_cli` and `benchmark_librispeech` with short WAVs and limited file counts.
- Cover new logic; document any gaps in the PR.

## Commit & Pull Request Guidelines

- Commits: imperative subject with optional scope (e.g., `parakeet: fix encoder port selection`).
- Keep changes focused; include rationale and before/after behavior.
- PRs should include summary, reproduction/validation steps, logs or screenshots, target device (CPU/NPU), and linked issues.

## Agent-Specific Instructions (Parakeet/OpenVINO)

- After editing `src/models/parakeet/` or related headers, clear compiled caches but keep downloaded models (Windows: `%LOCALAPPDATA%\eddy\models\parakeet-v2`, preserve `files/`). Quick PowerShell:

  ```powershell
  $b = "$env:LOCALAPPDATA\eddy\models\parakeet-v2";
  if (Test-Path $b) {
    Get-ChildItem $b -File | Remove-Item -Force
    Get-ChildItem $b -Directory | ? { $_.Name -ne 'files' } | Remove-Item -Recurse -Force
  }
  ```

- Rebuild: `cmake --build build --config Release --target eddy parakeet_cli benchmark_librispeech hf_fetch_models`. First run after a cache clear will recompile models and may take minutes.

## Configuration & Models

- OpenVINO env: use `run_bench_npu.bat` to preload. If needed, set `OpenVINO_DIR` (e.g., `C:\Program Files (x86)\Intel\openvino_2025.0.0\runtime\cmake`).
- GenAI (Whisper): set `OpenVINOGenAI_DIR` when `EDDY_ENABLE_WHISPER=ON`.
- Download models: Models auto-download on first run via `hf_fetch_models`. Manual download: run `hf_fetch_models.exe` or visit <https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-ov> (downloads into `%LOCALAPPDATA%\eddy\models\parakeet-v2\files`).
- Runtime knobs: `EDDY_OV_PERF`, `EDDY_OV_NUM_REQUESTS`, `EDDY_OV_THREADS`, `EDDY_CONTEXT_FRAMES`, `EDDY_BOUNDARY_SEARCH_FRAMES`, `EDDY_DISABLE_HOLDBACK=1`, `EDDY_DEDUP_PREV_TOKENS` (default 15).
