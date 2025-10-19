# Repository Guidelines

## Project Structure & Module Organization

- Source: `src/` (core, backends, `models/parakeet/`, pipelines, streaming) and public headers in `include/`.
- Examples: `examples/cpp/` (`parakeet_cli.cpp`, `benchmark_librispeech.cpp`, optional `whisper_example.cpp`).
- Build system: CMake files at repository root and `examples/`.
- Scripts & docs: `scripts/` (model fetch), `docs/` (e.g., `docs/Benchmark-Troubleshooting.md`).
- Assets/models: sample WAVs in root; model cache lives under user cache, not in repo.

## Build, Test, and Development Commands

- Configure: `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DEDDY_ENABLE_OPENVINO=ON`
- Build libs + tools: `cmake --build build --config Release --target eddy parakeet_cli benchmark_librispeech`
- Run CLI (NPU): `build\examples\cpp\Release\parakeet_cli.exe "<path-to-wav>" --device NPU`
- Benchmark (NPU): `build\examples\cpp\Release\benchmark_librispeech.exe --max-files 50 --device NPU`
- Optional tests: If a `tests/` subtree is added, enable with `-DBUILD_TESTING=ON` and run `ctest --test-dir build`.
 - Optional Whisper (GenAI): add `-DEDDY_ENABLE_WHISPER=ON` and provide `OpenVINOGenAI_DIR` if not discoverable.

## Coding Style & Naming Conventions

- C++20, consistent with existing code. Use 2-space indentation, braces on same line.
- Types/classes: `PascalCase`; functions/variables/files: `snake_case` (e.g., `parakeet_openvino.cpp`).
- Keep headers under `include/eddy/...` mirrored by sources under `src/...`.
- Prefer small, focused functions; avoid inline comments unless clarifying non-obvious logic.

## Testing Guidelines

- Add targeted unit or integration tests under `tests/` when feasible. Name files `<area>_test.cpp`.
- For manual checks, use `parakeet_cli` and `benchmark_librispeech` with short WAVs and limited file counts.
- Aim to keep new logic covered; document any gaps in the PR.

## Commit & Pull Request Guidelines

- Commit subject: imperative mood with optional scope, e.g., `parakeet: fix encoder port selection`.
- Keep changes focused; include rationale and before/after behavior in the body.
- PRs should include: summary, reproduction/validation steps, screenshots or logs, target device (CPU/GPU/NPU), and linked issues.

## Agent-Specific Instructions (Parakeet/OpenVINO)

- After editing `src/models/parakeet/` or related headers, clear compiled caches but keep downloaded models.
  - Windows cache: `%LOCALAPPDATA%\eddy\cache\models\parakeet-v2` (preserve `files/`). Quick PowerShell:
    ``$b="$env:LOCALAPPDATA\eddy\cache\models\parakeet-v2"; if(Test-Path $b){ Get-ChildItem $b -File|Remove-Item -Force; Get-ChildItem $b -Directory|? Name -ne 'files'|Remove-Item -Recurse -Force }``
- Rebuild: `cmake --build build --config Release --target eddy parakeet_cli benchmark_librispeech`.
- First run after a cache clear will recompile models and may take minutes.

## Configuration & Models

- OpenVINO env: use `run_bench_npu.bat` to preload environment. If needed, `OpenVINO_DIR` default: `C:\Program Files (x86)\Intel\openvino_2025.0.0\runtime\cmake`.
- GenAI (Whisper) CMake hint: set `OpenVINOGenAI_DIR` to the GenAI CMake package path when `EDDY_ENABLE_WHISPER=ON`.
- Model download (recommended, no Python):
  - Windows PowerShell:
    `powershell -ExecutionPolicy Bypass -File scripts\setup_parakeet_models.ps1`
    Downloads `parakeet_melspectogram.(xml|bin)`, `parakeet_encoder.(xml|bin)`, `parakeet_decoder.(xml|bin)`, `parakeet_joint.(xml|bin)`, `parakeet_vocab.json` into `%LOCALAPPDATA%\eddy\cache\models\parakeet-v2\files`.
  - Cross‑platform: build and run `hf_fetch_models` with `--target` pointing to your cache dir.
- Runtime knobs: `EDDY_OV_PERF`, `EDDY_OV_NUM_REQUESTS`, `EDDY_OV_THREADS`, `EDDY_OV_PRECISION`, plus chunking/dedup controls `EDDY_CONTEXT_FRAMES`, `EDDY_BOUNDARY_SEARCH_FRAMES`, `EDDY_DISABLE_HOLDBACK=1`.
