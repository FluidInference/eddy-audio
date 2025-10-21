# Run Guide and Troubleshooting (NPU/CPU)

This doc captures the practical gotchas we hit while running the C++ benchmark and how to avoid them next time.

## Quick Start (NPU)

- Recommended wrapper (ensures OpenVINO env is loaded):
  - `./run_bench_npu.bat --max-files 100 --device NPU`

- One‑liner (same effect):
  - `cmd /c ""C:\\Program Files (x86)\\Intel\\openvino_2025.0.0\\setupvars.bat" && "C:\\Users\\brand\\code\\eddy\\build\\examples\\cpp\\Release\\benchmark_librispeech.exe" --max-files 100 --device NPU"`

### Quick Start (CPU)

- No OpenVINO env needed, just run:
  - `build\examples\cpp\Release\benchmark_librispeech.exe --max-files 5 --device CPU`

### Quick Start (Python Benchmark)

- Install Python deps:
  - `cd benchmarks && uv sync`
- Run benchmark (automatically rebuilds C++ if needed):
  - `cd benchmarks && uv run benchmark.py --max-files 25 --device CPU`
- Note: The Python benchmark uses the native C++ library via the parakeet_cli executable, not the C ABI

### PowerShell Quoting (Important)

- Quoted paths are strings, not commands. Use the call operator `&` when quoting:
  - `& "C:\\Users\\brand\\code\\eddy\\build\\examples\\cpp\\Release\\benchmark_librispeech.exe" --max-files 100 --device NPU`
- If the path has no spaces, you can omit quotes:
  - `build\examples\cpp\Release\benchmark_librispeech.exe --max-files 100 --device NPU`

### OpenVINO Environment

- Detected install:
  - `setupvars.bat`: `C:\\Program Files (x86)\\Intel\\openvino_2025.0.0\\setupvars.bat`
  - `OpenVINO_DIR`: `C:\\Program Files (x86)\\Intel\\openvino_2025.0.0\\runtime\\cmake`
- NPU requires `setupvars.bat` each run to set PATH/plugins. Use the wrapper or the one‑liner.
- Persist `OpenVINO_DIR` (new shells):
  - `setx OpenVINO_DIR "C:\\Program Files (x86)\\Intel\\openvino_2025.0.0\\runtime\\cmake"`

### Fresh Rebuild

1) Clean everything (build, local models, cached models):
   - Delete `build\`
   - Delete `models\parakeet\`
   - Delete `%LOCALAPPDATA%\eddy\models\parakeet-v2\`
2) Configure + build:
   - `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`
   - `cmake --build build --config Release --target eddy parakeet_cli benchmark_librispeech hf_fetch_models`
3) Fetch models:
   - Models auto-download on first run, or run `hf_fetch_models.exe` manually

### Clear Compiled Cache (Keep Models)

- See `AGENTS.md` for exact commands. In short, remove files/directories in `%LOCALAPPDATA%\eddy\models\parakeet-v2` except `files\`.

### JSON Output, Filters, and Chunk Logs

- Save only “obvious errors” (e.g., WER ≥ 10%) to JSON:
  - `--min-wer 10`
- Multi‑chunk details (only when `chunk_count > 1`):
  - `chunks[]` entries include: `index`, `offset_frames`, `size_frames`, `is_last`, `tokens_predicted`, `tokens_appended`, `skip_prefix`, `holdback`, and `text` (the appended slice’s decoded text).

### Useful Debug Toggles

- `EDDY_DEBUG=1` — enable chunk/dedup diagnostics in stderr (suppressed by default).
- `EDDY_CONTEXT_FRAMES` — fixed overlap geometry per side (default 20 → 40‑frame overlap).
- `EDDY_DISABLE_HOLDBACK=1` — disables right‑context holdback (pure FluidAudio‑style dedup only).

### Common Errors and Fixes

- PowerShell: `Unexpected token 'max-files'`
  - You quoted the exe path without `&`. Use `& "...exe" --args` or run without quotes.

- No output on NPU
  - Not running under OpenVINO env. Use `setupvars.bat` or `run_bench_npu.bat`.

- Type mismatch (i32 vs i64) in length tensors
  - Fixed in code: preprocessor/encoder/decoder length tensors match the model ports’ element types.

- Seam‑echo duplicates ("... X Y Z X Y Z")
  - Replaced with FluidAudio‑style overlap dedup: punctuation guard → suffix‑prefix exact match → boundary‑window partial search.

### Interpreting Errors in all.json

- Short utterances (≤6 words) inflate WER (1 edit can be 20–50%). Check CER too.
- Proper‑noun/spacing variants dominate spikes:
  - Examples: `McCardle` vs `Mac Ardle`, `Keogh` vs `Keough`, `FrancisXavier` vs `Francis Xavier`, `Dedalos` vs `Dedalus`.
  - Consider normalization mappings in the benchmark (English variants JSON) for reporting fairness.

---

### Agent Notes (for automated coding agents)

- Always run NPU benchmarks via `run_bench_npu.bat` or through `setupvars.bat`.
- On code changes under `src/models/parakeet-v2/`:
  - Clear compiled cache (keep `files/`) as per `AGENTS.md`.
  - Rebuild: `cmake --build build --config Release --target eddy parakeet_cli benchmark_librispeech`.
  - If OpenVINO not found: set `OpenVINO_DIR` to `C:\\Program Files (x86)\\Intel\\openvino_2025.0.0\\runtime\\cmake`.
  - Use `--min-wer` to save focused JSON for regression triage.
