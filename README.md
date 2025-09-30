# eddy (Work In Progress)

eddy is a C++ inference library designed for native runtimes and multi-vendor edge NPUs, exposing a consistent C++ API plus language bindings for app developers (C#; more to follow). The current milestone focuses on the OpenVINO 2025.x backend for the Parakeet-TDT speech model family while we bring additional runtimes online.

## Repository Layout
- `include/` – public headers for the runtime, backend abstractions, and model bridges.
- `src/` – backend/runtime implementations and model-specific glue code.
- `docs/` – design notes and usage guides.

## Building
```
cmake -S . -B build -DOpenVINO_DIR=/opt/intel/openvino/runtime/cmake
cmake --build build
```

The build emits the static target `eddy` with OpenVINO linked in when `EDDY_ENABLE_OPENVINO=ON` (default).

## Parakeet OpenVINO Prototype
Refer to `docs/parakeet_openvino.md` for instructions on pulling the exported model from Hugging Face and running a smoke test through the new `OpenVINOParakeet` wrapper.

## Roadmap Snapshot
- Flesh out Parakeet preprocessing (feature pipeline, tokenizer, decoder).
- Add telemetry and zero-copy buffers per backend.
- Introduce unit and integration tests (GoogleTest) with small audio fixtures.
- Extend the backend layer to Qualcomm QNN and AMD MIGraphX once the OpenVINO path is validated.
