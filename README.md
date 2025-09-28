# Eddy SDK (Work In Progress)

Eddy is a C++ SDK targeting multi-vendor edge NPUs for speech and multimodal inference. The current milestone focuses on enabling OpenVINO 2025.x support for the Parakeet-TDT speech model family.

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
