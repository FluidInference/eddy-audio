# Debug Notes

## Silent Execution Issue (Resolved)

**Problem:**
The Debug build (`build/examples/cpp/Debug/parakeet_cli.exe`) ran but produced no output, not even help text or error messages.

**Root Cause:**
- Debug builds require debug runtime DLLs: `vcruntime140d.dll`, `msvcp140d.dll`, etc.
- These DLLs only exist if Visual Studio is installed
- Without them, Windows fails to load the executable before `main()` even starts
- The failure was silent - no error dialog, no output

**Solution:**
Use Release build instead:
```bash
cmake --build build --config Release --target parakeet_cli
```

Release builds use standard runtime DLLs (`vcruntime140.dll`, `msvcp140.dll`) that are already installed on most Windows systems.

**Lesson:**
When distributing or testing C++ applications on Windows, always use Release builds unless actively debugging. Debug builds have dependencies that aren't present on typical user systems.

## Current Working Configuration

- **Executable:** `build/examples/cpp/Release/parakeet_cli.exe`
- **Runner script:** `run_parakeet.bat` (uses Release build)
- **OpenVINO DLLs:** Copied to `build/examples/cpp/Release/` directory
- **Model Files:** Loaded from cache first, fallback to local
  - **Cache location:** `C:\Users\<user>\AppData\Local\eddy\cache\models\parakeet-v2\files\`
  - **Local fallback:** `models/parakeet/`
- **Compiled Model Cache:** `C:\Users\<user>\AppData\Local\eddy\cache\models\parakeet-v2\` (OpenVINO optimized binaries)

## Model Loading Strategy

The CLI now uses a cache-first approach:
1. Checks `%LOCALAPPDATA%\eddy\cache\models\parakeet-v2\files\` for model files
2. Falls back to `models/parakeet/` if not found in cache
3. OpenVINO compiled model cache stored in `%LOCALAPPDATA%\eddy\cache\models\parakeet-v2\`

To install models to cache:
```bash
mkdir -p ~/AppData/Local/eddy/cache/models/parakeet-v2/files
cp models/parakeet/* ~/AppData/Local/eddy/cache/models/parakeet-v2/files/
```

## Performance

Tested on CPU (Intel):
- 10s audio: 7.8x real-time (1275ms processing)
- 15s audio: 5.6x real-time (2664ms processing)
- Chunking works correctly for long audio (>10 seconds)
