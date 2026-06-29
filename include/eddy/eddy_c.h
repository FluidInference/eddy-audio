// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

/**
 * @file eddy_c.h
 * @brief C API for Eddy SDK - enables language bindings (C#, etc.)
 *
 * ## Thread Safety
 * - All functions are thread-safe unless otherwise noted
 * - Pipeline/model handles can be used from multiple threads concurrently
 * - Exception: eddy_whisper_set_language/set_task are NOT thread-safe with concurrent inference
 *
 * ## Error Handling
 * - Functions returning EddyError: On error, output parameters (result) are zeroed/not modified
 * - Functions returning handles: Return NULL on error, check error_message for details
 * - error_message parameters: Always call eddy_free_string() even if function succeeds
 *
 * ## Memory Management
 * - All strings/results allocated by Eddy must be freed by caller
 * - Use eddy_whisper_free_result() to free EddyWhisperResult (frees all nested strings)
 * - Use eddy_parakeet_free_result() to free EddyParakeetResult
 * - Use eddy_free_string() for standalone strings only
 * - Do NOT call eddy_free_string() on strings inside results freed by eddy_*_free_result()
 *
 * ## Null Safety
 * - Handle parameters (EddyWhisperPipeline, EddyParakeetModel) must NOT be NULL
 * - Passing NULL handles results in EDDY_ERROR_INVALID_ARGUMENT or undefined behavior
 * - Output parameters (result, error_message) can be NULL if you don't need them
 */

#ifndef EDDY_C_H
#define EDDY_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

// Platform-specific exports
// Note: Define EDDY_BUILD_SHARED only when building the Eddy library itself, not when using it
#ifdef _WIN32
  #ifdef EDDY_BUILD_SHARED
    #define EDDY_API __declspec(dllexport)
  #else
    #define EDDY_API __declspec(dllimport)
  #endif
#else
  #define EDDY_API __attribute__((visibility("default")))
#endif

// Opaque handle types
typedef void* EddyWhisperPipeline;
typedef void* EddyParakeetModel;

/**
 * @brief Configuration for Whisper pipeline
 */
typedef struct {
    const char* model_path;
    const char* device;           // "NPU", "CPU", "AUTO"
    const char* language;         // "en", "zh", "auto", etc.
    const char* task;             // "transcribe" or "translate"
    bool return_timestamps;
    bool enable_cache;
    const char* cache_dir;        // Can be NULL for default
} EddyWhisperConfig;

/**
 * @brief A chunk of transcribed text with timestamps
 *
 * Note: Individual chunk.text pointers are managed by EddyWhisperResult
 * Do NOT call eddy_free_string() on chunk.text - use eddy_whisper_free_result() instead
 */
typedef struct {
    float start_ts;
    float end_ts;
    char* text;  // Owned by EddyWhisperResult, freed by eddy_whisper_free_result()
} EddyWhisperChunk;

/**
 * @brief Result from Whisper transcription
 *
 * Resource ownership:
 * - Call eddy_whisper_free_result() to free the entire result
 * - This frees result.text, result.chunks array, and all chunk.text strings
 * - Do NOT manually free individual strings - eddy_whisper_free_result() handles everything
 */
typedef struct {
    char* text;                   // Full transcribed text
    EddyWhisperChunk* chunks;     // Array of chunks
    size_t num_chunks;
    float confidence;
    double inference_duration_ms;
} EddyWhisperResult;

/**
 * @brief Error codes
 */
typedef enum {
    EDDY_OK = 0,
    EDDY_ERROR_INVALID_ARGUMENT = 1,
    EDDY_ERROR_MODEL_LOAD_FAILED = 2,
    EDDY_ERROR_INFERENCE_FAILED = 3,
    EDDY_ERROR_FILE_NOT_FOUND = 4,
    EDDY_ERROR_UNKNOWN = 99
} EddyError;

// Whisper Pipeline API

/**
 * @brief Create a Whisper pipeline
 * @param config Configuration parameters
 * @param error_message Out parameter for error message (can be NULL). Must be freed with eddy_free_string.
 * @return Pipeline handle or NULL on failure
 */
EDDY_API EddyWhisperPipeline eddy_whisper_create(
    EddyWhisperConfig config,
    char** error_message
);

/**
 * @brief Destroy a Whisper pipeline
 * @param pipeline Pipeline handle
 */
EDDY_API void eddy_whisper_destroy(EddyWhisperPipeline pipeline);

/**
 * @brief Transcribe audio from a WAV file
 * @param pipeline Pipeline handle
 * @param wav_path Path to WAV file
 * @param result Out parameter for result. Must be freed with eddy_whisper_free_result.
 * @param error_message Out parameter for error message (can be NULL). Must be freed with eddy_free_string.
 * @return EDDY_OK on success, error code otherwise
 */
EDDY_API EddyError eddy_whisper_transcribe_file(
    EddyWhisperPipeline pipeline,
    const char* wav_path,
    EddyWhisperResult* result,
    char** error_message
);

/**
 * @brief Transcribe audio from raw PCM buffer
 * @param pipeline Pipeline handle
 * @param pcm Float32 PCM samples (normalized to [-1, 1])
 * @param length Number of samples
 * @param sample_rate Sample rate in Hz (must be 16000)
 * @param result Out parameter for result. Must be freed with eddy_whisper_free_result.
 * @param error_message Out parameter for error message (can be NULL). Must be freed with eddy_free_string.
 * @return EDDY_OK on success, error code otherwise
 */
EDDY_API EddyError eddy_whisper_transcribe_buffer(
    EddyWhisperPipeline pipeline,
    const float* pcm,
    size_t length,
    int sample_rate,
    EddyWhisperResult* result,
    char** error_message
);

/**
 * @brief Set the language for transcription
 * @param pipeline Pipeline handle
 * @param language Language code (e.g., "en", "zh") or "auto"
 */
EDDY_API void eddy_whisper_set_language(
    EddyWhisperPipeline pipeline,
    const char* language
);

/**
 * @brief Set the task (transcribe or translate)
 * @param pipeline Pipeline handle
 * @param task "transcribe" or "translate"
 */
EDDY_API void eddy_whisper_set_task(
    EddyWhisperPipeline pipeline,
    const char* task
);

/**
 * @brief Get the current language setting
 * @param pipeline Pipeline handle
 * @return Language string (do not free, valid until next call to eddy_whisper_set_language)
 */
EDDY_API const char* eddy_whisper_get_language(EddyWhisperPipeline pipeline);

// Memory management

/**
 * @brief Free a WhisperResult
 * @param result Result to free
 */
EDDY_API void eddy_whisper_free_result(EddyWhisperResult* result);

/**
 * @brief Free a string allocated by Eddy
 * @param str String to free
 */
EDDY_API void eddy_free_string(char* str);

// -----------------------------
// Parakeet (OpenVINO) C API
// -----------------------------

typedef struct {
    const char* device;      // "CPU", "NPU", or "AUTO"
    const char* model_dir;   // Directory containing parakeet_*.xml/bin/json; NULL to use Eddy cache
    int blank_token_id;      // Typically 1024
} EddyParakeetConfig;

typedef struct {
    char* text;          // must be freed with eddy_parakeet_free_result
    int* token_ids;      // must be freed with eddy_parakeet_free_result
    size_t num_tokens;
    float confidence;
    double latency_ms;
} EddyParakeetResult;

EDDY_API EddyParakeetModel eddy_parakeet_create(EddyParakeetConfig config, char** error_message);
EDDY_API void eddy_parakeet_destroy(EddyParakeetModel model);
EDDY_API EddyError eddy_parakeet_infer_file(EddyParakeetModel model, const char* wav_path, EddyParakeetResult* result, char** error_message);
EDDY_API EddyError eddy_parakeet_infer_buffer(EddyParakeetModel model, const float* pcm, size_t length, int sample_rate, EddyParakeetResult* result, char** error_message);
EDDY_API char* eddy_parakeet_decode_tokens(EddyParakeetModel model, const int* token_ids, size_t count);
EDDY_API void eddy_parakeet_free_result(EddyParakeetResult* result);

// -----------------------------
// Nemotron streaming (OpenVINO) C API
// -----------------------------

// Opaque handle for a Nemotron streaming model.
typedef void* EddyNemotronModel;

typedef struct {
    const char* device;     // "CPU", "NPU", or "AUTO" (default "CPU")
    const char* model_dir;  // Dir with nemotron_*.xml/bin + metadata.json.
                            // NULL or "cache" => Eddy cache for "nemotron-streaming".
    const char* language;   // "en-US", "zh-CN", ..., or "auto" (default "auto")
} EddyNemotronConfig;

typedef struct {
    char* text;                // full transcript (lang-tag tokens stripped); free with eddy_nemotron_free_result
    char* detected_language;   // first <xx-XX> tag emitted, or "" ; freed with the result
    int* token_ids;            // raw emitted token ids (pre-strip); must be freed with eddy_nemotron_free_result
    size_t num_tokens;
    int prompt_id_used;        // integer prompt id selected for conditioning
    double latency_ms;
} EddyNemotronResult;

/**
 * @brief Create a Nemotron streaming model.
 * @param config Device / model_dir / language. Language conditions decoding and
 *               is fixed at creation; recreate the handle to change it.
 * @param error_message Out param for error (can be NULL). Free with eddy_free_string.
 * @return Model handle or NULL on failure.
 */
EDDY_API EddyNemotronModel eddy_nemotron_create(EddyNemotronConfig config, char** error_message);

/** @brief Destroy a Nemotron model handle. */
EDDY_API void eddy_nemotron_destroy(EddyNemotronModel model);

/**
 * @brief Transcribe a 16 kHz mono WAV file.
 * @param result Out param; free with eddy_nemotron_free_result.
 */
EDDY_API EddyError eddy_nemotron_infer_file(EddyNemotronModel model, const char* wav_path, EddyNemotronResult* result, char** error_message);

/**
 * @brief Transcribe a raw float32 PCM buffer (16 kHz mono, normalized [-1, 1]).
 * @param result Out param; free with eddy_nemotron_free_result.
 */
EDDY_API EddyError eddy_nemotron_infer_buffer(EddyNemotronModel model, const float* pcm, size_t length, int sample_rate, EddyNemotronResult* result, char** error_message);

/** @brief Free an EddyNemotronResult (text, detected_language, token_ids). */
EDDY_API void eddy_nemotron_free_result(EddyNemotronResult* result);

// Utility

/**
 * @brief Get the Eddy SDK version string
 * @return Version string (e.g., "0.1.0")
 */
EDDY_API const char* eddy_version(void);

/**
 * @brief Progress callback for model downloads
 * @param filename Current file being downloaded
 * @param current_file Current file index (1-based)
 * @param total_files Total number of files to download
 * @param user_data User-provided data pointer
 */
typedef void (*EddyDownloadProgressCallback)(const char* filename, int current_file, int total_files, void* user_data);

/**
 * @brief Download Parakeet model files from HuggingFace
 *
 * Downloads required model files to the specified directory. Skips files that already exist.
 *
 * @param model_name Model name (e.g., "parakeet-v2")
 * @param target_dir Target directory path (will be created if needed)
 * @param progress_callback Optional progress callback (can be NULL)
 * @param user_data User data passed to progress callback
 * @param error_message Optional error message output (call eddy_free_string() to free)
 * @return EDDY_OK on success, error code otherwise
 *
 * @note Requires curl to be available in PATH
 */
EDDY_API EddyError eddy_download_parakeet_models(
    const char* model_name,
    const char* target_dir,
    EddyDownloadProgressCallback progress_callback,
    void* user_data,
    char** error_message);

#ifdef __cplusplus
}
#endif

#endif // EDDY_C_H
