// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

/**
 * @file eddy_c.h
 * @brief C API for Eddy SDK - enables language bindings (C#, etc.)
 */

#ifndef EDDY_C_H
#define EDDY_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

// Platform-specific exports
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

/**
 * @brief Configuration for Whisper pipeline
 */
typedef struct {
    const char* model_path;
    const char* device;           // "NPU", "CPU", "GPU", "AUTO"
    const char* language;         // "en", "zh", "auto", etc.
    const char* task;             // "transcribe" or "translate"
    bool return_timestamps;
    bool enable_cache;
    const char* cache_dir;        // Can be NULL for default
} EddyWhisperConfig;

/**
 * @brief A chunk of transcribed text with timestamps
 */
typedef struct {
    float start_ts;
    float end_ts;
    char* text;  // Must be freed by caller using eddy_free_string
} EddyWhisperChunk;

/**
 * @brief Result from Whisper transcription
 */
typedef struct {
    char* text;                   // Full transcribed text, must be freed by caller
    EddyWhisperChunk* chunks;     // Array of chunks, must be freed by caller
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

// Utility

/**
 * @brief Get the Eddy SDK version string
 * @return Version string (e.g., "0.1.0")
 */
EDDY_API const char* eddy_version(void);

#ifdef __cplusplus
}
#endif

#endif // EDDY_C_H
