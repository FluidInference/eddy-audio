// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/eddy_c.h"
#include "eddy/pipelines/whisper_pipeline.hpp"

#include <cstring>
#include <string>
#include <exception>
#include <memory>

// Helper to allocate and copy a C string
static char* copy_string(const std::string& str) {
    char* result = new char[str.length() + 1];
    std::strcpy(result, str.c_str());
    return result;
}

// Helper to capture exception message
static char* capture_exception(const std::exception& e) {
    return copy_string(std::string("[Eddy Error] ") + e.what());
}

extern "C" {

const char* eddy_version(void) {
    return "0.1.0";
}

void eddy_free_string(char* str) {
    delete[] str;
}

void eddy_whisper_free_result(EddyWhisperResult* result) {
    if (!result) return;

    if (result->text) {
        delete[] result->text;
        result->text = nullptr;
    }

    if (result->chunks) {
        for (size_t i = 0; i < result->num_chunks; i++) {
            if (result->chunks[i].text) {
                delete[] result->chunks[i].text;
            }
        }
        delete[] result->chunks;
        result->chunks = nullptr;
    }

    result->num_chunks = 0;
}

EddyWhisperPipeline eddy_whisper_create(
    EddyWhisperConfig config,
    char** error_message
) {
    try {
        eddy::WhisperConfig cpp_config;
        cpp_config.model_path = config.model_path ? config.model_path : "";
        cpp_config.device = config.device ? config.device : "NPU";
        cpp_config.language = config.language ? config.language : "en";
        cpp_config.task = config.task ? config.task : "transcribe";
        cpp_config.return_timestamps = config.return_timestamps;
        cpp_config.enable_cache = config.enable_cache;
        if (config.cache_dir) {
            cpp_config.cache_dir = config.cache_dir;
        }

        auto* pipeline = new eddy::WhisperPipeline(cpp_config);
        return static_cast<EddyWhisperPipeline>(pipeline);

    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = capture_exception(e);
        }
        return nullptr;
    } catch (...) {
        if (error_message) {
            *error_message = copy_string("[Eddy Error] Unknown exception occurred");
        }
        return nullptr;
    }
}

void eddy_whisper_destroy(EddyWhisperPipeline pipeline) {
    if (!pipeline) return;
    delete static_cast<eddy::WhisperPipeline*>(pipeline);
}

EddyError eddy_whisper_transcribe_file(
    EddyWhisperPipeline pipeline,
    const char* wav_path,
    EddyWhisperResult* result,
    char** error_message
) {
    if (!pipeline || !wav_path || !result) {
        if (error_message) {
            *error_message = copy_string("[Eddy Error] Invalid argument: null pointer");
        }
        return EDDY_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* pipe = static_cast<eddy::WhisperPipeline*>(pipeline);
        eddy::WhisperResult cpp_result = pipe->transcribe(wav_path);

        // Convert result
        result->text = copy_string(cpp_result.text);
        result->confidence = cpp_result.confidence;
        result->inference_duration_ms = cpp_result.inference_duration_ms;

        // Convert chunks
        result->num_chunks = cpp_result.chunks.size();
        if (result->num_chunks > 0) {
            result->chunks = new EddyWhisperChunk[result->num_chunks];
            for (size_t i = 0; i < result->num_chunks; i++) {
                result->chunks[i].start_ts = cpp_result.chunks[i].start_ts;
                result->chunks[i].end_ts = cpp_result.chunks[i].end_ts;
                result->chunks[i].text = copy_string(cpp_result.chunks[i].text);
            }
        } else {
            result->chunks = nullptr;
        }

        return EDDY_OK;

    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = capture_exception(e);
        }
        return EDDY_ERROR_INFERENCE_FAILED;
    } catch (...) {
        if (error_message) {
            *error_message = copy_string("[Eddy Error] Unknown exception during transcription");
        }
        return EDDY_ERROR_UNKNOWN;
    }
}

EddyError eddy_whisper_transcribe_buffer(
    EddyWhisperPipeline pipeline,
    const float* pcm,
    size_t length,
    int sample_rate,
    EddyWhisperResult* result,
    char** error_message
) {
    if (!pipeline || !pcm || !result) {
        if (error_message) {
            *error_message = copy_string("[Eddy Error] Invalid argument: null pointer");
        }
        return EDDY_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* pipe = static_cast<eddy::WhisperPipeline*>(pipeline);
        eddy::WhisperResult cpp_result = pipe->transcribe(pcm, length, sample_rate);

        // Convert result
        result->text = copy_string(cpp_result.text);
        result->confidence = cpp_result.confidence;
        result->inference_duration_ms = cpp_result.inference_duration_ms;

        // Convert chunks
        result->num_chunks = cpp_result.chunks.size();
        if (result->num_chunks > 0) {
            result->chunks = new EddyWhisperChunk[result->num_chunks];
            for (size_t i = 0; i < result->num_chunks; i++) {
                result->chunks[i].start_ts = cpp_result.chunks[i].start_ts;
                result->chunks[i].end_ts = cpp_result.chunks[i].end_ts;
                result->chunks[i].text = copy_string(cpp_result.chunks[i].text);
            }
        } else {
            result->chunks = nullptr;
        }

        return EDDY_OK;

    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = capture_exception(e);
        }
        return EDDY_ERROR_INFERENCE_FAILED;
    } catch (...) {
        if (error_message) {
            *error_message = copy_string("[Eddy Error] Unknown exception during transcription");
        }
        return EDDY_ERROR_UNKNOWN;
    }
}

void eddy_whisper_set_language(
    EddyWhisperPipeline pipeline,
    const char* language
) {
    if (!pipeline || !language) return;
    auto* pipe = static_cast<eddy::WhisperPipeline*>(pipeline);
    pipe->set_language(language);
}

void eddy_whisper_set_task(
    EddyWhisperPipeline pipeline,
    const char* task
) {
    if (!pipeline || !task) return;
    auto* pipe = static_cast<eddy::WhisperPipeline*>(pipeline);
    pipe->set_task(task);
}

const char* eddy_whisper_get_language(EddyWhisperPipeline pipeline) {
    if (!pipeline) return "";
    auto* pipe = static_cast<eddy::WhisperPipeline*>(pipeline);
    // Note: returning a temporary string's c_str() is dangerous
    // In practice, the caller should not rely on this persisting
    static thread_local std::string lang_storage;
    lang_storage = pipe->get_language();
    return lang_storage.c_str();
}

} // extern "C"