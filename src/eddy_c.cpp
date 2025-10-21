// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/eddy_c.h"

#if defined(EDDY_WITH_OPENVINO_GENAI)
#include "eddy/pipelines/whisper_pipeline.hpp"
#endif

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/core/app_dir.hpp"
#include "eddy/models/parakeet-v2/parakeet.hpp"
#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "eddy/utils/ensure_models.hpp"
#include "eddy/utils/audio_utils.hpp"

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
#if !defined(EDDY_WITH_OPENVINO_GENAI)
    (void)config; (void)error_message;
    return nullptr;
#else
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
#endif
}

void eddy_whisper_destroy(EddyWhisperPipeline pipeline) {
#if defined(EDDY_WITH_OPENVINO_GENAI)
    if (!pipeline) return;
    delete static_cast<eddy::WhisperPipeline*>(pipeline);
#else
    (void)pipeline;
#endif
}

EddyError eddy_whisper_transcribe_file(
    EddyWhisperPipeline pipeline,
    const char* wav_path,
    EddyWhisperResult* result,
    char** error_message
) {
#if !defined(EDDY_WITH_OPENVINO_GENAI)
    (void)pipeline; (void)wav_path; (void)result; (void)error_message;
    return EDDY_ERROR_INVALID_ARGUMENT;
#else
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
#endif
}

EddyError eddy_whisper_transcribe_buffer(
    EddyWhisperPipeline pipeline,
    const float* pcm,
    size_t length,
    int sample_rate,
    EddyWhisperResult* result,
    char** error_message
) {
#if !defined(EDDY_WITH_OPENVINO_GENAI)
    (void)pipeline; (void)pcm; (void)length; (void)sample_rate; (void)result; (void)error_message;
    return EDDY_ERROR_INVALID_ARGUMENT;
#else
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
#endif
}

void eddy_whisper_set_language(
    EddyWhisperPipeline pipeline,
    const char* language
) {
#if defined(EDDY_WITH_OPENVINO_GENAI)
    if (!pipeline || !language) return;
    auto* pipe = static_cast<eddy::WhisperPipeline*>(pipeline);
    pipe->set_language(language);
#else
    (void)pipeline; (void)language;
#endif
}

void eddy_whisper_set_task(
    EddyWhisperPipeline pipeline,
    const char* task
) {
#if defined(EDDY_WITH_OPENVINO_GENAI)
    if (!pipeline || !task) return;
    auto* pipe = static_cast<eddy::WhisperPipeline*>(pipeline);
    pipe->set_task(task);
#else
    (void)pipeline; (void)task;
#endif
}

const char* eddy_whisper_get_language(EddyWhisperPipeline pipeline) {
#if defined(EDDY_WITH_OPENVINO_GENAI)
    if (!pipeline) return "";
    auto* pipe = static_cast<eddy::WhisperPipeline*>(pipeline);
    // Note: returning a temporary string's c_str() is dangerous
    // In practice, the caller should not rely on this persisting
    static thread_local std::string lang_storage;
    lang_storage = pipe->get_language();
    return lang_storage.c_str();
#else
    (void)pipeline;
    return "";
#endif
}

// -----------------------------
// Parakeet C API
// -----------------------------

typedef struct {
    std::shared_ptr<eddy::parakeet::IParakeetModel> model;
} CParakeet;

EDDY_API void eddy_parakeet_free_result(EddyParakeetResult* result) {
    if (!result) return;
    if (result->text) { delete[] result->text; result->text = nullptr; }
    if (result->token_ids) { delete[] result->token_ids; result->token_ids = nullptr; }
    result->num_tokens = 0;
}

EDDY_API EddyParakeetModel eddy_parakeet_create(EddyParakeetConfig config, char** error_message) {
    try {
        std::string device = config.device ? config.device : "CPU";

        auto backend = std::make_shared<eddy::OpenVINOBackend>(
            eddy::OpenVINOOptions{ .device = device, .cache_dir = eddy::get_model_dir("parakeet-v2").string() }
        );

        // Resolve model directory: prefer explicit, else cache and ensure availability
        std::filesystem::path model_dir;
        if (config.model_dir && std::string(config.model_dir).size() > 0) {
            model_dir = config.model_dir;
        } else {
            model_dir = eddy::get_model_assets_dir("parakeet-v2");
            std::string err;
            (void)eddy::parakeet::check_models_available(model_dir, &err);
#if defined(_WIN32)
            if (!std::filesystem::exists(model_dir)) {
                auto legacy = eddy::get_app_data_dir() / "cache" / "models" / "parakeet-v2" / "files";
                if (std::filesystem::exists(legacy)) model_dir = legacy;
            }
#endif
        }

        eddy::parakeet::ModelPaths paths{
            .preprocessor = {.path = (model_dir / "parakeet_melspectogram.xml").string()},
            .encoder = {.path = (model_dir / "parakeet_encoder.xml").string()},
            .decoder = {.path = (model_dir / "parakeet_decoder.xml").string()},
            .joint = {.path = (model_dir / "parakeet_joint.xml").string()},
            .tokenizer_json = (model_dir / "parakeet_vocab.json").string()
        };

        eddy::parakeet::RuntimeConfig cfg{
            .device = device,
            .blank_token_id = config.blank_token_id > 0 ? config.blank_token_id : 1024,
            .duration_bins = {0,1,2,3,4}
        };

        auto model = eddy::parakeet::make_openvino_parakeet(backend, paths, cfg);
        auto* handle = new CParakeet{model};
        return static_cast<EddyParakeetModel>(handle);
    } catch (const std::exception& e) {
        if (error_message) *error_message = capture_exception(e);
        return nullptr;
    } catch (...) {
        if (error_message) *error_message = copy_string("[Eddy Error] Unknown exception in create");
        return nullptr;
    }
}

EDDY_API void eddy_parakeet_destroy(EddyParakeetModel handle) {
    if (!handle) return;
    delete static_cast<CParakeet*>(handle);
}

static EddyError parakeet_infer_common(CParakeet* h, const float* pcm, size_t length, int sample_rate, EddyParakeetResult* out, char** err) {
    if (!h || !pcm || !out) {
        if (err) *err = copy_string("[Eddy Error] Invalid argument: null pointer");
        return EDDY_ERROR_INVALID_ARGUMENT;
    }
    if (sample_rate != 16000) {
        if (err) *err = copy_string("[Eddy Error] Parakeet expects 16kHz mono audio");
        return EDDY_ERROR_INVALID_ARGUMENT;
    }
    try {
        eddy::parakeet::AudioSegment seg;
        seg.sample_rate = 16000;
        seg.pcm.assign(pcm, pcm + length);
        eddy::parakeet::SegmentOptions opt;
        auto res = h->model->infer(seg, opt);

        out->text = copy_string(res.text);
        out->confidence = res.overall_confidence;
        out->latency_ms = res.latency_ms;
        out->num_tokens = res.token_ids.size();
        if (out->num_tokens > 0) {
            out->token_ids = new int[out->num_tokens];
            for (size_t i = 0; i < out->num_tokens; ++i) out->token_ids[i] = res.token_ids[i];
        } else {
            out->token_ids = nullptr;
        }
        return EDDY_OK;
    } catch (const std::exception& e) {
        if (err) *err = capture_exception(e);
        return EDDY_ERROR_INFERENCE_FAILED;
    } catch (...) {
        if (err) *err = copy_string("[Eddy Error] Unknown exception during parakeet inference");
        return EDDY_ERROR_UNKNOWN;
    }
}

EDDY_API EddyError eddy_parakeet_infer_file(EddyParakeetModel handle, const char* wav_path, EddyParakeetResult* out, char** err) {
    if (!handle || !wav_path || !out) {
        if (err) *err = copy_string("[Eddy Error] Invalid argument: null pointer");
        return EDDY_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto* h = static_cast<CParakeet*>(handle);
        auto pcm = eddy::audio::read_wav(wav_path);
        return parakeet_infer_common(h, pcm.data(), pcm.size(), 16000, out, err);
    } catch (const std::exception& e) {
        if (err) *err = capture_exception(e);
        return EDDY_ERROR_FILE_NOT_FOUND;
    } catch (...) {
        if (err) *err = copy_string("[Eddy Error] Unknown exception in infer_file");
        return EDDY_ERROR_UNKNOWN;
    }
}

EDDY_API EddyError eddy_parakeet_infer_buffer(EddyParakeetModel handle, const float* pcm, size_t length, int sample_rate, EddyParakeetResult* out, char** err) {
    auto* h = static_cast<CParakeet*>(handle);
    return parakeet_infer_common(h, pcm, length, sample_rate, out, err);
}

EDDY_API char* eddy_parakeet_decode_tokens(EddyParakeetModel handle, const int* token_ids, size_t count) {
    if (!handle || !token_ids || count == 0) return copy_string("");
    auto* h = static_cast<CParakeet*>(handle);
    std::vector<int> ids(token_ids, token_ids + count);
    auto txt = h->model->decode_tokens(ids);
    return copy_string(txt);
}

} // extern "C"
