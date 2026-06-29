// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/eddy_c.h"

#if defined(EDDY_WITH_OPENVINO_GENAI)
#include "eddy/pipelines/whisper_pipeline.hpp"
#endif

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/core/app_dir.hpp"
#include "eddy/core/model_configs.hpp"
#include "eddy/models/parakeet-v2/parakeet.hpp"
#include "eddy/models/parakeet-v2/parakeet_openvino.hpp"
#include "eddy/models/nemotron/nemotron.hpp"
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

EddyError eddy_download_parakeet_models(
    const char* model_name,
    const char* target_dir,
    EddyDownloadProgressCallback progress_callback,
    void* user_data,
    char** error_message) {

    if (!model_name || !target_dir) {
        if (error_message) {
            *error_message = copy_string("model_name and target_dir must not be NULL");
        }
        return EDDY_ERROR_INVALID_ARGUMENT;
    }

    try {
        // Look up model configuration
        const auto it = eddy::model_configs::MODEL_MAP.find(model_name);
        if (it == eddy::model_configs::MODEL_MAP.end()) {
            if (error_message) {
                *error_message = copy_string("Unknown model: " + std::string(model_name));
            }
            return EDDY_ERROR_INVALID_ARGUMENT;
        }

        const eddy::ModelConfig& config = it->second;

        // Create progress callback wrapper
        eddy::model_utils::DownloadProgressCallback cpp_callback = nullptr;
        if (progress_callback) {
            cpp_callback = [progress_callback, user_data](const std::string& filename, int current, int total) {
                progress_callback(filename.c_str(), current, total, user_data);
            };
        }

        // Download models
        std::string last_error;
        bool success = eddy::model_utils::download_models(
            config,
            std::filesystem::path(target_dir),
            &last_error,
            cpp_callback,
            true  // skip_existing
        );

        if (!success) {
            if (error_message) {
                *error_message = copy_string(last_error);
            }
            return EDDY_ERROR_UNKNOWN;
        }

        if (error_message) {
            *error_message = nullptr;
        }
        return EDDY_OK;

    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = capture_exception(e);
        }
        return EDDY_ERROR_UNKNOWN;
    }
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

        // Convert chunks (exception-safe)
        result->num_chunks = cpp_result.chunks.size();
        if (result->num_chunks > 0) {
            result->chunks = new EddyWhisperChunk[result->num_chunks];
            // Initialize all text pointers to nullptr for safe cleanup
            for (size_t i = 0; i < result->num_chunks; i++) {
                result->chunks[i].text = nullptr;
            }
            try {
                for (size_t i = 0; i < result->num_chunks; i++) {
                    result->chunks[i].start_ts = cpp_result.chunks[i].start_ts;
                    result->chunks[i].end_ts = cpp_result.chunks[i].end_ts;
                    result->chunks[i].text = copy_string(cpp_result.chunks[i].text);
                }
            } catch (...) {
                // Cleanup partially allocated chunks
                for (size_t i = 0; i < result->num_chunks; i++) {
                    if (result->chunks[i].text) delete[] result->chunks[i].text;
                }
                delete[] result->chunks;
                result->chunks = nullptr;
                result->num_chunks = 0;
                throw; // Re-throw to be caught by outer handler
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

        // Convert chunks (exception-safe)
        result->num_chunks = cpp_result.chunks.size();
        if (result->num_chunks > 0) {
            result->chunks = new EddyWhisperChunk[result->num_chunks];
            // Initialize all text pointers to nullptr for safe cleanup
            for (size_t i = 0; i < result->num_chunks; i++) {
                result->chunks[i].text = nullptr;
            }
            try {
                for (size_t i = 0; i < result->num_chunks; i++) {
                    result->chunks[i].start_ts = cpp_result.chunks[i].start_ts;
                    result->chunks[i].end_ts = cpp_result.chunks[i].end_ts;
                    result->chunks[i].text = copy_string(cpp_result.chunks[i].text);
                }
            } catch (...) {
                // Cleanup partially allocated chunks
                for (size_t i = 0; i < result->num_chunks; i++) {
                    if (result->chunks[i].text) delete[] result->chunks[i].text;
                }
                delete[] result->chunks;
                result->chunks = nullptr;
                result->num_chunks = 0;
                throw; // Re-throw to be caught by outer handler
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

        // Infer model version from blank_token_id
        const char* model_name = (config.blank_token_id == 8192) ? "parakeet-v3" : "parakeet-v2";

        auto backend = std::make_shared<eddy::OpenVINOBackend>(
            eddy::OpenVINOOptions{ .device = device, .cache_dir = eddy::get_model_dir(model_name).string() }
        );

        // Resolve model directory: prefer explicit, else cache and ensure availability
        std::filesystem::path model_dir;
        if (config.model_dir && std::string(config.model_dir).size() > 0) {
            std::string dir_str = config.model_dir;
            // Treat "cache" as a special value meaning "use default cache location"
            if (dir_str == "cache") {
                model_dir = eddy::get_model_assets_dir(model_name);
            } else {
                model_dir = config.model_dir;
            }
        } else {
            model_dir = eddy::get_model_assets_dir(model_name);
        }

        std::string err;
        (void)eddy::model_utils::check_models_available(model_dir, &err);
#if defined(_WIN32)
        if (!std::filesystem::exists(model_dir)) {
            auto legacy = eddy::get_app_data_dir() / "cache" / "models" / "parakeet-v2" / "files";
            if (std::filesystem::exists(legacy)) model_dir = legacy;
        }
#endif

        // Both v2 and v3 use the same vocab filename (as per HuggingFace repos)
        std::string vocab_filename = "parakeet_vocab.json";

        eddy::parakeet::ModelPaths paths{
            .preprocessor = {.path = (model_dir / "parakeet_melspectogram.xml").string()},
            .encoder = {.path = (model_dir / "parakeet_encoder.xml").string()},
            .decoder = {.path = (model_dir / "parakeet_decoder.xml").string()},
            .joint = {.path = (model_dir / "parakeet_joint.xml").string()},
            .tokenizer_json = (model_dir / vocab_filename).string()
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
    if (!handle) {
        if (err) *err = copy_string("[Eddy Error] Invalid argument: null handle");
        return EDDY_ERROR_INVALID_ARGUMENT;
    }
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

// -----------------------------
// Nemotron streaming C API
// -----------------------------

static constexpr const char* kNemotronModelName = "nemotron-streaming";

struct CNemotron {
    std::unique_ptr<eddy::nemotron::OpenVINONemotron> model;
};

EDDY_API void eddy_nemotron_free_result(EddyNemotronResult* result) {
    if (!result) return;
    if (result->text) { delete[] result->text; result->text = nullptr; }
    if (result->detected_language) { delete[] result->detected_language; result->detected_language = nullptr; }
    if (result->token_ids) { delete[] result->token_ids; result->token_ids = nullptr; }
    result->num_tokens = 0;
}

EDDY_API EddyNemotronModel eddy_nemotron_create(EddyNemotronConfig config, char** error_message) {
    try {
        const std::string device = config.device ? config.device : "CPU";
        const std::string language = config.language ? config.language : "auto";

        // Resolve model directory: explicit dir, else (NULL/"cache") the Eddy cache.
        const std::string md = config.model_dir ? config.model_dir : "";
        std::filesystem::path model_dir =
            (!md.empty() && md != "cache") ? std::filesystem::path(md)
                                           : eddy::get_model_assets_dir(kNemotronModelName);

        auto backend = std::make_shared<eddy::OpenVINOBackend>(
            eddy::OpenVINOOptions{ .device = device,
                                   .cache_dir = eddy::get_model_dir(kNemotronModelName).string() }
        );

        // Fail fast at create time if the model files are missing, matching the
        // eddy_parakeet_create contract. Otherwise the first failure only surfaces
        // deep inside ensure_compiled() on the initial transcribe()/warmup() call.
        {
            std::string check_err;
            if (!eddy::model_utils::check_models_available(
                    model_dir, &check_err, eddy::model_configs::NEMOTRON_FILES)) {
                throw std::runtime_error(
                    "Nemotron model files not available in '" + model_dir.string() +
                    "': " + check_err);
            }
        }

        eddy::nemotron::ModelPaths paths{
            .preprocessor  = (model_dir / "nemotron_preprocessor.xml").string(),
            .encoder       = (model_dir / "nemotron_encoder.xml").string(),
            .decoder       = (model_dir / "nemotron_decoder.xml").string(),
            .joint         = (model_dir / "nemotron_joint.xml").string(),
            .vocab_json    = (model_dir / "nemotron_vocab.json").string(),
            .metadata_json = (model_dir / "metadata.json").string(),
        };

        eddy::nemotron::Config cfg;
        cfg.device = device;
        cfg.language = language;

        auto handle = std::make_unique<CNemotron>();
        handle->model = std::make_unique<eddy::nemotron::OpenVINONemotron>(backend, paths, cfg);
        return static_cast<EddyNemotronModel>(handle.release());
    } catch (const std::exception& e) {
        if (error_message) *error_message = capture_exception(e);
        return nullptr;
    } catch (...) {
        if (error_message) *error_message = copy_string("[Eddy Error] Unknown exception in nemotron create");
        return nullptr;
    }
}

EDDY_API void eddy_nemotron_destroy(EddyNemotronModel handle) {
    if (!handle) return;
    delete static_cast<CNemotron*>(handle);
}

static EddyError nemotron_fill_result(const eddy::nemotron::TranscriptionResult& res, EddyNemotronResult* out) {
    out->text = copy_string(res.text);
    out->detected_language = copy_string(res.detected_language);
    out->prompt_id_used = res.prompt_id_used;
    out->latency_ms = res.latency_ms;
    out->num_tokens = res.token_ids.size();
    if (out->num_tokens > 0) {
        out->token_ids = new int[out->num_tokens];
        for (size_t i = 0; i < out->num_tokens; ++i) out->token_ids[i] = res.token_ids[i];
    } else {
        out->token_ids = nullptr;
    }
    return EDDY_OK;
}

EDDY_API EddyError eddy_nemotron_infer_buffer(EddyNemotronModel handle, const float* pcm, size_t length,
                                              int sample_rate, EddyNemotronResult* out, char** err) {
    if (!handle || !pcm || !out) {
        if (err) *err = copy_string("[Eddy Error] Invalid argument: null pointer");
        return EDDY_ERROR_INVALID_ARGUMENT;
    }
    // Zero-init before any other early return so callers that follow the
    // "always safe to eddy_nemotron_free_result" contract never delete[]
    // uninitialized pointers (and so a throw mid-fill is cleaned up in catch).
    *out = EddyNemotronResult{};
    if (sample_rate != 16000) {
        if (err) *err = copy_string("[Eddy Error] Nemotron expects 16kHz mono audio");
        return EDDY_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto* h = static_cast<CNemotron*>(handle);
        std::vector<float> samples(pcm, pcm + length);
        return nemotron_fill_result(h->model->transcribe(samples), out);
    } catch (const std::exception& e) {
        eddy_nemotron_free_result(out);
        if (err) *err = capture_exception(e);
        return EDDY_ERROR_INFERENCE_FAILED;
    } catch (...) {
        eddy_nemotron_free_result(out);
        if (err) *err = copy_string("[Eddy Error] Unknown exception during nemotron inference");
        return EDDY_ERROR_UNKNOWN;
    }
}

EDDY_API EddyError eddy_nemotron_infer_file(EddyNemotronModel handle, const char* wav_path,
                                            EddyNemotronResult* out, char** err) {
    if (!handle || !wav_path || !out) {
        if (err) *err = copy_string("[Eddy Error] Invalid argument: null pointer");
        return EDDY_ERROR_INVALID_ARGUMENT;
    }
    // Zero-init on every path (FILE_NOT_FOUND, a read_wav throw on a malformed
    // file, ...) so a caller that frees *out after any error never delete[]s
    // uninitialized pointers.
    *out = EddyNemotronResult{};
    // Only a genuinely missing file is FILE_NOT_FOUND; read_wav also throws for
    // format/channel/sample-rate/decode errors, which are not filesystem issues.
    std::error_code ec;
    if (!std::filesystem::exists(wav_path, ec)) {
        if (err) *err = copy_string("[Eddy Error] WAV file not found: " + std::string(wav_path));
        return EDDY_ERROR_FILE_NOT_FOUND;
    }
    try {
        auto pcm = eddy::audio::read_wav(wav_path);
        return eddy_nemotron_infer_buffer(handle, pcm.data(), pcm.size(), 16000, out, err);
    } catch (const std::exception& e) {
        if (err) *err = capture_exception(e);
        return EDDY_ERROR_INFERENCE_FAILED;
    } catch (...) {
        if (err) *err = copy_string("[Eddy Error] Unknown exception in nemotron infer_file");
        return EDDY_ERROR_UNKNOWN;
    }
}

} // extern "C"
