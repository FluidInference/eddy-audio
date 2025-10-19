// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/pipelines/whisper_pipeline.hpp"
#include "eddy/utils/audio_utils.hpp"

#include <openvino/genai/whisper_pipeline.hpp>
#include <openvino/openvino.hpp>

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace eddy {

struct WhisperPipeline::Impl {
    WhisperConfig config;
    std::unique_ptr<ov::genai::WhisperPipeline> pipeline;

    explicit Impl(const WhisperConfig& cfg) : config(cfg) {
        std::cout << "[Eddy] Initializing Whisper pipeline..." << std::endl;
        std::cout << "[Eddy] Model: " << config.model_path << std::endl;
        std::cout << "[Eddy] Device: " << config.device << std::endl;

        // Set up OpenVINO properties
        ov::AnyMap properties;

        // Enable model caching if requested
        if (config.enable_cache && !config.cache_dir.empty()) {
            properties[ov::cache_dir.name()] = config.cache_dir;
            std::cout << "[Eddy] Cache directory: " << config.cache_dir << std::endl;
        }

        std::cout << "[Eddy] Loading model... (First run on NPU may take 5+ minutes for compilation)" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // Create the OpenVINO GenAI WhisperPipeline
        pipeline = std::make_unique<ov::genai::WhisperPipeline>(
            config.model_path,
            config.device,
            properties
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "[Eddy] Model loaded in " << duration.count() << " ms" << std::endl;

        // Configure generation settings
        auto gen_config = pipeline->get_generation_config();

        // Set language
        if (config.language != "auto") {
            gen_config.language = "<|" + config.language + "|>";
        }

        // Set task
        gen_config.task = config.task;

        // Enable timestamps
        gen_config.return_timestamps = config.return_timestamps;

        pipeline->set_generation_config(gen_config);

        std::cout << "[Eddy] Whisper pipeline ready!" << std::endl;
    }
};

WhisperPipeline::WhisperPipeline(const WhisperConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

WhisperPipeline::~WhisperPipeline() = default;

WhisperPipeline::WhisperPipeline(WhisperPipeline&&) noexcept = default;
WhisperPipeline& WhisperPipeline::operator=(WhisperPipeline&&) noexcept = default;

WhisperResult WhisperPipeline::transcribe(const std::string& wav_path) {
    std::cout << "[Eddy] Reading audio file: " << wav_path << std::endl;

    // Read WAV file
    auto pcm = audio::read_wav(wav_path);

    std::cout << "[Eddy] Audio loaded: " << pcm.size() << " samples ("
              << (pcm.size() / 16000.0) << " seconds)" << std::endl;

    // Transcribe
    return transcribe(pcm.data(), pcm.size(), 16000);
}

WhisperResult WhisperPipeline::transcribe(const float* pcm, size_t length, int sample_rate) {
    if (sample_rate != 16000) {
        throw std::runtime_error("Only 16kHz sample rate is supported, got " + std::to_string(sample_rate));
    }

    std::cout << "[Eddy] Running transcription..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Create input
    ov::genai::RawSpeechInput input(pcm, pcm + length);

    // Run inference
    auto ov_result = impl_->pipeline->generate(input);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Convert result
    WhisperResult result;

    if (!ov_result.texts.empty()) {
        result.text = ov_result.texts[0];
    }

    if (!ov_result.scores.empty()) {
        result.confidence = ov_result.scores[0];
    }

    // Convert chunks if available
    if (ov_result.chunks.has_value()) {
        for (const auto& chunk : *ov_result.chunks) {
            WhisperChunk eddy_chunk;
            eddy_chunk.start_ts = chunk.start_ts;
            eddy_chunk.end_ts = chunk.end_ts;
            eddy_chunk.text = chunk.text;
            result.chunks.push_back(eddy_chunk);
        }
    }

    result.inference_duration_ms = static_cast<double>(duration.count());

    std::cout << "[Eddy] Transcription complete in " << result.inference_duration_ms << " ms" << std::endl;
    std::cout << "[Eddy] Result: " << result.text << std::endl;

    return result;
}

void WhisperPipeline::set_language(const std::string& language) {
    impl_->config.language = language;

    auto gen_config = impl_->pipeline->get_generation_config();
    if (language != "auto") {
        gen_config.language = "<|" + language + "|>";
    } else {
        gen_config.language = std::nullopt;
    }
    impl_->pipeline->set_generation_config(gen_config);

    std::cout << "[Eddy] Language set to: " << language << std::endl;
}

std::string WhisperPipeline::get_language() const {
    return impl_->config.language;
}

void WhisperPipeline::set_task(const std::string& task) {
    impl_->config.task = task;

    auto gen_config = impl_->pipeline->get_generation_config();
    gen_config.task = task;
    impl_->pipeline->set_generation_config(gen_config);

    std::cout << "[Eddy] Task set to: " << task << std::endl;
}

const WhisperConfig& WhisperPipeline::get_config() const {
    return impl_->config;
}

}  // namespace eddy