// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace eddy {

/**
 * @brief Configuration for WhisperPipeline
 */
struct WhisperConfig {
    /// Path to the Whisper model directory (containing .xml/.bin files)
    std::string model_path;

    /// Device to run inference on: "NPU", "CPU", or "AUTO"
    std::string device = "NPU";

    /// Language code (e.g., "en", "zh", "es") or "auto" for auto-detection
    std::string language = "en";

    /// Task: "transcribe" or "translate" (translate to English)
    std::string task = "transcribe";

    /// Whether to return word-level timestamps
    bool return_timestamps = true;

    /// Enable model compilation caching (recommended for NPU)
    bool enable_cache = true;

    /// Cache directory for compiled models (empty = default OpenVINO cache location)
    std::string cache_dir;
};

/**
 * @brief A chunk of transcribed text with timestamps
 */
struct WhisperChunk {
    /// Start time in seconds
    float start_ts = 0.0f;

    /// End time in seconds (-1.0 if not available)
    float end_ts = -1.0f;

    /// Transcribed text for this chunk
    std::string text;
};

/**
 * @brief Result from Whisper transcription
 */
struct WhisperResult {
    /// Full transcribed text
    std::string text;

    /// Word/segment-level chunks with timestamps (if enabled)
    std::vector<WhisperChunk> chunks;

    /// Average confidence score (0.0 to 1.0)
    float confidence = 0.0f;

    /// Inference duration in milliseconds
    double inference_duration_ms = 0.0;
};

/**
 * @brief Whisper speech recognition pipeline
 *
 * Wraps OpenVINO GenAI's WhisperPipeline for easy audio transcription.
 * Supports language selection, timestamps, and NPU acceleration.
 */
class WhisperPipeline {
public:
    /**
     * @brief Construct a WhisperPipeline with the given configuration
     *
     * Note: First run with NPU device may take 5+ minutes for model compilation.
     * Subsequent runs will be fast if caching is enabled.
     *
     * @param config Configuration parameters
     */
    explicit WhisperPipeline(const WhisperConfig& config);

    /**
     * @brief Destructor
     */
    ~WhisperPipeline();

    // Non-copyable
    WhisperPipeline(const WhisperPipeline&) = delete;
    WhisperPipeline& operator=(const WhisperPipeline&) = delete;

    // Moveable
    WhisperPipeline(WhisperPipeline&&) noexcept;
    WhisperPipeline& operator=(WhisperPipeline&&) noexcept;

    /**
     * @brief Transcribe audio from a WAV file
     *
     * @param wav_path Path to WAV file (must be 16kHz, mono or stereo)
     * @return WhisperResult containing transcribed text and metadata
     */
    WhisperResult transcribe(const std::string& wav_path);

    /**
     * @brief Transcribe audio from raw PCM float32 buffer
     *
     * @param pcm Pointer to float32 PCM samples (normalized to [-1, 1])
     * @param length Number of samples
     * @param sample_rate Sample rate in Hz (default 16000)
     * @return WhisperResult containing transcribed text and metadata
     */
    WhisperResult transcribe(const float* pcm, size_t length, int sample_rate = 16000);

    /**
     * @brief Set the language for transcription
     *
     * @param language Language code (e.g., "en", "zh") or "auto"
     */
    void set_language(const std::string& language);

    /**
     * @brief Get the current language setting
     */
    std::string get_language() const;

    /**
     * @brief Set the task (transcribe or translate)
     *
     * @param task "transcribe" or "translate"
     */
    void set_task(const std::string& task);

    /**
     * @brief Get the current configuration
     */
    const WhisperConfig& get_config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eddy