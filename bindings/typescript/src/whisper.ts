// Copyright (C) 2025 Fluid Inference
// SPDX-License-Identifier: Apache-2.0

import * as ref from 'ref-napi';
import { lib, EddyWhisperConfig, EddyWhisperResult, EddyError, getErrorMessage } from './native';

/**
 * Configuration for Whisper pipeline
 */
export interface WhisperConfig {
    /** Path to the Whisper model directory (containing .xml/.bin files) */
    modelPath: string;
    /** Device to run inference on: "NPU", "CPU", "GPU", or "AUTO" */
    device?: string;
    /** Language code (e.g., "en", "zh", "es") or "auto" for auto-detection */
    language?: string;
    /** Task: "transcribe" or "translate" (translate to English) */
    task?: string;
    /** Whether to return word-level timestamps */
    returnTimestamps?: boolean;
    /** Enable model compilation caching (recommended for NPU) */
    enableCache?: boolean;
    /** Cache directory for compiled models (undefined = default OpenVINO cache location) */
    cacheDir?: string;
}

/**
 * A chunk of transcribed text with timestamps
 */
export interface WhisperChunk {
    /** Start time in seconds */
    startTime: number;
    /** End time in seconds (-1.0 if not available) */
    endTime: number;
    /** Transcribed text for this chunk */
    text: string;
}

/**
 * Result from Whisper transcription
 */
export interface WhisperResult {
    /** Full transcribed text */
    text: string;
    /** Word/segment-level chunks with timestamps (if enabled) */
    chunks: WhisperChunk[];
    /** Average confidence score (0.0 to 1.0) */
    confidence: number;
    /** Inference duration in milliseconds */
    inferenceDurationMs: number;
}

/**
 * Whisper speech recognition pipeline
 *
 * Wraps OpenVINO GenAI's WhisperPipeline for easy audio transcription.
 * Supports language selection, timestamps, and NPU acceleration.
 */
export class WhisperPipeline {
    private handle: Buffer;
    private disposed: boolean = false;

    /**
     * Construct a WhisperPipeline with the given configuration
     *
     * Note: First run with NPU device may take 5+ minutes for model compilation.
     * Subsequent runs will be fast if caching is enabled.
     *
     * @param config Configuration parameters
     * @throws Error if model loading fails
     */
    constructor(config: WhisperConfig) {
        const nativeConfig = new EddyWhisperConfig({
            model_path: ref.allocCString(config.modelPath),
            device: ref.allocCString(config.device || 'NPU'),
            language: ref.allocCString(config.language || 'en'),
            task: ref.allocCString(config.task || 'transcribe'),
            return_timestamps: config.returnTimestamps !== false,
            enable_cache: config.enableCache !== false,
            cache_dir: config.cacheDir ? ref.allocCString(config.cacheDir) : ref.NULL
        });

        const errorPtr = ref.alloc(ref.refType('char'));
        this.handle = lib.eddy_whisper_create(nativeConfig, errorPtr);

        if (this.handle.isNull()) {
            const error = getErrorMessage(errorPtr.deref());
            throw new Error(`Failed to create Whisper pipeline: ${error}`);
        }
    }

    /**
     * Transcribe audio from a WAV file
     *
     * @param wavPath Path to WAV file (must be 16kHz, mono or stereo)
     * @returns Promise resolving to WhisperResult containing transcribed text and metadata
     * @throws Error if transcription fails
     */
    async transcribe(wavPath: string): Promise<WhisperResult> {
        this.throwIfDisposed();

        return new Promise((resolve, reject) => {
            const result = ref.alloc(EddyWhisperResult);
            const errorPtr = ref.alloc(ref.refType('char'));

            const errorCode = lib.eddy_whisper_transcribe_file(
                this.handle,
                wavPath,
                result,
                errorPtr
            );

            if (errorCode !== EddyError.EDDY_OK) {
                const error = getErrorMessage(errorPtr.deref());
                lib.eddy_whisper_free_result(result);
                reject(new Error(`Transcription failed: ${error}`));
                return;
            }

            try {
                const converted = this.convertResult(result.deref());
                lib.eddy_whisper_free_result(result);
                resolve(converted);
            } catch (err) {
                lib.eddy_whisper_free_result(result);
                reject(err);
            }
        });
    }

    /**
     * Transcribe audio from raw PCM float32 buffer
     *
     * @param pcm Float32Array of PCM samples (normalized to [-1, 1])
     * @param sampleRate Sample rate in Hz (default 16000)
     * @returns Promise resolving to WhisperResult containing transcribed text and metadata
     * @throws Error if transcription fails
     */
    async transcribeBuffer(pcm: Float32Array, sampleRate: number = 16000): Promise<WhisperResult> {
        this.throwIfDisposed();

        return new Promise((resolve, reject) => {
            const buffer = Buffer.from(pcm.buffer);
            const result = ref.alloc(EddyWhisperResult);
            const errorPtr = ref.alloc(ref.refType('char'));

            const errorCode = lib.eddy_whisper_transcribe_buffer(
                this.handle,
                buffer,
                pcm.length,
                sampleRate,
                result,
                errorPtr
            );

            if (errorCode !== EddyError.EDDY_OK) {
                const error = getErrorMessage(errorPtr.deref());
                lib.eddy_whisper_free_result(result);
                reject(new Error(`Transcription failed: ${error}`));
                return;
            }

            try {
                const converted = this.convertResult(result.deref());
                lib.eddy_whisper_free_result(result);
                resolve(converted);
            } catch (err) {
                lib.eddy_whisper_free_result(result);
                reject(err);
            }
        });
    }

    /**
     * Set the language for transcription
     *
     * @param language Language code (e.g., "en", "zh") or "auto"
     */
    setLanguage(language: string): void {
        this.throwIfDisposed();
        lib.eddy_whisper_set_language(this.handle, language);
    }

    /**
     * Set the task (transcribe or translate)
     *
     * @param task "transcribe" or "translate"
     */
    setTask(task: string): void {
        this.throwIfDisposed();
        lib.eddy_whisper_set_task(this.handle, task);
    }

    /**
     * Get the current language setting
     */
    getLanguage(): string {
        this.throwIfDisposed();
        const ptr = lib.eddy_whisper_get_language(this.handle);
        return ref.readCString(ptr, 0);
    }

    private convertResult(nativeResult: any): WhisperResult {
        const text = ref.readCString(nativeResult.text, 0);
        const chunks: WhisperChunk[] = [];

        if (nativeResult.num_chunks > 0 && !nativeResult.chunks.isNull()) {
            for (let i = 0; i < nativeResult.num_chunks; i++) {
                const chunkPtr = ref.ref(nativeResult.chunks, i * ref.sizeof.pointer);
                const chunk = chunkPtr.deref();
                const chunkText = ref.readCString(chunk.text, 0);

                chunks.push({
                    startTime: chunk.start_ts,
                    endTime: chunk.end_ts,
                    text: chunkText
                });
            }
        }

        return {
            text,
            chunks,
            confidence: nativeResult.confidence,
            inferenceDurationMs: nativeResult.inference_duration_ms
        };
    }

    private throwIfDisposed(): void {
        if (this.disposed) {
            throw new Error('WhisperPipeline has been disposed');
        }
    }

    /**
     * Dispose of the pipeline and free native resources
     */
    dispose(): void {
        if (this.disposed) return;

        if (!this.handle.isNull()) {
            lib.eddy_whisper_destroy(this.handle);
        }

        this.disposed = true;
    }
}

/**
 * Get the eddy version
 */
export function getVersion(): string {
    const ptr = lib.eddy_version();
    return ref.readCString(ptr, 0);
}
