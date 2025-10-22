// Copyright (C) 2025 Fluid Inference
// SPDX-License-Identifier: Apache-2.0

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Eddy;

/// <summary>
/// Configuration for Whisper pipeline
/// </summary>
public class WhisperConfig
{
    /// <summary>
    /// Path to the Whisper model directory (containing .xml/.bin files)
    /// </summary>
    public required string ModelPath { get; set; }

    /// <summary>
    /// Device to run inference on: "NPU", "CPU", or "AUTO"
    /// </summary>
    public string Device { get; set; } = "NPU";

    /// <summary>
    /// Language code (e.g., "en", "zh", "es") or "auto" for auto-detection
    /// </summary>
    public string Language { get; set; } = "en";

    /// <summary>
    /// Task: "transcribe" or "translate" (translate to English)
    /// </summary>
    public string Task { get; set; } = "transcribe";

    /// <summary>
    /// Whether to return word-level timestamps
    /// </summary>
    public bool ReturnTimestamps { get; set; } = true;

    /// <summary>
    /// Enable model compilation caching (recommended for NPU)
    /// </summary>
    public bool EnableCache { get; set; } = true;

    /// <summary>
    /// Cache directory for compiled models (null = default OpenVINO cache location)
    /// </summary>
    public string? CacheDir { get; set; } = null;
}

/// <summary>
/// A chunk of transcribed text with timestamps
/// </summary>
public class WhisperChunk
{
    /// <summary>
    /// Start time in seconds
    /// </summary>
    public float StartTime { get; init; }

    /// <summary>
    /// End time in seconds (-1.0 if not available)
    /// </summary>
    public float EndTime { get; init; }

    /// <summary>
    /// Transcribed text for this chunk
    /// </summary>
    public required string Text { get; init; }
}

/// <summary>
/// Result from Whisper transcription
/// </summary>
public class WhisperResult
{
    /// <summary>
    /// Full transcribed text
    /// </summary>
    public required string Text { get; init; }

    /// <summary>
    /// Word/segment-level chunks with timestamps (if enabled)
    /// </summary>
    public List<WhisperChunk> Chunks { get; init; } = new();

    /// <summary>
    /// Average confidence score (0.0 to 1.0)
    /// </summary>
    public float Confidence { get; init; }

    /// <summary>
    /// Inference duration in milliseconds
    /// </summary>
    public double InferenceDurationMs { get; init; }
}

/// <summary>
/// Whisper speech recognition pipeline
///
/// Wraps OpenVINO GenAI's WhisperPipeline for easy audio transcription.
/// Supports language selection, timestamps, and NPU acceleration.
/// </summary>
public class WhisperPipeline : IDisposable
{
    private IntPtr _handle;
    private bool _disposed = false;

    /// <summary>
    /// Construct a WhisperPipeline with the given configuration
    ///
    /// Note: First run with NPU device may take 5+ minutes for model compilation.
    /// Subsequent runs will be fast if caching is enabled.
    /// </summary>
    /// <param name="config">Configuration parameters</param>
    /// <exception cref="EddyException">Thrown if model loading fails</exception>
    public WhisperPipeline(WhisperConfig config)
    {
        var modelPathPtr = Marshal.StringToHGlobalAnsi(config.ModelPath);
        var devicePtr = Marshal.StringToHGlobalAnsi(config.Device);
        var languagePtr = Marshal.StringToHGlobalAnsi(config.Language);
        var taskPtr = Marshal.StringToHGlobalAnsi(config.Task);
        var cacheDirPtr = config.CacheDir != null
            ? Marshal.StringToHGlobalAnsi(config.CacheDir)
            : IntPtr.Zero;

        try
        {
            var nativeConfig = new Native.EddyWhisperConfig
            {
                model_path = modelPathPtr,
                device = devicePtr,
                language = languagePtr,
                task = taskPtr,
                return_timestamps = config.ReturnTimestamps,
                enable_cache = config.EnableCache,
                cache_dir = cacheDirPtr
            };

            _handle = Native.eddy_whisper_create(nativeConfig, out var errorMsg);

            if (_handle == IntPtr.Zero)
            {
                var error = Native.PtrToStringAndFree(errorMsg) ?? "Unknown error";
                throw new EddyException($"Failed to create Whisper pipeline: {error}");
            }
        }
        finally
        {
            Marshal.FreeHGlobal(modelPathPtr);
            Marshal.FreeHGlobal(devicePtr);
            Marshal.FreeHGlobal(languagePtr);
            Marshal.FreeHGlobal(taskPtr);
            if (cacheDirPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(cacheDirPtr);
        }
    }

    /// <summary>
    /// Transcribe audio from a WAV file
    /// </summary>
    /// <param name="wavPath">Path to WAV file (must be 16kHz, mono or stereo)</param>
    /// <returns>WhisperResult containing transcribed text and metadata</returns>
    /// <exception cref="EddyException">Thrown if transcription fails</exception>
    public WhisperResult Transcribe(string wavPath)
    {
        ThrowIfDisposed();

        var error = Native.eddy_whisper_transcribe_file(
            _handle,
            wavPath,
            out var result,
            out var errorMsg
        );

        if (error != Native.EddyError.EDDY_OK)
        {
            var errStr = Native.PtrToStringAndFree(errorMsg) ?? "Unknown error";
            throw new EddyException($"Transcription failed: {errStr}");
        }

        try
        {
            return ConvertResult(result);
        }
        finally
        {
            Native.eddy_whisper_free_result(ref result);
        }
    }

    /// <summary>
    /// Transcribe audio from raw PCM float32 buffer
    /// </summary>
    /// <param name="pcm">Float32 PCM samples (normalized to [-1, 1])</param>
    /// <param name="sampleRate">Sample rate in Hz (default 16000)</param>
    /// <returns>WhisperResult containing transcribed text and metadata</returns>
    /// <exception cref="EddyException">Thrown if transcription fails</exception>
    public WhisperResult Transcribe(float[] pcm, int sampleRate = 16000)
    {
        ThrowIfDisposed();

        var error = Native.eddy_whisper_transcribe_buffer(
            _handle,
            pcm,
            (nuint)pcm.Length,
            sampleRate,
            out var result,
            out var errorMsg
        );

        if (error != Native.EddyError.EDDY_OK)
        {
            var errStr = Native.PtrToStringAndFree(errorMsg) ?? "Unknown error";
            throw new EddyException($"Transcription failed: {errStr}");
        }

        try
        {
            return ConvertResult(result);
        }
        finally
        {
            Native.eddy_whisper_free_result(ref result);
        }
    }

    /// <summary>
    /// Set the language for transcription
    /// </summary>
    /// <param name="language">Language code (e.g., "en", "zh") or "auto"</param>
    public void SetLanguage(string language)
    {
        ThrowIfDisposed();
        Native.eddy_whisper_set_language(_handle, language);
    }

    /// <summary>
    /// Set the task (transcribe or translate)
    /// </summary>
    /// <param name="task">"transcribe" or "translate"</param>
    public void SetTask(string task)
    {
        ThrowIfDisposed();
        Native.eddy_whisper_set_task(_handle, task);
    }

    /// <summary>
    /// Get the current language setting
    /// </summary>
    public string GetLanguage()
    {
        ThrowIfDisposed();
        var ptr = Native.eddy_whisper_get_language(_handle);
        return Marshal.PtrToStringAnsi(ptr) ?? "en";
    }

    private WhisperResult ConvertResult(Native.EddyWhisperResult nativeResult)
    {
        var text = Marshal.PtrToStringAnsi(nativeResult.text) ?? "";
        var chunks = new List<WhisperChunk>();

        if (nativeResult.num_chunks > 0 && nativeResult.chunks != IntPtr.Zero)
        {
            var chunkSize = Marshal.SizeOf<Native.EddyWhisperChunk>();
            for (nuint i = 0; i < nativeResult.num_chunks; i++)
            {
                var chunkPtr = IntPtr.Add(nativeResult.chunks, (int)(i * (nuint)chunkSize));
                var nativeChunk = Marshal.PtrToStructure<Native.EddyWhisperChunk>(chunkPtr);
                var chunkText = Marshal.PtrToStringAnsi(nativeChunk.text) ?? "";

                chunks.Add(new WhisperChunk
                {
                    StartTime = nativeChunk.start_ts,
                    EndTime = nativeChunk.end_ts,
                    Text = chunkText
                });
            }
        }

        return new WhisperResult
        {
            Text = text,
            Chunks = chunks,
            Confidence = nativeResult.confidence,
            InferenceDurationMs = nativeResult.inference_duration_ms
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(WhisperPipeline));
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_handle != IntPtr.Zero)
        {
            Native.eddy_whisper_destroy(_handle);
            _handle = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~WhisperPipeline()
    {
        Dispose();
    }
}

/// <summary>
/// Exception thrown by eddy
/// </summary>
public class EddyException : Exception
{
    public EddyException(string message) : base(message) { }
    public EddyException(string message, Exception inner) : base(message, inner) { }
}
