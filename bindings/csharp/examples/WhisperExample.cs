// Copyright (C) 2025 Fluid Inference
// SPDX-License-Identifier: Apache-2.0

using System;
using Eddy;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Usage: WhisperExample <MODEL_DIR> <WAV_FILE> [DEVICE] [LANGUAGE]");
            Console.WriteLine("Example: WhisperExample ./models/whisper-v3-turbo ./audio.wav NPU en");
            return;
        }

        string modelPath = args[0];
        string wavFile = args[1];
        string device = args.Length > 2 ? args[2] : "NPU";
        string language = args.Length > 3 ? args[3] : "en";

        try
        {
            Console.WriteLine("=== eddy Whisper C# Example ===");
            Console.WriteLine($"Model: {modelPath}");
            Console.WriteLine($"Audio: {wavFile}");
            Console.WriteLine($"Device: {device}");
            Console.WriteLine($"Language: {language}");
            Console.WriteLine();

            // Configure Whisper pipeline
            var config = new WhisperConfig
            {
                ModelPath = modelPath,
                Device = device,
                Language = language,
                Task = "transcribe",
                ReturnTimestamps = true,
                EnableCache = true,
                CacheDir = "./cache"
            };

            // Create pipeline
            Console.WriteLine("Creating Whisper pipeline...");
            using var pipeline = new WhisperPipeline(config);
            Console.WriteLine();

            // Transcribe
            Console.WriteLine("Transcribing audio...");
            var result = pipeline.Transcribe(wavFile);
            Console.WriteLine();

            // Print results
            Console.WriteLine("=== Transcription Result ===");
            Console.WriteLine($"Text: {result.Text}");
            Console.WriteLine($"Confidence: {result.Confidence * 100:F2}%");
            Console.WriteLine($"Inference Time: {result.InferenceDurationMs:F2} ms");
            Console.WriteLine();

            // Print timestamps if available
            if (result.Chunks.Count > 0)
            {
                Console.WriteLine("=== Timestamps ===");
                foreach (var chunk in result.Chunks)
                {
                    string endTime = chunk.EndTime >= 0 ? $"{chunk.EndTime:F2}" : "?";
                    Console.WriteLine($"[{chunk.StartTime:F2} -> {endTime}] {chunk.Text}");
                }
            }
        }
        catch (EddyException ex)
        {
            Console.Error.WriteLine($"eddy error: {ex.Message}");
            Environment.Exit(1);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            Environment.Exit(1);
        }
    }
}
