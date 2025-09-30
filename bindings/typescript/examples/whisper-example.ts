// Copyright (C) 2025 Fluid Inference
// SPDX-License-Identifier: Apache-2.0

import { WhisperPipeline, getVersion } from '../src';

async function main() {
    const args = process.argv.slice(2);

    if (args.length < 2) {
        console.log('Usage: ts-node whisper-example.ts <MODEL_DIR> <WAV_FILE> [DEVICE] [LANGUAGE]');
        console.log('Example: ts-node whisper-example.ts ./models/whisper-v3-turbo ./audio.wav NPU en');
        process.exit(1);
    }

    const [modelPath, wavFile, device = 'NPU', language = 'en'] = args;

    try {
        console.log('=== eddy Whisper TypeScript Example ===');
        console.log(`SDK Version: ${getVersion()}`);
        console.log(`Model: ${modelPath}`);
        console.log(`Audio: ${wavFile}`);
        console.log(`Device: ${device}`);
        console.log(`Language: ${language}`);
        console.log();

        // Configure Whisper pipeline
        const config = {
            modelPath,
            device,
            language,
            task: 'transcribe',
            returnTimestamps: true,
            enableCache: true,
            cacheDir: './cache'
        };

        // Create pipeline
        console.log('Creating Whisper pipeline...');
        const pipeline = new WhisperPipeline(config);
        console.log();

        // Transcribe
        console.log('Transcribing audio...');
        const result = await pipeline.transcribe(wavFile);
        console.log();

        // Print results
        console.log('=== Transcription Result ===');
        console.log(`Text: ${result.text}`);
        console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
        console.log(`Inference Time: ${result.inferenceDurationMs.toFixed(2)} ms`);
        console.log();

        // Print timestamps if available
        if (result.chunks.length > 0) {
            console.log('=== Timestamps ===');
            for (const chunk of result.chunks) {
                const endTime = chunk.endTime >= 0 ? chunk.endTime.toFixed(2) : '?';
                console.log(`[${chunk.startTime.toFixed(2)} -> ${endTime}] ${chunk.text}`);
            }
        }

        // Cleanup
        pipeline.dispose();

    } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : error);
        process.exit(1);
    }
}

main();
