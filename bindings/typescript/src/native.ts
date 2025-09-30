// Copyright (C) 2025 Fluid Inference
// SPDX-License-Identifier: Apache-2.0

import * as ffi from 'ffi-napi';
import * as ref from 'ref-napi';
import * as StructType from 'ref-struct-di';
import * as path from 'path';

const Struct = StructType(ref);

// Determine library name based on platform
const libName = process.platform === 'win32' ? 'eddy.dll' : 'libeddy.so';
const libPath = path.join(__dirname, '..', '..', '..', 'build', 'Release', libName);

// Define native types
const FloatPtr = ref.refType('float');
const CharPtr = ref.refType('char');
const CharPtrPtr = ref.refType(CharPtr);
const VoidPtr = 'pointer';

// Structures
export const EddyWhisperConfig = Struct({
    model_path: CharPtr,
    device: CharPtr,
    language: CharPtr,
    task: CharPtr,
    return_timestamps: 'bool',
    enable_cache: 'bool',
    cache_dir: CharPtr
});

export const EddyWhisperChunk = Struct({
    start_ts: 'float',
    end_ts: 'float',
    text: CharPtr
});

export const EddyWhisperResult = Struct({
    text: CharPtr,
    chunks: ref.refType(EddyWhisperChunk),
    num_chunks: 'size_t',
    confidence: 'float',
    inference_duration_ms: 'double'
});

export enum EddyError {
    EDDY_OK = 0,
    EDDY_ERROR_INVALID_ARGUMENT = 1,
    EDDY_ERROR_MODEL_LOAD_FAILED = 2,
    EDDY_ERROR_INFERENCE_FAILED = 3,
    EDDY_ERROR_FILE_NOT_FOUND = 4,
    EDDY_ERROR_UNKNOWN = 99
}

// Native function bindings
export const lib = ffi.Library(libPath, {
    'eddy_whisper_create': [VoidPtr, [EddyWhisperConfig, CharPtrPtr]],
    'eddy_whisper_destroy': ['void', [VoidPtr]],
    'eddy_whisper_transcribe_file': ['int', [VoidPtr, 'string', ref.refType(EddyWhisperResult), CharPtrPtr]],
    'eddy_whisper_transcribe_buffer': ['int', [VoidPtr, FloatPtr, 'size_t', 'int', ref.refType(EddyWhisperResult), CharPtrPtr]],
    'eddy_whisper_set_language': ['void', [VoidPtr, 'string']],
    'eddy_whisper_set_task': ['void', [VoidPtr, 'string']],
    'eddy_whisper_get_language': [CharPtr, [VoidPtr]],
    'eddy_whisper_free_result': ['void', [ref.refType(EddyWhisperResult)]],
    'eddy_free_string': ['void', [CharPtr]],
    'eddy_version': [CharPtr, []]
});

export function getErrorMessage(errorPtr: Buffer): string {
    if (errorPtr.isNull()) return 'Unknown error';
    const message = ref.readCString(errorPtr, 0);
    lib.eddy_free_string(errorPtr);
    return message;
}
