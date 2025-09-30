// Copyright (C) 2025 Fluid Inference
// SPDX-License-Identifier: Apache-2.0

using System;
using System.Runtime.InteropServices;

namespace Eddy;

internal static class Native
{
    private const string LibName = "eddy_c";

    // Opaque handle
    internal struct EddyWhisperPipelineHandle : IDisposable
    {
        private IntPtr handle;

        public EddyWhisperPipelineHandle(IntPtr h) => handle = h;
        public bool IsValid => handle != IntPtr.Zero;
        public IntPtr Handle => handle;

        public void Dispose()
        {
            if (handle != IntPtr.Zero)
            {
                eddy_whisper_destroy(handle);
                handle = IntPtr.Zero;
            }
        }
    }

    // Config structures
    [StructLayout(LayoutKind.Sequential)]
    internal struct EddyWhisperConfig
    {
        public IntPtr model_path;
        public IntPtr device;
        public IntPtr language;
        public IntPtr task;
        [MarshalAs(UnmanagedType.I1)]
        public bool return_timestamps;
        [MarshalAs(UnmanagedType.I1)]
        public bool enable_cache;
        public IntPtr cache_dir;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct EddyWhisperChunk
    {
        public float start_ts;
        public float end_ts;
        public IntPtr text;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct EddyWhisperResult
    {
        public IntPtr text;
        public IntPtr chunks;
        public nuint num_chunks;
        public float confidence;
        public double inference_duration_ms;
    }

    internal enum EddyError
    {
        EDDY_OK = 0,
        EDDY_ERROR_INVALID_ARGUMENT = 1,
        EDDY_ERROR_MODEL_LOAD_FAILED = 2,
        EDDY_ERROR_INFERENCE_FAILED = 3,
        EDDY_ERROR_FILE_NOT_FOUND = 4,
        EDDY_ERROR_UNKNOWN = 99
    }

    // Whisper API
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr eddy_whisper_create(
        EddyWhisperConfig config,
        out IntPtr error_message
    );

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void eddy_whisper_destroy(IntPtr pipeline);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern EddyError eddy_whisper_transcribe_file(
        IntPtr pipeline,
        [MarshalAs(UnmanagedType.LPStr)] string wav_path,
        out EddyWhisperResult result,
        out IntPtr error_message
    );

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern EddyError eddy_whisper_transcribe_buffer(
        IntPtr pipeline,
        float[] pcm,
        nuint length,
        int sample_rate,
        out EddyWhisperResult result,
        out IntPtr error_message
    );

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern void eddy_whisper_set_language(
        IntPtr pipeline,
        [MarshalAs(UnmanagedType.LPStr)] string language
    );

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern void eddy_whisper_set_task(
        IntPtr pipeline,
        [MarshalAs(UnmanagedType.LPStr)] string task
    );

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr eddy_whisper_get_language(IntPtr pipeline);

    // Memory management
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void eddy_whisper_free_result(ref EddyWhisperResult result);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void eddy_free_string(IntPtr str);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr eddy_version();

    // Helper to marshal string
    internal static string? PtrToStringAndFree(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) return null;
        var str = Marshal.PtrToStringAnsi(ptr);
        eddy_free_string(ptr);
        return str;
    }
}
