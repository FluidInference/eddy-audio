#!/usr/bin/env python3
"""
Benchmark Whisper OpenVINO (eddy optimized) vs PyTorch Whisper
Compares FluidInference whisper-large-v3-turbo-fp16-ov-npu with OpenAI Whisper
"""
import time
import numpy as np
from pathlib import Path
import librosa

def benchmark_whisper_pytorch(audio_files, model_name="base.en"):
    """Benchmark PyTorch Whisper on given audio files"""
    import whisper

    print(f"Loading PyTorch Whisper {model_name} model...")
    model = whisper.load_model(model_name)

    results = []

    for audio_file in audio_files:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print(f"Skipping {audio_file} (not found)")
            continue

        print(f"\nProcessing: {audio_path.name}")

        # Load audio to get duration
        audio, sr = librosa.load(audio_file, sr=16000)
        duration = len(audio) / sr

        # Transcribe with timing
        start_time = time.time()
        result = model.transcribe(audio_file)
        elapsed = time.time() - start_time

        rtf = elapsed / duration

        print(f"  Duration: {duration:.2f}s")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.2f}x")
        print(f"  Text: {result['text'][:80]}...")

        results.append({
            'file': audio_path.name,
            'duration': duration,
            'elapsed': elapsed,
            'rtf': rtf,
            'text': result['text']
        })

    # Calculate average RTF
    if results:
        avg_rtf = np.mean([r['rtf'] for r in results])
        total_duration = sum(r['duration'] for r in results)
        total_elapsed = sum(r['elapsed'] for r in results)

        print(f"\n{'='*60}")
        print(f"PyTorch Whisper {model_name} Summary:")
        print(f"  Files processed: {len(results)}")
        print(f"  Total audio: {total_duration:.2f}s")
        print(f"  Total time: {total_elapsed:.2f}s")
        print(f"  Average RTF: {avg_rtf:.2f}x")
        print(f"{'='*60}")

        return avg_rtf

    return None

def benchmark_whisper_openvino(audio_files, model_path, device="NPU"):
    """Benchmark OpenVINO Whisper (eddy optimized) on given audio files"""
    try:
        from openvino import Core
        from openvino.runtime import opset13 as ops
        import openvino_genai as ov_genai
    except ImportError:
        print("ERROR: openvino-genai not installed. Install with:")
        print("  pip install openvino openvino-genai")
        return None

    print(f"Loading OpenVINO Whisper from {model_path}...")
    print(f"Device: {device}")

    # Create pipeline
    pipe = ov_genai.WhisperPipeline(model_path, device)

    results = []

    for audio_file in audio_files:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print(f"Skipping {audio_file} (not found)")
            continue

        print(f"\nProcessing: {audio_path.name}")

        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        duration = len(audio) / sr

        # Transcribe with timing
        start_time = time.time()
        result = pipe.generate(audio.astype(np.float32))
        elapsed = time.time() - start_time

        rtf = elapsed / duration

        # Extract text from result
        text = result.texts[0] if hasattr(result, 'texts') else str(result)

        print(f"  Duration: {duration:.2f}s")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.2f}x")
        print(f"  Text: {text[:80]}...")

        results.append({
            'file': audio_path.name,
            'duration': duration,
            'elapsed': elapsed,
            'rtf': rtf,
            'text': text
        })

    # Calculate average RTF
    if results:
        avg_rtf = np.mean([r['rtf'] for r in results])
        total_duration = sum(r['duration'] for r in results)
        total_elapsed = sum(r['elapsed'] for r in results)

        print(f"\n{'='*60}")
        print(f"OpenVINO Whisper ({device}) Summary:")
        print(f"  Files processed: {len(results)}")
        print(f"  Total audio: {total_duration:.2f}s")
        print(f"  Total time: {total_elapsed:.2f}s")
        print(f"  Average RTF: {avg_rtf:.2f}x")
        print(f"{'='*60}")

        return avg_rtf

    return None

def download_whisper_ov_model():
    """Download FluidInference Whisper OpenVINO model from HuggingFace"""
    from huggingface_hub import snapshot_download

    model_id = "FluidInference/whisper-large-v3-turbo-fp16-ov-npu"
    cache_dir = Path.home() / ".cache" / "eddy" / "models" / "whisper-large-v3-turbo"

    print(f"Downloading {model_id}...")
    model_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir=cache_dir / "files",
        local_dir_use_symlinks=False
    )

    print(f"Model downloaded to: {model_path}")
    return model_path

if __name__ == "__main__":
    # Test files
    test_files = [
        "test_0011.wav",
        "test_0021.wav",
        "test_0037.wav",
        "test_1089_134686_0012.wav"
    ]

    print("="*60)
    print("Whisper Benchmark: OpenVINO vs PyTorch")
    print("="*60)

    # Try OpenVINO Whisper first
    try:
        import openvino_genai
        print("\n[1/2] Benchmarking OpenVINO Whisper (eddy optimized)")
        print("="*60)

        # Check if model exists, download if not
        model_path = Path.home() / ".cache" / "eddy" / "models" / "whisper-large-v3-turbo" / "files"
        if not model_path.exists():
            print("\nModel not found, downloading from HuggingFace...")
            model_path = download_whisper_ov_model()

        ov_rtf_npu = benchmark_whisper_openvino(test_files, str(model_path), device="NPU")
        ov_rtf_cpu = benchmark_whisper_openvino(test_files, str(model_path), device="CPU")

    except ImportError:
        print("\nOpenVINO GenAI not installed. Skipping OpenVINO benchmark.")
        print("Install with: pip install openvino openvino-genai")
        ov_rtf_npu = None
        ov_rtf_cpu = None

    # Benchmark PyTorch Whisper
    print("\n\n[2/2] Benchmarking PyTorch Whisper (baseline)")
    print("="*60)
    pytorch_rtf = benchmark_whisper_pytorch(test_files, model_name="base.en")

    # Summary comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON:")
    print(f"{'='*60}")

    if ov_rtf_npu:
        print(f"  OpenVINO Whisper (NPU):     {ov_rtf_npu:.2f}x RTF")
        print(f"  OpenVINO Whisper (CPU):     {ov_rtf_cpu:.2f}x RTF")
        print(f"  PyTorch Whisper (CPU):      {pytorch_rtf:.2f}x RTF")
        print(f"\n  Speedup (NPU vs PyTorch):   {pytorch_rtf / ov_rtf_npu:.2f}x faster")
        print(f"  Speedup (CPU vs PyTorch):   {pytorch_rtf / ov_rtf_cpu:.2f}x faster")
    else:
        print(f"  PyTorch Whisper (CPU):      {pytorch_rtf:.2f}x RTF")
        print(f"\n  (OpenVINO benchmark skipped - install openvino-genai to compare)")

    print(f"{'='*60}")
