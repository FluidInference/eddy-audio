#!/usr/bin/env python3
"""
Fast benchmark using C++ for inference + Python for WER calculation.

This is the EFFICIENT approach:
- C++ loads models once, processes all files (fast)
- Python calculates WER with Whisper normalization (accurate)

Workflow:
1. Run C++ benchmark → outputs transcriptions.json
2. Python WER calculation → outputs final_results.json

Usage:
    python benchmark.py --max-files 2620
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import os
import re
from typing import Dict, Any

# Import dependencies
try:
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: cd benchmarks && uv pip install whisper-normalizer jiwer")
    sys.exit(1)

# Initialize Whisper's English text normalizer (industry standard)
english_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    """
    Normalize text for WER calculation using OpenAI Whisper's normalizer.

    This is the industry standard used by:
    - OpenAI Whisper
    - Hugging Face Open ASR Leaderboard
    - FluidAudio/Parakeet benchmarks
    - Most competitive ASR systems
    """
    return english_normalizer(text)


def load_librispeech_transcripts(base_dir: Path = None):
    """
    Load LibriSpeech transcripts from .trans.txt files.

    Returns dict mapping file_id -> transcript text.
    """
    # Use cache directory
    if base_dir is None:
        if os.name == 'nt':  # Windows
            cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        else:  # Linux/Mac
            cache_base = Path.home() / '.cache'
        base_dir = cache_base / 'eddy' / 'datasets' / 'LibriSpeech' / 'test-clean'

    if not base_dir.exists():
        raise FileNotFoundError(
            f"LibriSpeech dataset not found at {base_dir}\n"
            f"C++ benchmark should have downloaded it automatically."
        )

    transcripts = {}

    # Find all .trans.txt files
    for trans_file in base_dir.rglob("*.trans.txt"):
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                # Format: "file_id TRANSCRIPT TEXT"
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    file_id, text = parts
                    transcripts[file_id] = text

    return transcripts


def calculate_wer(hypothesis: str, reference: str) -> Dict[str, Any]:
    """Calculate WER metrics using jiwer"""
    hyp_norm = normalize_text(hypothesis)
    ref_norm = normalize_text(reference)

    # Calculate WER
    wer_score = jiwer.wer(ref_norm, hyp_norm)

    # Get detailed measures using process_words
    output = jiwer.process_words(ref_norm, hyp_norm)

    return {
        "wer": wer_score * 100,  # as percentage
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "hypothesis_normalized": hyp_norm,
        "reference_normalized": ref_norm
    }


def recalculate_wer_with_normalization(cpp_output_file: str, final_output_file: str):
    """
    Recalculate WER from C++ results using Python normalization.

    Returns (python_wer, python_median_wer) or (None, None) on error.
    """
    # Load C++ benchmark results
    print("Calculating WER with Python normalization...")

    try:
        with open(cpp_output_file, 'r', encoding='utf-8') as f:
            # Read raw content and fix Windows path escaping if needed
            content = f.read()
            # Replace unescaped backslashes in paths (C:\Users -> C:\\Users)
            content = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', content)
            data = json.loads(content)
    except Exception as e:
        print(f"ERROR: Could not load C++ results: {e}")
        return None, None

    results = data.get('results', [])

    # Load LibriSpeech references
    try:
        transcripts = load_librispeech_transcripts()
    except Exception as e:
        print(f"ERROR: Could not load references: {e}")
        return None, None

    # Calculate WER for each result
    processed = 0
    failed = 0

    for result in results:
        file_id = result.get('file_id')
        hypothesis = result.get('hypothesis', '')

        # Get reference
        reference = transcripts.get(file_id)
        if not reference:
            print(f"WARN: No reference found for {file_id}")
            failed += 1
            continue

        # Calculate WER with Python normalization
        wer_metrics = calculate_wer(hypothesis, reference)

        # Update result
        result['reference'] = reference
        result.update(wer_metrics)
        processed += 1

    # Calculate summary statistics
    if processed > 0:
        avg_wer = sum(r.get("wer", 0) for r in results if "wer" in r) / processed
        median_wer = sorted(r.get("wer", 0) for r in results if "wer" in r)[processed // 2]

        # Update summary
        data['summary']['avg_wer'] = avg_wer
        data['summary']['median_wer'] = median_wer
        data['summary']['total_files'] = processed

        # Save results
        with open(final_output_file, "w") as f:
            json.dump(data, f, indent=2)

        return avg_wer, median_wer
    else:
        print("ERROR: No results processed")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Fast benchmark (C++ inference + Python WER)"
    )
    parser.add_argument("--max-files", type=int, default=25,
                       help="Maximum number of files to process (default: 25)")
    parser.add_argument("--device", default="CPU",
                       choices=["CPU", "GPU", "NPU", "AUTO"],
                       help="OpenVINO device: CPU, GPU, NPU, AUTO (default: CPU)")
    parser.add_argument("--output", default="eddy_benchmark_results.json",
                       help="Final output JSON file (default: eddy_benchmark_results.json)")

    args = parser.parse_args()

    print("=" * 80)
    print("eddy Fast Benchmark (C++ + Python)")
    print("=" * 80)
    print()
    print("Step 1: C++ inference (fast, models loaded once)")
    print("Step 2: Python WER calculation (Whisper normalization)")
    print()

    # Check if C++ benchmark exists (look in parent directory)
    cpp_benchmark = Path("../build/examples/cpp/Release/benchmark_librispeech.exe")
    if not cpp_benchmark.exists():
        print(f"ERROR: C++ benchmark not found at {cpp_benchmark}")
        print("\nBuild it with (from project root):")
        print("  cmake --build build --config Release --target benchmark_librispeech")
        sys.exit(1)

    # Step 1: Run C++ benchmark
    print("=" * 80)
    print("STEP 1: Running C++ benchmark...")
    print("=" * 80)
    print()

    # Note: C++ benchmark currently ignores --output and saves to hardcoded filename
    temp_output = "eddy_benchmark_results_cpp.json"
    cpp_cmd = [
        str(cpp_benchmark),
        "--max-files", str(args.max_files),
        "--device", args.device
    ]

    try:
        result = subprocess.run(cpp_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: C++ benchmark failed: {e}")
        sys.exit(1)

    # Load and display C++ results
    import json
    try:
        with open(temp_output, 'r', encoding='utf-8') as f:
            content = f.read()
            import re
            content = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', content)
            cpp_data = json.loads(content)

        cpp_summary = cpp_data.get('summary', {})
        cpp_wer = cpp_summary.get('average_wer_percent', 0)
        cpp_median_wer = cpp_summary.get('median_wer_percent', 0)
        cpp_rtfx = cpp_summary.get('overall_rtfx', 0)

        print()
        print("C++ Results (no text normalization):")
        print(f"  Average WER: {cpp_wer:.2f}%")
        print(f"  Median WER:  {cpp_median_wer:.2f}%")
        print(f"  Overall RTFx: {cpp_rtfx:.1f}x")
    except Exception as e:
        print(f"\nWARN: Could not load C++ results: {e}")
        cpp_wer = None

    # Step 2: Calculate WER with Python
    print()
    print("=" * 80)
    print("STEP 2: Calculating WER with Python normalization...")
    print("=" * 80)
    print()

    # Recalculate WER with normalization
    python_wer, python_median_wer = recalculate_wer_with_normalization(
        temp_output,
        args.output
    )

    if python_wer is not None:
        print()
        print("Python Results (with Whisper normalization):")
        print(f"  Average WER: {python_wer:.2f}%")
        print(f"  Median WER:  {python_median_wer:.2f}%")
    else:
        print("\nERROR: WER calculation failed")
        sys.exit(1)

    # Keep C++ output for reference (don't delete)
    print(f"\nC++ results preserved at: {temp_output}")

    print()
    print("=" * 80)
    print("COMPARISON: Impact of Text Normalization")
    print("=" * 80)
    if cpp_wer is not None and python_wer is not None:
        improvement = cpp_wer - python_wer
        improvement_pct = (improvement / cpp_wer * 100) if cpp_wer > 0 else 0

        print(f"C++ WER (no normalization):      {cpp_wer:.2f}%")
        print(f"Python WER (Whisper normalization): {python_wer:.2f}%")
        print(f"Improvement:                      {improvement:.2f}% ({improvement_pct:.1f}% better)")
        print()
        print("Text normalization (OpenAI Whisper standard):")
        print("  - Lowercase conversion")
        print("  - Punctuation removal")
        print("  - Number standardization")
        print("  - Whitespace normalization")
        print("  - Remove filler words and special tokens")
    print()
    print("=" * 80)
    print(f"Final results saved to: {args.output}")
    print("=" * 80)
    print()
    print("This approach is 2-3x faster than the subprocess-per-file approach!")


if __name__ == "__main__":
    main()
