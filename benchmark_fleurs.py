"""
FLEURS Multilingual ASR Benchmark

This script evaluates Parakeet v3 on the FLEURS (Federated Learning Evaluation
Representation United States) multilingual dataset.

Supports 24 languages with automatic download from HuggingFace.
Based on FluidAudio's Swift implementation:
    FluidAudio/Sources/FluidAudioCLI/Commands/ASR/FleursBenchmark.swift

Quick test (10 files):
    uv run python benchmark_fleurs.py --languages en_us --samples 10 --device NPU

Compare with FluidAudio (Swift):
    swift run fluidaudio fleurs-benchmark --languages en_us --samples 10
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error
import ctypes as C

import numpy as np
import soundfile as sf

# Add parent directory to path for benchmark imports
sys.path.insert(0, str(Path(__file__).parent / "benchmarks"))
from benchmark import normalize_text, calculate_wer, load_lib, find_eddy_c_lib, EddyParakeetConfig, EddyParakeetResult


# ============================================================================
# Implementation based on FluidAudio Swift benchmark
# Source: FluidAudio/Sources/FluidAudioCLI/Commands/ASR/FleursBenchmark.swift
# ============================================================================

# Language codes mapped to Parakeet TDT v3 supported languages
# Based on the model's training data with reported WER performance
# (Matches FluidAudio/FleursBenchmark.swift:supportedLanguages)
SUPPORTED_LANGUAGES = {
    # Best performing languages (WER < 5%)
    "en_us": "English (US)",  # 4.85% WER
    "es_419": "Spanish (Spain)",  # 3.45% WER
    "it_it": "Italian (Italy)",  # 3.00% WER
    "fr_fr": "French (France)",  # 5.15% WER
    "de_de": "German (Germany)",  # 5.04% WER

    # Good performance (WER 5-10%)
    "ru_ru": "Russian (Russia)",  # 5.51% WER
    "nl_nl": "Dutch (Netherlands)",  # 7.48% WER
    "pl_pl": "Polish (Poland)",  # 7.31% WER
    "uk_ua": "Ukrainian (Ukraine)",  # 6.79% WER
    "sk_sk": "Slovak (Slovakia)",  # 8.82% WER

    # Moderate performance (WER 10-15%)
    "cs_cz": "Czech (Czechia)",  # 11.01% WER
    "bg_bg": "Bulgarian (Bulgaria)",  # 12.64% WER
    "hr_hr": "Croatian (Croatia)",  # 12.46% WER
    "ro_ro": "Romanian (Romania)",  # 12.44% WER
    "fi_fi": "Finnish (Finland)",  # 13.21% WER

    # Lower performance (WER > 15%)
    "hu_hu": "Hungarian (Hungary)",  # 15.72% WER
    "sv_se": "Swedish (Sweden)",  # 15.08% WER
    "et_ee": "Estonian (Estonia)",  # 17.73% WER
    "da_dk": "Danish (Denmark)",  # 18.41% WER
    "lt_lt": "Lithuanian (Lithuania)",  # 20.35% WER
    "el_gr": "Greek (Greece)",  # 20.70% WER
    "mt_mt": "Maltese (Malta)",  # 20.46% WER
    "lv_lv": "Latvian (Latvia)",  # 22.84% WER
    "sl_si": "Slovenian (Slovenia)",  # 24.03% WER
}

HIGH_WER_THRESHOLD = 0.30  # 30% WER threshold for flagging issues


@dataclass
class FLEURSSample:
    """Represents a single FLEURS test sample."""
    audio_path: str
    transcription: str
    language: str
    sample_id: str


@dataclass
class LanguageResults:
    """Results for a specific language."""
    language: str
    language_name: str
    wer: float
    cer: float
    rtfx: float
    samples_processed: int
    samples_skipped: int
    total_duration: float
    processing_time: float


@dataclass
class HighWERCase:
    """Case with high WER for analysis."""
    language: str
    sample_id: str
    reference: str
    hypothesis: str
    normalized_ref: str
    normalized_hyp: str
    wer: float
    duration: float
    audio_path: str


class FLEURSBenchmark:
    """FLEURS multilingual dataset benchmark for ASR evaluation."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        debug: bool = False,
        lib = None,
        parakeet = None
    ):
        """
        Initialize FLEURS benchmark.

        Args:
            cache_dir: Directory for caching FLEURS data
            debug: Enable debug logging
            lib: Loaded C library (eddy_c)
            parakeet: Parakeet model handle
        """
        if cache_dir is None:
            # Default: %LOCALAPPDATA%/eddy/datasets/FLEURS on Windows
            if sys.platform == "win32":
                local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
                cache_dir = os.path.join(local_appdata, "eddy", "datasets", "FLEURS")
            else:
                cache_dir = os.path.expanduser("~/Library/Application Support/FluidAudio/FLEURS")

        self.cache_dir = Path(cache_dir)
        self.debug = debug
        self.lib = lib
        self.parakeet = parakeet
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_language_samples(
        self,
        language: str,
        max_samples: Optional[int] = None
    ) -> bool:
        """
        Download FLEURS dataset for a specific language from HuggingFace.

        Based on FluidAudio Swift implementation:
        FluidAudio/Sources/FluidAudioCLI/Commands/ASR/FleursBenchmark.swift:downloadLanguageSamples

        Args:
            language: Language code (e.g., 'en_us')
            max_samples: Maximum number of samples to download (None = all)

        Returns:
            True if successful, False otherwise
        """
        if language not in SUPPORTED_LANGUAGES:
            print(f"Warning: Unsupported language: {language}")
            return False

        lang_dir = self.cache_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        trans_file = lang_dir / f"{language}.trans.txt"

        # Check if already downloaded
        if trans_file.exists():
            with open(trans_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            existing_audio = list(lang_dir.glob("*.wav"))
            expected_count = len(lines) if max_samples is None else min(len(lines), max_samples)

            if len(existing_audio) >= expected_count:
                print(f"FLEURS {language} already downloaded ({len(existing_audio)} files)")
                return True
            else:
                print(f"Found {len(existing_audio)} audio files, expected {expected_count}. Re-downloading.")

        print(f"Downloading FLEURS dataset for {SUPPORTED_LANGUAGES[language]}...")

        # Download from HuggingFace: FluidInference/fleurs
        dataset_repo = "FluidInference/fleurs"
        api_base_url = f"https://huggingface.co/api/datasets/{dataset_repo}/tree/main/{language}"

        try:
            # List files in the language directory
            with urllib.request.urlopen(api_base_url) as response:
                files_data = json.loads(response.read())

            # Download transcript file first
            for file_info in files_data:
                if file_info["type"] == "file" and file_info["path"].endswith(f"{language}.trans.txt"):
                    trans_url = f"https://huggingface.co/datasets/{dataset_repo}/resolve/main/{file_info['path']}"
                    print(f"Downloading transcript file...")

                    with urllib.request.urlopen(trans_url) as response:
                        trans_content = response.read().decode('utf-8')

                    with open(trans_file, 'w', encoding='utf-8') as f:
                        f.write(trans_content)

                    lines = [line.strip() for line in trans_content.split('\n') if line.strip()]
                    print(f"Downloaded {len(lines)} transcriptions")
                    break

            # Download audio files
            audio_files = [
                f for f in files_data
                if f["type"] == "file" and f["path"].endswith(".wav")
            ]

            max_download = len(audio_files) if max_samples is None else min(max_samples, len(audio_files))
            downloaded_count = 0

            for i, file_info in enumerate(audio_files[:max_download]):
                file_name = os.path.basename(file_info["path"])
                audio_file = lang_dir / file_name

                # Skip if already exists and is valid
                if audio_file.exists():
                    try:
                        # Quick validation
                        data, sr = sf.read(str(audio_file))
                        downloaded_count += 1
                        continue
                    except Exception:
                        # File is corrupted, re-download
                        print(f"Detected corrupted file {file_name}, re-downloading...")
                        audio_file.unlink()

                # Download audio file
                audio_url = f"https://huggingface.co/datasets/{dataset_repo}/resolve/main/{file_info['path']}"

                try:
                    urllib.request.urlretrieve(audio_url, str(audio_file))

                    # Validate downloaded file
                    try:
                        data, sr = sf.read(str(audio_file))
                        downloaded_count += 1

                        if (downloaded_count) % 10 == 0:
                            print(f"Downloaded {downloaded_count}/{max_download} audio files...")
                    except Exception:
                        print(f"Warning: Downloaded file {file_name} is not valid audio")
                        audio_file.unlink()

                except Exception as e:
                    print(f"Warning: Could not download {file_name}: {e}")

            print(f"Downloaded {downloaded_count} audio files")
            return True

        except Exception as e:
            print(f"Error downloading from HuggingFace: {e}")
            return False

    def load_samples(
        self,
        languages: List[str],
        max_samples_per_lang: Optional[int] = None
    ) -> List[FLEURSSample]:
        """
        Load FLEURS samples for benchmarking. Downloads if missing.

        Args:
            languages: List of language codes
            max_samples_per_lang: Maximum samples per language (None = all)

        Returns:
            List of FLEURS samples
        """
        all_samples = []

        for language in languages:
            lang_dir = self.cache_dir / language

            # Download if not exists
            if not lang_dir.exists() or not list(lang_dir.glob("*.wav")):
                print(f"Downloading {language}...")
                self.download_language_samples(language, max_samples_per_lang)

            if not lang_dir.exists():
                print(f"Warning: Failed to download {language}. Skipping.")
                continue

            # Load transcriptions from .trans.txt file (LibriSpeech format)
            trans_file = lang_dir / f"{language}.trans.txt"
            transcriptions = {}

            if trans_file.exists():
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split(' ', 1)
                        file_id = parts[0]
                        transcription = parts[1] if len(parts) > 1 else ""
                        transcriptions[file_id] = transcription

            # Load audio files and match with transcriptions
            audio_files = sorted(lang_dir.glob("*.wav"))

            if max_samples_per_lang is not None:
                audio_files = audio_files[:max_samples_per_lang]

            for audio_file in audio_files:
                file_id = audio_file.stem
                transcription = transcriptions.get(file_id, "")

                sample = FLEURSSample(
                    audio_path=str(audio_file),
                    transcription=transcription,
                    language=language,
                    sample_id=file_id
                )
                all_samples.append(sample)

            if audio_files:
                print(f"Loaded {len(audio_files)} samples for {language}")

        return all_samples

    def process_language_samples(
        self,
        samples: List[FLEURSSample],
        language: str
    ) -> Tuple[LanguageResults, List[HighWERCase]]:
        """
        Process samples for a specific language.

        Based on FluidAudio Swift implementation:
        FluidAudio/Sources/FluidAudioCLI/Commands/ASR/FleursBenchmark.swift:processLanguageSamples

        Args:
            samples: List of samples for this language
            language: Language code

        Returns:
            Tuple of (LanguageResults, list of high WER cases)
        """
        total_wer = 0.0
        total_cer = 0.0
        total_duration = 0.0
        total_processing_time = 0.0
        processed_count = 0
        skipped_count = 0

        high_wer_cases = []

        for sample in samples:
            if not os.path.exists(sample.audio_path):
                print(f"Warning: Audio file not found: {sample.audio_path}")
                skipped_count += 1
                continue

            try:
                # Load audio
                audio_data, sample_rate = sf.read(sample.audio_path)

                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    # Simple resampling (for production, use librosa or scipy)
                    duration = len(audio_data) / sample_rate
                    new_length = int(duration * 16000)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data), new_length),
                        np.arange(len(audio_data)),
                        audio_data
                    )

                audio_data = audio_data.astype(np.float32)
                audio_duration = len(audio_data) / 16000.0

                if self.debug:
                    print(f"  Processing {sample.audio_path}")
                    print(f"    Duration: {audio_duration:.2f}s, samples: {len(audio_data)}")

                # Call C API for transcription
                audio_buffer = audio_data.ctypes.data_as(C.POINTER(C.c_float))
                result = EddyParakeetResult()
                error_msg = C.c_char_p()

                ret_code = self.lib.eddy_parakeet_infer_buffer(
                    self.parakeet,
                    audio_buffer,
                    len(audio_data),
                    16000,  # sample rate
                    C.byref(result),
                    C.byref(error_msg),
                )

                if ret_code != 0:
                    error_text = error_msg.value.decode() if error_msg.value else "Unknown error"
                    raise RuntimeError(f"Inference failed: {error_text}")

                hypothesis = result.text.decode()
                processing_time = result.latency_ms / 1000.0  # Convert to seconds

                # Free the result
                self.lib.eddy_parakeet_free_result(C.byref(result))
                if error_msg.value:
                    self.lib.eddy_free_string(error_msg)

                # Calculate metrics if reference is available
                if sample.transcription:
                    metrics = calculate_wer(hypothesis, sample.transcription)
                    wer = metrics["wer"] / 100.0  # Convert from percentage
                    cer = metrics["cer"] / 100.0

                    total_wer += wer
                    total_cer += cer

                    # Track high WER cases
                    if wer > HIGH_WER_THRESHOLD:
                        normalized_ref = normalize_text(sample.transcription)
                        normalized_hyp = normalize_text(hypothesis)

                        high_wer_cases.append(HighWERCase(
                            language=language,
                            sample_id=sample.sample_id,
                            reference=sample.transcription,
                            hypothesis=hypothesis,
                            normalized_ref=normalized_ref,
                            normalized_hyp=normalized_hyp,
                            wer=wer,
                            duration=audio_duration,
                            audio_path=sample.audio_path
                        ))

                total_duration += audio_duration
                total_processing_time += processing_time
                processed_count += 1

                if self.debug:
                    print(f"    Hypothesis: {hypothesis}")
                    if sample.transcription:
                        print(f"    Reference:  {sample.transcription}")

            except Exception as e:
                print(f"Warning: Transcription error for {sample.sample_id}: {e}")
                skipped_count += 1

        # Calculate averages
        avg_wer = total_wer / processed_count if processed_count > 0 else 0.0
        avg_cer = total_cer / processed_count if processed_count > 0 else 0.0
        rtfx = total_duration / total_processing_time if total_processing_time > 0 else 0.0

        return (
            LanguageResults(
                language=language,
                language_name=SUPPORTED_LANGUAGES.get(language, language),
                wer=avg_wer,
                cer=avg_cer,
                rtfx=rtfx,
                samples_processed=processed_count,
                samples_skipped=skipped_count,
                total_duration=total_duration,
                processing_time=total_processing_time
            ),
            high_wer_cases
        )

    def run_benchmark(
        self,
        languages: List[str],
        max_samples_per_lang: Optional[int] = None
    ) -> Tuple[List[LanguageResults], List[HighWERCase]]:
        """
        Run multilingual FLEURS benchmark.

        Args:
            languages: List of language codes to test
            max_samples_per_lang: Maximum samples per language

        Returns:
            Tuple of (list of LanguageResults, list of high WER cases)
        """
        print("Starting FLEURS Multilingual ASR Benchmark")
        print("=" * 50)

        # Load samples (download first if missing)
        samples = self.load_samples(languages, max_samples_per_lang)

        if not samples:
            print("No samples found. Please ensure FLEURS data is available.")
            return ([], [])

        print(f"Processing {len(samples)} samples across {len(languages)} languages")

        # Group samples by language
        from itertools import groupby
        samples_by_lang = {}
        for lang, group in groupby(sorted(samples, key=lambda x: x.language), key=lambda x: x.language):
            samples_by_lang[lang] = list(group)

        results = []
        all_high_wer_cases = []

        for language, lang_samples in samples_by_lang.items():
            print(f"Processing {SUPPORTED_LANGUAGES.get(language, language)}...")

            lang_result, high_wer_cases = self.process_language_samples(
                lang_samples, language
            )

            results.append(lang_result)
            all_high_wer_cases.extend(high_wer_cases)

            # Print language summary
            skipped_info = f", {lang_result.samples_skipped} skipped" if lang_result.samples_skipped > 0 else ""
            print(
                f"{language}: WER={lang_result.wer * 100:.1f}%, "
                f"CER={lang_result.cer * 100:.1f}%, "
                f"RTFx={lang_result.rtfx:.1f}x "
                f"({lang_result.samples_processed} processed{skipped_info})"
            )

        return (results, all_high_wer_cases)

    def save_results(
        self,
        results: List[LanguageResults],
        output_path: str,
        languages: List[str],
        max_samples: Optional[int]
    ):
        """Save benchmark results to JSON file."""
        from datetime import datetime

        output = {
            "benchmark": "FLEURS Multilingual ASR",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "languages": languages,
                "samplesPerLanguage": max_samples if max_samples is not None else "all"
            },
            "results": [
                {
                    "language": r.language,
                    "languageName": r.language_name,
                    "wer": r.wer,
                    "cer": r.cer,
                    "rtfx": r.rtfx,
                    "samplesProcessed": r.samples_processed,
                    "samplesSkipped": r.samples_skipped,
                    "totalDuration": r.total_duration,
                    "processingTime": r.processing_time
                }
                for r in results
            ],
            "summary": {
                "averageWER": sum(r.wer for r in results) / len(results) if results else 0,
                "averageCER": sum(r.cer for r in results) / len(results) if results else 0,
                "averageRTFx": sum(r.rtfx for r in results) / len(results) if results else 0,
                "totalSamples": sum(r.samples_processed for r in results),
                "totalSkipped": sum(r.samples_skipped for r in results),
                "totalDuration": sum(r.total_duration for r in results),
                "totalProcessingTime": sum(r.processing_time for r in results)
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, sort_keys=True)

    def print_summary(self, results: List[LanguageResults]):
        """Print formatted summary table."""
        print("\n" + "=" * 89)
        print("FLEURS BENCHMARK SUMMARY")
        print("=" * 89)
        print()

        # Header
        print(f"{'Language':<25} | {'WER%':<6} | {'CER%':<6} | {'RTFx':<7} | {'Duration':<8} | {'Processed':<9} | {'Skipped':<7}")
        print("-" * 89)

        # Results
        for result in sorted(results, key=lambda r: r.language_name):
            wer_str = f"{result.wer * 100:.1f}"
            cer_str = f"{result.cer * 100:.1f}"
            rtfx_str = f"{result.rtfx:.1f}"
            duration_str = f"{result.total_duration:.1f}s"
            processed_str = str(result.samples_processed)
            skipped_str = str(result.samples_skipped) if result.samples_skipped > 0 else "-"

            print(
                f"{result.language_name:<25} | "
                f"{wer_str:<6} | "
                f"{cer_str:<6} | "
                f"{rtfx_str:<7} | "
                f"{duration_str:<8} | "
                f"{processed_str:<9} | "
                f"{skipped_str:<7}"
            )

        # Summary
        if results:
            print("-" * 89)
            avg_wer = sum(r.wer for r in results) / len(results)
            avg_cer = sum(r.cer for r in results) / len(results)
            avg_rtfx = sum(r.rtfx for r in results) / len(results)
            total_duration = sum(r.total_duration for r in results)
            total_processed = sum(r.samples_processed for r in results)
            total_skipped = sum(r.samples_skipped for r in results)

            print(
                f"{'AVERAGE':<25} | "
                f"{avg_wer * 100:<6.1f} | "
                f"{avg_cer * 100:<6.1f} | "
                f"{avg_rtfx:<7.1f} | "
                f"{total_duration:<8.1f}s | "
                f"{total_processed:<9} | "
                f"{total_skipped if total_skipped > 0 else '-':<7}"
            )

            if total_skipped > 0:
                print(f"\nNote: {total_skipped} samples were skipped due to errors")

    def print_high_wer_cases(self, cases: List[HighWERCase]):
        """Print all high WER cases for analysis."""
        if not cases:
            print(f"No high WER cases (> {int(HIGH_WER_THRESHOLD * 100)}%) detected.")
            return

        print(f"\nAll High WER Cases (>{int(HIGH_WER_THRESHOLD * 100)}%) Across Languages (sorted by WER):")
        print("=" * 80)

        # Sort by WER descending
        sorted_cases = sorted(cases, key=lambda c: (-c.wer, c.language))

        for case in sorted_cases:
            lang_name = SUPPORTED_LANGUAGES.get(case.language, case.language)
            print(f"Language: {lang_name} | File: {case.sample_id} "
                  f"(WER: {case.wer * 100:.1f}%, Duration: {case.duration:.2f}s)")
            print(f"Path: {case.audio_path}")
            print("-" * 40)
            print(f"Normalized Reference:  {case.normalized_ref}")
            print(f"Normalized Hypothesis: {case.normalized_hyp}")
            print(f"Original Hypothesis:   {case.hypothesis}")
            print("-" * 40)

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="FLEURS Multilingual ASR Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 files on NPU - recommended)
  python benchmark_fleurs.py --languages en_us --samples 10 --device NPU

  # Test all 24 languages with all available samples (~14,085 total)
  python benchmark_fleurs.py --device NPU

  # Test specific languages only
  python benchmark_fleurs.py --languages en_us,fr_fr,de_de,es_419 --device NPU

  # Custom output
  python benchmark_fleurs.py --output my_results.json --device NPU
        """
    )

    parser.add_argument(
        "--languages",
        type=str,
        default="all",
        help=f"Comma-separated language codes or 'all' (default: all). "
             f"Available: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="all",
        help="Number of samples per language or 'all' (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fleurs_benchmark_results.json",
        help="Output JSON file path (default: fleurs_benchmark_results.json)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory for caching FLEURS data (default: system-specific)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing Parakeet models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "NPU", "GPU"],
        help="Inference device (default: CPU)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--cpp",
        action="store_true",
        help="Use native C++ benchmark for 2-3x faster processing (requires build/examples/cpp/Release/benchmark_fleurs.exe)"
    )

    args = parser.parse_args()

    # Parse languages
    if args.languages.lower() == "all":
        languages = sorted(SUPPORTED_LANGUAGES.keys())
    else:
        languages = [lang.strip() for lang in args.languages.split(",")]
        # Validate languages
        invalid = [lang for lang in languages if lang not in SUPPORTED_LANGUAGES]
        if invalid:
            print(f"Error: Invalid language codes: {', '.join(invalid)}")
            print(f"Available: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}")
            return 1

    # Parse samples
    max_samples = None if args.samples.lower() == "all" else int(args.samples)

    print("FLEURS Multilingual ASR Benchmark")
    print("=" * 50)
    print(f"Languages: {'all (' + str(len(languages)) + ' languages)' if args.languages.lower() == 'all' else ', '.join(languages)}")
    print(f"Samples per language: {args.samples}")
    print(f"Output file: {args.output}")
    if args.cpp:
        print("Mode: C++ Native (2-3x faster)")
    else:
        print("Mode: Python (use --cpp for faster C++ implementation)")
    print()

    # If C++ mode is requested, call the C++ executable
    if args.cpp:
        import subprocess

        # Determine cache directory
        cache_dir = args.cache_dir
        if cache_dir is None:
            if sys.platform == "win32":
                local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
                cache_dir = os.path.join(local_appdata, "eddy", "datasets", "FLEURS")
            else:
                cache_dir = os.path.expanduser("~/Library/Application Support/FluidAudio/FLEURS")

        # Find the C++ benchmark executable
        project_root = Path(__file__).parent
        cpp_executable = project_root / "build" / "examples" / "cpp" / "Release" / "benchmark_fleurs.exe"

        if not cpp_executable.exists():
            print(f"Error: C++ benchmark not found at {cpp_executable}")
            print("Build it with: cmake --build build --config Release --target benchmark_fleurs")
            return 1

        # Build command
        cmd = [
            str(cpp_executable),
            cache_dir,
            "--languages", ",".join(languages),
            "--samples", str(max_samples if max_samples else 0),
            "--device", args.device,
            "--output", args.output
        ]

        if args.debug:
            cmd.append("--debug")

        print(f"Running C++ benchmark: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Error: C++ benchmark failed with exit code {e.returncode}")
            return e.returncode
        except FileNotFoundError:
            print(f"Error: Could not execute {cpp_executable}")
            return 1

    # Initialize C library
    print("Loading C library...")
    try:
        lib_path = find_eddy_c_lib()
        lib = load_lib(str(lib_path))
        print(f"Loaded: {lib_path}")
    except Exception as e:
        print(f"Error loading C library: {e}")
        return 1

    # Initialize Parakeet v3
    print(f"Initializing ASR system on {args.device} (Parakeet v3)...")
    model_dir = args.model_dir if args.model_dir else None

    config = EddyParakeetConfig(
        device=args.device.encode(),
        model_dir=model_dir.encode() if model_dir else None,
        blank_token_id=8192,  # Parakeet v3
    )

    error_msg = C.c_char_p()
    parakeet = lib.eddy_parakeet_create(config, C.byref(error_msg))

    if not parakeet:
        error_text = error_msg.value.decode() if error_msg.value else "Unknown error"
        print(f"Error initializing ASR system: {error_text}")
        if error_msg.value:
            lib.eddy_free_string(error_msg)
        return 1

    print("ASR system initialized")

    # Initialize benchmark
    benchmark = FLEURSBenchmark(
        cache_dir=args.cache_dir,
        debug=args.debug,
        lib=lib,
        parakeet=parakeet
    )

    # Run benchmark
    try:
        results, high_wer_cases = benchmark.run_benchmark(
            languages=languages,
            max_samples_per_lang=max_samples
        )

        # Save results
        benchmark.save_results(results, args.output, languages, max_samples)

        # Print summary
        benchmark.print_summary(results)

        # Print high WER cases
        benchmark.print_high_wer_cases(high_wer_cases)

        print(f"\nResults saved to {args.output}")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if parakeet:
            lib.eddy_parakeet_destroy(parakeet)

    return 0


if __name__ == "__main__":
    sys.exit(main())
