#!/usr/bin/env python3
"""
Download Parakeet-TDT 0.6B v2 OpenVINO model files into Eddy's cache.

Defaults to repo: alexwengg/parakeet-tdt-0.6b-v2-ov

Files are placed in:
  Windows: %LOCALAPPDATA%\eddy\cache\models\parakeet-v2\files\
  macOS:   ~/Library/Caches/eddy/models/parakeet-v2/files/
  Linux:   $XDG_CACHE_HOME/eddy/models/parakeet-v2/files or ~/.cache/eddy/models/parakeet-v2/files

Usage:
  python scripts/fetch_parakeet_ov.py
  python scripts/fetch_parakeet_ov.py --repo-id alexwengg/parakeet-tdt-0.6b-v2-ov
  python scripts/fetch_parakeet_ov.py --target /custom/path
"""

import argparse
import os
import sys
from pathlib import Path


def default_cache_dir() -> Path:
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise SystemExit("LOCALAPPDATA is not set; cannot resolve cache directory on Windows")
        return Path(local) / "eddy" / "cache" / "models" / "parakeet-v2" / "files"
    elif sys.platform == "darwin":
        home = os.environ.get("HOME")
        if not home:
            raise SystemExit("HOME is not set; cannot resolve cache directory on macOS")
        return Path(home) / "Library" / "Caches" / "eddy" / "models" / "parakeet-v2" / "files"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path(os.environ.get("HOME", "~")).expanduser() / ".cache"
        return base / "eddy" / "models" / "parakeet-v2" / "files"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Parakeet OV model into Eddy cache")
    parser.add_argument(
        "--repo-id",
        default="alexwengg/parakeet-tdt-0.6b-v2-ov",
        help="Hugging Face repo id to download",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Override target directory (will be created)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        return 2

    target_dir = args.target or default_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{args.repo_id}' to: {target_dir}")
    patterns = ["*.xml", "*.bin", "*.json"]

    # Place snapshot directly in target_dir (no symlinks for portability on Windows)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=patterns,
        repo_type="model",
        tqdm_class=None,
    )

    # Heuristic renaming: map common exported names to the exact filenames the CLI expects
    def rename_pair(src_xml: Path, dst_xml: Path) -> None:
        if not src_xml.exists() or src_xml == dst_xml:
            return
        src_bin = src_xml.with_suffix(".bin")
        dst_bin = dst_xml.with_suffix(".bin")
        # Remove any existing destination
        for p in (dst_xml, dst_bin):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        try:
            src_xml.rename(dst_xml)
            if src_bin.exists():
                src_bin.rename(dst_bin)
            print(f"Renamed {src_xml.name} -> {dst_xml.name}")
        except Exception as e:
            print(f"WARNING: Failed to rename {src_xml.name} -> {dst_xml.name}: {e}")

    xmls = list(target_dir.glob("*.xml"))

    # Preprocessor: look for files hinting at mel/preprocess
    if not (target_dir / "parakeet_melspectogram.xml").exists():
        candidates = [p for p in xmls if any(k in p.name.lower() for k in ("melspectro", "preproc", "preprocess"))]
        if len(candidates) == 1:
            rename_pair(candidates[0], target_dir / "parakeet_melspectogram.xml")

    # Encoder
    if not (target_dir / "parakeet_encoder.xml").exists():
        candidates = [p for p in xmls if "encoder" in p.name.lower()]
        if len(candidates) == 1:
            rename_pair(candidates[0], target_dir / "parakeet_encoder.xml")

    # Decoder
    if not (target_dir / "parakeet_decoder.xml").exists():
        candidates = [p for p in xmls if "decoder" in p.name.lower()]
        if len(candidates) == 1:
            rename_pair(candidates[0], target_dir / "parakeet_decoder.xml")

    # Joint
    if not (target_dir / "parakeet_joint.xml").exists():
        candidates = [p for p in xmls if "joint" in p.name.lower() or "decision" in p.name.lower()]
        if len(candidates) == 1:
            rename_pair(candidates[0], target_dir / "parakeet_joint.xml")

    # Vocabulary file: normalize common names
    vocab_aliases = [
        "parakeet_vocab.json",
        "vocab.json",
        "tokenizer.json",
    ]
    if not (target_dir / "parakeet_vocab.json").exists():
        for alias in vocab_aliases[1:]:
            src = target_dir / alias
            if src.exists():
                try:
                    src.rename(target_dir / "parakeet_vocab.json")
                    print(f"Renamed {alias} -> parakeet_vocab.json")
                except Exception as e:
                    print(f"WARNING: Failed to rename {alias} -> parakeet_vocab.json: {e}")
                break

    # Quick sanity checklist of expected files
    expected = [
        "parakeet_melspectogram.xml",
        "parakeet_encoder.xml",
        "parakeet_decoder.xml",
        "parakeet_joint.xml",
        "parakeet_vocab.json",
    ]

    missing = [name for name in expected if not (target_dir / name).exists()]
    if missing:
        print("WARNING: Some expected files not found:")
        for m in missing:
            print(f"  - {m}")
        print("If the repo uses different names, update your code paths or rename accordingly.")
    else:
        print("All expected Parakeet OV files present.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

