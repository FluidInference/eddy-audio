# Third-Party Licenses

This directory contains license information for third-party dependencies used by eddy.

## Core Dependencies

### NVIDIA Parakeet TDT Models
- **Version**: v2 (0.6b), v3 (1.1b)
- **License**: CC-BY-4.0
- **Source**: [NVIDIA NeMo Parakeet TDT](https://huggingface.co/collections/nvidia/parakeet-tdt-family-6733b7a0df18b25e7689b7b0)

### OpenAI Whisper Model
- **Version**: large-v3-turbo
- **License**: MIT
- **Source**: [OpenAI Whisper](https://github.com/openai/whisper)

### Intel OpenVINO
- **Version**: 2025.0+
- **License**: Apache 2.0
- **Source**: [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)

## Additional Runtime Dependencies

- **libsndfile** (LGPL-2.1+): Audio file I/O - [github.com/libsndfile/libsndfile](https://github.com/libsndfile/libsndfile)
- **libsamplerate** (BSD-2-Clause): Audio resampling - [github.com/libsndfile/libsamplerate](https://github.com/libsndfile/libsamplerate)

## Benchmark Datasets

- **LibriSpeech** (CC-BY-4.0): [OpenSLR](http://www.openslr.org/12)
- **FLEURS** (CC-BY-4.0): [Google Research](https://huggingface.co/datasets/google/fleurs)

## Attribution Requirements

When using eddy, please ensure compliance with:
- CC-BY-4.0 attribution for NVIDIA Parakeet TDT models
- MIT license terms for OpenAI Whisper
- Apache 2.0 license for OpenVINO
- LGPL requirements for libsndfile (if dynamically linked)
