# Third-Party Licenses

This directory contains license information for third-party dependencies used by eddy.

## Models

### Parakeet TDT (v2 and v3)
- **License**: CC-BY-4.0
- **Source**: [NVIDIA NeMo](https://huggingface.co/collections/nvidia/parakeet-tdt-family-6733b7a0df18b25e7689b7b0)
- **Description**: Automatic speech recognition models based on FastConformer-RNNT architecture
- **Models**:
  - [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) - English only
  - [parakeet-tdt-1.1b](https://huggingface.co/nvidia/parakeet-tdt-1.1b) - Multilingual (24 languages)

## Runtime Dependencies

### OpenVINO Toolkit
- **License**: Apache 2.0
- **Source**: [Intel OpenVINO](https://github.com/openvinotoolkit/openvino)
- **Description**: Cross-platform inference runtime for AI models

### libsndfile
- **License**: LGPL-2.1 or LGPL-3.0
- **Source**: [libsndfile](https://github.com/libsndfile/libsndfile)
- **Description**: Library for reading and writing audio files (WAV, FLAC, OGG, etc.)

### libsamplerate
- **License**: BSD-2-Clause
- **Source**: [libsamplerate](https://github.com/libsndfile/libsamplerate)
- **Description**: High-quality audio sample rate conversion library

## Build Dependencies

### vcpkg
- **License**: MIT
- **Source**: [Microsoft vcpkg](https://github.com/microsoft/vcpkg)
- **Description**: C++ package manager for dependency management

### CMake
- **License**: BSD-3-Clause
- **Source**: [CMake](https://cmake.org/)
- **Description**: Cross-platform build system generator

## Benchmark Datasets

### LibriSpeech
- **License**: CC-BY-4.0
- **Source**: [OpenSLR](http://www.openslr.org/12)
- **Description**: Large-scale English speech corpus for ASR evaluation
- **Citation**: Panayotov et al., "Librispeech: an ASR corpus based on public domain audio books," ICASSP 2015

### FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)
- **License**: CC-BY-4.0
- **Source**: [Google Research](https://huggingface.co/datasets/google/fleurs)
- **Description**: Multilingual speech corpus covering 102 languages
- **Citation**: Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech," SLT 2022

## License Texts

Full license texts for each dependency can be found in their respective subdirectories or source repositories linked above.

## Attribution Requirements

When using eddy, please ensure compliance with:
- CC-BY-4.0 attribution requirements for Parakeet TDT models
- LGPL requirements for libsndfile (if dynamically linked)
- Other dependency license terms as applicable to your use case
