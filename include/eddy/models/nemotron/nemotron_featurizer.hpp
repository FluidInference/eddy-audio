// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0
//
// Native C++ log-mel featurizer for the Nemotron streaming model — a drop-in
// replacement for the `nemotron_preprocessor.xml` OpenVINO IR.
//
// It reproduces NeMo's AudioToMelSpectrogramPreprocessor exactly (verified
// against the IR to fp16-storage precision): preemphasis 0.97 -> center pad
// n_fft/2 zeros -> framed STFT (Hann(win_length) centred in n_fft, hop) ->
// power spectrum -> slaney mel filterbank -> log(x + guard). The IR carries no
// per-feature normalisation; frames past the valid audio length are zeroed
// (matching the IR's length mask).

#pragma once

#include <cstddef>
#include <vector>

namespace eddy::nemotron {

class MelFeaturizer {
 public:
  // Nemotron defaults: 16 kHz, 128 mels, 25 ms Hann window (400 samples),
  // 10 ms hop (160), 512-pt FFT, preemphasis 0.97, log guard 6e-8 (the value
  // stored in the exported IR; NeMo's 2^-24 rounds to this at fp16).
  explicit MelFeaturizer(int sample_rate = 16000, int n_mels = 128);

  // Compute log-mel for `n` samples of 16 kHz mono float PCM. `valid_samples`
  // is the number of non-padding samples (frames whose index >= valid_samples/
  // hop are zeroed, mirroring the OV preprocessor). Fills `out_mel` with
  // [n_mels * frames] in bin-major layout (out_mel[bin*frames + t]) and sets
  // `out_frames`.
  void compute(const float* audio, std::size_t n, int valid_samples,
               std::vector<float>& out_mel, std::size_t& out_frames) const;

  int n_mels() const { return n_mels_; }
  int sample_rate() const { return sample_rate_; }

 private:
  int sample_rate_;
  int n_mels_;
  int n_fft_;
  int hop_;
  int win_length_;
  int n_freq_;  // n_fft_/2 + 1
  float preemph_;
  float log_guard_;

  std::vector<float> window_;  // [n_fft_]: Hann(win_length_) centred, 0 elsewhere
  std::vector<float> mel_fb_;  // [n_mels_ * n_freq_], row-major (slaney)

  // Radix-2 FFT precomputed tables (size n_fft_).
  std::vector<int> bitrev_;
  std::vector<float> tw_cos_;  // [n_fft_/2]
  std::vector<float> tw_sin_;  // [n_fft_/2]

  void fft(std::vector<float>& re, std::vector<float>& im) const;
};

}  // namespace eddy::nemotron
