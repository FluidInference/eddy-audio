// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/models/nemotron/nemotron_featurizer.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <stdexcept>
#include <string>

namespace eddy::nemotron {

namespace {

// librosa slaney hz<->mel (htk=False).
inline double hz_to_mel(double hz) {
  const double f_sp = 200.0 / 3.0;
  const double min_log_hz = 1000.0;
  const double min_log_mel = min_log_hz / f_sp;
  const double logstep = std::log(6.4) / 27.0;
  if (hz < min_log_hz) return hz / f_sp;
  return min_log_mel + std::log(hz / min_log_hz) / logstep;
}

inline double mel_to_hz(double mel) {
  const double f_sp = 200.0 / 3.0;
  const double min_log_hz = 1000.0;
  const double min_log_mel = min_log_hz / f_sp;
  const double logstep = std::log(6.4) / 27.0;
  if (mel < min_log_mel) return f_sp * mel;
  return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

}  // namespace

MelFeaturizer::MelFeaturizer(int sample_rate, int n_mels)
    : sample_rate_(sample_rate),
      n_mels_(n_mels),
      n_fft_(512),
      hop_(static_cast<int>(sample_rate * 0.01 + 0.5)),   // 10 ms -> 160
      win_length_(static_cast<int>(sample_rate * 0.025 + 0.5)),  // 25 ms -> 400
      n_freq_(512 / 2 + 1),
      preemph_(0.97f),
      log_guard_(6e-8f) {
  // The featurizer is calibrated for NeMo's 16 kHz Nemotron config (25 ms /
  // 10 ms framing -> win 400 / hop 160, 512-pt FFT). A 512 FFT only fits the
  // window if win_length <= n_fft; guard so an unexpected sample rate fails
  // loudly instead of silently producing garbage mel.
  if (win_length_ > n_fft_) {
    throw std::runtime_error(
        "MelFeaturizer: window length " + std::to_string(win_length_) +
        " exceeds n_fft " + std::to_string(n_fft_) +
        " (sample_rate " + std::to_string(sample_rate_) +
        " unsupported by the 512-pt featurizer).");
  }

  // Hann window (periodic=False) of win_length_, centred in the n_fft_ frame
  // (torch pads (n_fft - win_length)/2 on the left). Zeros elsewhere.
  window_.assign(n_fft_, 0.0f);
  const int off = (n_fft_ - win_length_) / 2;
  for (int i = 0; i < win_length_; ++i) {
    const double w = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / (win_length_ - 1));
    window_[off + i] = static_cast<float>(w);
  }

  // Slaney mel filterbank [n_mels_, n_freq_], norm='slaney', fmin=0, fmax=sr/2.
  const double fmin = 0.0;
  const double fmax = sample_rate_ / 2.0;
  std::vector<double> f_pts(n_mels_ + 2);
  {
    const double mmin = hz_to_mel(fmin);
    const double mmax = hz_to_mel(fmax);
    for (int i = 0; i < n_mels_ + 2; ++i) {
      const double mel = mmin + (mmax - mmin) * i / (n_mels_ + 1);
      f_pts[i] = mel_to_hz(mel);
    }
  }
  std::vector<double> fft_freqs(n_freq_);
  for (int k = 0; k < n_freq_; ++k) {
    fft_freqs[k] = (sample_rate_ / 2.0) * k / (n_freq_ - 1);
  }
  mel_fb_.assign(static_cast<size_t>(n_mels_) * n_freq_, 0.0f);
  for (int i = 0; i < n_mels_; ++i) {
    const double lo = f_pts[i], ce = f_pts[i + 1], hi = f_pts[i + 2];
    const double enorm = 2.0 / (hi - lo);  // slaney normalization
    for (int k = 0; k < n_freq_; ++k) {
      const double left = (fft_freqs[k] - lo) / (ce - lo);
      const double right = (hi - fft_freqs[k]) / (hi - ce);
      double v = std::min(left, right);
      if (v < 0.0) v = 0.0;
      mel_fb_[static_cast<size_t>(i) * n_freq_ + k] = static_cast<float>(v * enorm);
    }
  }

  // Radix-2 FFT tables: bit-reversal permutation + twiddle factors.
  bitrev_.resize(n_fft_);
  int log2n = 0;
  while ((1 << log2n) < n_fft_) ++log2n;
  for (int i = 0; i < n_fft_; ++i) {
    int r = 0;
    for (int b = 0; b < log2n; ++b)
      if (i & (1 << b)) r |= 1 << (log2n - 1 - b);
    bitrev_[i] = r;
  }
  tw_cos_.resize(n_fft_ / 2);
  tw_sin_.resize(n_fft_ / 2);
  for (int i = 0; i < n_fft_ / 2; ++i) {
    const double ang = -2.0 * M_PI * i / n_fft_;
    tw_cos_[i] = static_cast<float>(std::cos(ang));
    tw_sin_[i] = static_cast<float>(std::sin(ang));
  }
}

// In-place iterative radix-2 Cooley-Tukey FFT, size n_fft_.
void MelFeaturizer::fft(std::vector<float>& re, std::vector<float>& im) const {
  const int n = n_fft_;
  for (int i = 0; i < n; ++i) {
    const int j = bitrev_[i];
    if (j > i) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    const int half = len >> 1;
    const int step = n / len;  // twiddle stride
    for (int base = 0; base < n; base += len) {
      for (int k = 0; k < half; ++k) {
        const float wc = tw_cos_[k * step];
        const float ws = tw_sin_[k * step];
        const int a = base + k;
        const int b = base + k + half;
        const float br = re[b] * wc - im[b] * ws;
        const float bi = re[b] * ws + im[b] * wc;
        re[b] = re[a] - br;
        im[b] = im[a] - bi;
        re[a] += br;
        im[a] += bi;
      }
    }
  }
}

void MelFeaturizer::compute(const float* audio, std::size_t n, int valid_samples,
                            std::vector<float>& out_mel, std::size_t& out_frames) const {
  // Preemphasis: y[0]=x[0]; y[i]=x[i]-0.97*x[i-1].
  std::vector<float> y(n);
  if (n > 0) y[0] = audio[0];
  for (std::size_t i = 1; i < n; ++i) y[i] = audio[i] - preemph_ * audio[i - 1];

  // Center pad n_fft/2 zeros each side, then frame with hop. The padded length
  // is n + n_fft; #frames = 1 + (padded - n_fft)/hop = 1 + n/hop.
  const int pad = n_fft_ / 2;
  const std::size_t frames = 1 + n / static_cast<std::size_t>(hop_);
  out_frames = frames;
  out_mel.assign(static_cast<size_t>(n_mels_) * frames, 0.0f);

  // Frames at index >= valid_samples/hop are zeroed. This intentionally has no
  // "+1" (unlike `frames` above): it mirrors the OV preprocessor's length mask,
  // whose mel_length = audio_length/hop (verified — for a full chunk it zeroes
  // exactly the trailing frame, which the encoder-input assembly then trims).
  const std::size_t valid_frames =
      static_cast<std::size_t>(valid_samples) / static_cast<std::size_t>(hop_);

  std::vector<float> re(n_fft_), im(n_fft_), power(n_freq_);
  for (std::size_t f = 0; f < frames; ++f) {
    if (f >= valid_frames) continue;  // length mask: zero (already zeroed)

    // Frame covers padded[f*hop : f*hop+n_fft]; padded index j maps to y[j-pad].
    // ptrdiff_t (not long, which is 32-bit on Win64) to avoid overflow on long audio.
    const std::ptrdiff_t start = static_cast<std::ptrdiff_t>(f * static_cast<std::size_t>(hop_)) - pad;
    for (int t = 0; t < n_fft_; ++t) {
      const std::ptrdiff_t src = start + t;
      const float s = (src >= 0 && src < static_cast<std::ptrdiff_t>(n)) ? y[static_cast<std::size_t>(src)] : 0.0f;
      re[t] = s * window_[t];
      im[t] = 0.0f;
    }

    fft(re, im);
    for (int k = 0; k < n_freq_; ++k) power[k] = re[k] * re[k] + im[k] * im[k];

    // mel = mel_fb (n_mels x n_freq) @ power; then log(mel + guard).
    for (int mbin = 0; mbin < n_mels_; ++mbin) {
      const float* row = &mel_fb_[static_cast<size_t>(mbin) * n_freq_];
      float acc = 0.0f;
      for (int k = 0; k < n_freq_; ++k) acc += row[k] * power[k];
      out_mel[static_cast<size_t>(mbin) * frames + f] = std::log(acc + log_guard_);
    }
  }
}

}  // namespace eddy::nemotron
