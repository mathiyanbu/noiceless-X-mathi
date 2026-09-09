#include "stft.hpp"
#include "pffft/pffft.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace noiselessx {
namespace dsp {

StftEngine::StftEngine(size_t fft_size, size_t hop_size) {
    initialize(fft_size, hop_size);
}

StftEngine::~StftEngine() {
    cleanup();
}

StftEngine::StftEngine(StftEngine&& other) noexcept {
    *this = std::move(other);
}

StftEngine& StftEngine::operator=(StftEngine&& other) noexcept {
    if (this != &other) {
        cleanup();

        fft_size_ = other.fft_size_;
        hop_size_ = other.hop_size_;
        num_bins_ = other.num_bins_;
        pffft_setup_ = other.pffft_setup_;
        window_ = other.window_;
        fft_in_ = other.fft_in_;
        fft_out_ = other.fft_out_;
        fft_work_ = other.fft_work_;

        history_buffer_ = std::move(other.history_buffer_);
        accum_audio_buffer_ = std::move(other.accum_audio_buffer_);
        accum_weight_buffer_ = std::move(other.accum_weight_buffer_);

        other.pffft_setup_ = nullptr;
        other.window_ = nullptr;
        other.fft_in_ = nullptr;
        other.fft_out_ = nullptr;
        other.fft_work_ = nullptr;
    }
    return *this;
}

void StftEngine::cleanup() {
    if (pffft_setup_) {
        pffft_destroy_setup(pffft_setup_);
        pffft_setup_ = nullptr;
    }
    if (window_) { pffft_aligned_free(window_); window_ = nullptr; }
    if (fft_in_) { pffft_aligned_free(fft_in_); fft_in_ = nullptr; }
    if (fft_out_) { pffft_aligned_free(fft_out_); fft_out_ = nullptr; }
    if (fft_work_) { pffft_aligned_free(fft_work_); fft_work_ = nullptr; }
}

void StftEngine::initialize(size_t fft_size, size_t hop_size) {
    cleanup();

    fft_size_ = fft_size;
    hop_size_ = hop_size;
    num_bins_ = (fft_size / 2) + 1;

    pffft_setup_ = pffft_new_setup(static_cast<int>(fft_size_), PFFFT_REAL);
    if (!pffft_setup_) {
        throw std::runtime_error("[StftEngine] Failed to create PFFFT setup for size " + std::to_string(fft_size_));
    }

    // Allocate SIMD-aligned buffers (16-byte aligned)
    window_ = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
    fft_in_ = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
    fft_out_ = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));
    fft_work_ = static_cast<float*>(pffft_aligned_malloc(fft_size_ * sizeof(float)));

    // Compute periodic Hann window
    compute_window();

    // Initialize state buffers
    history_buffer_.assign(fft_size_, 0.0f);
    const size_t accum_size = fft_size_ + hop_size_ * 4;
    accum_audio_buffer_.assign(accum_size, 0.0f);
    accum_weight_buffer_.assign(accum_size, 0.0f);
}

void StftEngine::compute_window() {
    const double pi = 3.14159265358979323846;
    for (size_t n = 0; n < fft_size_; ++n) {
        // Periodic Hann window: 0.5 * (1 - cos(2*pi*n / N))
        window_[n] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * pi * static_cast<double>(n) / static_cast<double>(fft_size_))));
    }
}

void StftEngine::process_analysis(std::span<const float> hop_in, SpectralFrame& out_frame) {
    // 1. Shift history buffer by hop_size_ and insert new samples
    if (hop_size_ < fft_size_) {
        std::memmove(history_buffer_.data(), history_buffer_.data() + hop_size_,
                     (fft_size_ - hop_size_) * sizeof(float));
    }
    const size_t copy_count = std::min(hop_in.size(), hop_size_);
    std::memcpy(history_buffer_.data() + (fft_size_ - hop_size_), hop_in.data(), copy_count * sizeof(float));

    // 2. Apply analysis window
    for (size_t n = 0; n < fft_size_; ++n) {
        fft_in_[n] = history_buffer_[n] * window_[n];
    }

    // 3. Real FFT transform with ordered complex output
    pffft_transform_ordered(pffft_setup_, fft_in_, fft_out_, fft_work_, PFFFT_FORWARD);

    // 4. Ensure output frame storage is allocated
    if (out_frame.magnitude.size() != num_bins_) {
        out_frame.magnitude.resize(num_bins_);
        out_frame.phase.resize(num_bins_);
        out_frame.real.resize(num_bins_);
        out_frame.imag.resize(num_bins_);
        out_frame.num_bins = num_bins_;
    }

    // 5. Unpack ordered format into canonical bins
    // k = 0 (DC)
    out_frame.real[0] = fft_out_[0];
    out_frame.imag[0] = 0.0f;
    out_frame.magnitude[0] = std::abs(fft_out_[0]);
    out_frame.phase[0] = (fft_out_[0] >= 0.0f) ? 0.0f : -3.14159265f;

    // k = Nyquist (N/2)
    const size_t nyquist_bin = num_bins_ - 1;
    out_frame.real[nyquist_bin] = fft_out_[1];
    out_frame.imag[nyquist_bin] = 0.0f;
    out_frame.magnitude[nyquist_bin] = std::abs(fft_out_[1]);
    out_frame.phase[nyquist_bin] = (fft_out_[1] >= 0.0f) ? 0.0f : -3.14159265f;

    // k in [1, N/2 - 1]
    for (size_t k = 1; k < nyquist_bin; ++k) {
        const float re = fft_out_[2 * k];
        const float im = fft_out_[2 * k + 1];
        out_frame.real[k] = re;
        out_frame.imag[k] = im;
        out_frame.magnitude[k] = std::sqrt(re * re + im * im);
        out_frame.phase[k] = std::atan2(im, re);
    }
}

void StftEngine::process_synthesis(
    std::span<const float> magnitude,
    std::span<const float> phase,
    std::span<float> hop_out
) {
    // Convert polar (magnitude, phase) to Cartesian (real, imag) in fft_in_
    // DC (k = 0)
    fft_in_[0] = magnitude[0] * std::cos(phase[0]);
    // Nyquist (k = N/2)
    const size_t nyquist_bin = num_bins_ - 1;
    fft_in_[1] = magnitude[nyquist_bin] * std::cos(phase[nyquist_bin]);

    // Bins 1 .. N/2 - 1
    for (size_t k = 1; k < nyquist_bin; ++k) {
        const float m = magnitude[k];
        const float p = phase[k];
        fft_in_[2 * k]     = m * std::cos(p);
        fft_in_[2 * k + 1] = m * std::sin(p);
    }

    // Perform inverse transform
    pffft_transform_ordered(pffft_setup_, fft_in_, fft_out_, fft_work_, PFFFT_BACKWARD);

    // Scale by 1/N
    const float inv_n = 1.0f / static_cast<float>(fft_size_);
    for (size_t n = 0; n < fft_size_; ++n) {
        fft_out_[n] *= inv_n;
    }

    // Overlap-Add with weighted synthesis window
    for (size_t n = 0; n < fft_size_; ++n) {
        accum_audio_buffer_[n]  += fft_out_[n] * window_[n];
        accum_weight_buffer_[n] += window_[n] * window_[n];
    }

    // Extract normalized output hop
    const size_t out_count = std::min(hop_out.size(), hop_size_);
    for (size_t i = 0; i < out_count; ++i) {
        const float w = accum_weight_buffer_[i];
        const float norm = (w > 1e-6f) ? w : 1.0f;
        hop_out[i] = accum_audio_buffer_[i] / norm;
    }

    // Shift accumulation buffers by hop_size_
    const size_t shift_len = accum_audio_buffer_.size() - hop_size_;
    std::memmove(accum_audio_buffer_.data(), accum_audio_buffer_.data() + hop_size_, shift_len * sizeof(float));
    std::memmove(accum_weight_buffer_.data(), accum_weight_buffer_.data() + hop_size_, shift_len * sizeof(float));

    // Zero the newly cleared tail
    std::memset(accum_audio_buffer_.data() + shift_len, 0, hop_size_ * sizeof(float));
    std::memset(accum_weight_buffer_.data() + shift_len, 0, hop_size_ * sizeof(float));
}

void StftEngine::process_synthesis_complex(
    std::span<const float> real,
    std::span<const float> imag,
    std::span<float> hop_out
) {
    fft_in_[0] = real[0];
    const size_t nyquist_bin = num_bins_ - 1;
    fft_in_[1] = real[nyquist_bin];

    for (size_t k = 1; k < nyquist_bin; ++k) {
        fft_in_[2 * k]     = real[k];
        fft_in_[2 * k + 1] = imag[k];
    }

    pffft_transform_ordered(pffft_setup_, fft_in_, fft_out_, fft_work_, PFFFT_BACKWARD);

    const float inv_n = 1.0f / static_cast<float>(fft_size_);
    for (size_t n = 0; n < fft_size_; ++n) {
        fft_out_[n] *= inv_n;
    }

    for (size_t n = 0; n < fft_size_; ++n) {
        accum_audio_buffer_[n]  += fft_out_[n] * window_[n];
        accum_weight_buffer_[n] += window_[n] * window_[n];
    }

    const size_t out_count = std::min(hop_out.size(), hop_size_);
    for (size_t i = 0; i < out_count; ++i) {
        const float w = accum_weight_buffer_[i];
        const float norm = (w > 1e-6f) ? w : 1.0f;
        hop_out[i] = accum_audio_buffer_[i] / norm;
    }

    const size_t shift_len = accum_audio_buffer_.size() - hop_size_;
    std::memmove(accum_audio_buffer_.data(), accum_audio_buffer_.data() + hop_size_, shift_len * sizeof(float));
    std::memmove(accum_weight_buffer_.data(), accum_weight_buffer_.data() + hop_size_, shift_len * sizeof(float));
    std::memset(accum_audio_buffer_.data() + shift_len, 0, hop_size_ * sizeof(float));
    std::memset(accum_weight_buffer_.data() + shift_len, 0, hop_size_ * sizeof(float));
}

void StftEngine::reset() {
    std::fill(history_buffer_.begin(), history_buffer_.end(), 0.0f);
    std::fill(accum_audio_buffer_.begin(), accum_audio_buffer_.end(), 0.0f);
    std::fill(accum_weight_buffer_.begin(), accum_weight_buffer_.end(), 0.0f);
}

} // namespace dsp
} // namespace noiselessx
