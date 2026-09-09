#pragma once

#include <cstddef>
#include <cmath>
#include <span>

namespace noiselessx::dsp {

/**
 * @brief Direct Form II Transposed Biquad Filter.
 * 
 * Implements standard 2nd-order IIR filtering with Butterworth response.
 * Ideal for high-pass filtering (rumble / handling noise suppression).
 * Zero heap allocation.
 */
class BiquadFilter {
public:
    enum class FilterType {
        HIGH_PASS,
        LOW_PASS,
        BAND_PASS
    };

    explicit BiquadFilter(float sample_rate = 16000.0f, float cutoff_hz = 80.0f, float q = 0.70710678f) {
        configure_high_pass(sample_rate, cutoff_hz, q);
        reset();
    }

    /**
     * @brief Configure filter as 2nd-order Butterworth High-Pass.
     * @param sample_rate Audio sampling rate in Hz (e.g. 16000)
     * @param cutoff_hz Cutoff frequency in Hz (-3dB point, e.g. 80.0f)
     * @param q Quality factor (0.7071 for maximally flat Butterworth)
     */
    void configure_high_pass(float sample_rate, float cutoff_hz, float q = 0.70710678f) {
        sample_rate_ = sample_rate;
        cutoff_hz_ = cutoff_hz;
        q_ = q;

        const float pi = 3.14159265358979323846f;
        const float omega0 = 2.0f * pi * (cutoff_hz / sample_rate);
        const float cos_w0 = std::cos(omega0);
        const float sin_w0 = std::sin(omega0);
        const float alpha = sin_w0 / (2.0f * q);

        const float b0 = (1.0f + cos_w0) * 0.5f;
        const float b1 = -(1.0f + cos_w0);
        const float b2 = (1.0f + cos_w0) * 0.5f;
        const float a0 = 1.0f + alpha;
        const float a1 = -2.0f * cos_w0;
        const float a2 = 1.0f - alpha;

        // Normalize coefficients by a0
        b0_ = b0 / a0;
        b1_ = b1 / a0;
        b2_ = b2 / a0;
        a1_ = a1 / a0;
        a2_ = a2 / a0;
    }

    /**
     * @brief Process a single sample using Direct Form II Transposed.
     */
    [[nodiscard]] inline float process_sample(float in) noexcept {
        const float out = b0_ * in + s1_;
        s1_ = b1_ * in - a1_ * out + s2_;
        s2_ = b2_ * in - a2_ * out;
        return out;
    }

    /**
     * @brief In-place process an array of samples.
     */
    void process(float* buffer, size_t count) noexcept {
        if (!buffer || count == 0) return;
        for (size_t i = 0; i < count; ++i) {
            buffer[i] = process_sample(buffer[i]);
        }
    }

    /**
     * @brief In-place process a span.
     */
    void process(std::span<float> buffer) noexcept {
        process(buffer.data(), buffer.size());
    }

    void reset() noexcept {
        s1_ = 0.0f;
        s2_ = 0.0f;
    }

    [[nodiscard]] float get_cutoff_hz() const noexcept { return cutoff_hz_; }
    [[nodiscard]] float get_sample_rate() const noexcept { return sample_rate_; }

private:
    float sample_rate_{16000.0f};
    float cutoff_hz_{80.0f};
    float q_{0.70710678f};

    // Filter normalized coefficients
    float b0_{1.0f};
    float b1_{0.0f};
    float b2_{0.0f};
    float a1_{0.0f};
    float a2_{0.0f};

    // Direct Form II Transposed delay states
    float s1_{0.0f};
    float s2_{0.0f};
};

} // namespace noiselessx::dsp
