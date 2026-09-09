#pragma once

#include <cstddef>
#include <cmath>
#include <span>

namespace noiselessx::dsp {

/**
 * @brief Real-time DC Offset Removal using running mean subtraction.
 * 
 * y[n] = x[n] - mean[n]
 * mean[n] = (1 - alpha) * mean[n-1] + alpha * x[n]
 * 
 * Guarantees zero memory allocation inside per-frame processing.
 */
class DcBlocker {
public:
    explicit DcBlocker(float sample_rate = 16000.0f, float cutoff_hz = 5.0f) {
        set_cutoff(sample_rate, cutoff_hz);
        reset();
    }

    void set_cutoff(float sample_rate, float cutoff_hz) {
        sample_rate_ = sample_rate;
        cutoff_hz_ = cutoff_hz;
        // alpha = 1 - exp(-2 * pi * cutoff / sample_rate)
        if (sample_rate_ > 0.0f && cutoff_hz_ > 0.0f) {
            alpha_ = 1.0f - std::exp(-2.0f * 3.14159265358979323846f * cutoff_hz_ / sample_rate_);
        } else {
            alpha_ = 0.001f;
        }
    }

    /**
     * @brief Process a single sample through the DC blocker.
     */
    [[nodiscard]] inline float process_sample(float sample) noexcept {
        running_mean_ = (1.0f - alpha_) * running_mean_ + alpha_ * sample;
        return sample - running_mean_;
    }

    /**
     * @brief In-place processing of an audio buffer.
     */
    void process(float* buffer, size_t count) noexcept {
        if (!buffer || count == 0) return;
        for (size_t i = 0; i < count; ++i) {
            buffer[i] = process_sample(buffer[i]);
        }
    }

    /**
     * @brief In-place processing of a span.
     */
    void process(std::span<float> buffer) noexcept {
        process(buffer.data(), buffer.size());
    }

    /**
     * @brief Reset running mean state.
     */
    void reset() noexcept {
        running_mean_ = 0.0f;
    }

    [[nodiscard]] float get_running_mean() const noexcept {
        return running_mean_;
    }

private:
    float sample_rate_{16000.0f};
    float cutoff_hz_{5.0f};
    float alpha_{0.00196f};
    float running_mean_{0.0f};
};

} // namespace noiselessx::dsp
