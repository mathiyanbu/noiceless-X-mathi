#pragma once

#include <array>
#include <cstddef>
#include <cmath>

namespace noiselessx {
namespace audio {

class Decimator3 {
public:
    Decimator3() { reset(); }

    void reset() {
        history_.fill(0.0f);
        history_index_ = 0;
        phase_ = 0;
        initialize_coefficients();
    }

    size_t process(const float* input, size_t count, float* output, size_t capacity) {
        size_t produced = 0;
        for (size_t i = 0; i < count; ++i) {
            history_[history_index_] = input[i];
            history_index_ = (history_index_ + 1) % history_.size();

            if (++phase_ == 3) {
                phase_ = 0;
                if (produced < capacity) {
                    float value = 0.0f;
                    for (size_t tap = 0; tap < history_.size(); ++tap) {
                        const size_t index = (history_index_ + tap) % history_.size();
                        value += coefficients_[tap] * history_[index];
                    }
                    output[produced++] = value;
                }
            }
        }
        return produced;
    }

private:
    void initialize_coefficients() {
        constexpr double cutoff = 1.0 / 6.0;
        constexpr double center = 7.0;
        double sum = 0.0;
        for (size_t tap = 0; tap < coefficients_.size(); ++tap) {
            const double x = static_cast<double>(tap) - center;
            const double sinc = (std::abs(x) < 1e-12)
                ? 1.0
                : std::sin(2.0 * 3.14159265358979323846 * cutoff * x)
                    / (2.0 * 3.14159265358979323846 * cutoff * x);
            const double window = 0.54 - 0.46 * std::cos(
                2.0 * 3.14159265358979323846 * static_cast<double>(tap) / 14.0);
            coefficients_[tap] = static_cast<float>(2.0 * cutoff * sinc * window);
            sum += coefficients_[tap];
        }
        for (float& coefficient : coefficients_) {
            coefficient = static_cast<float>(coefficient / sum);
        }
    }

    std::array<float, 15> history_{};
    std::array<float, 15> coefficients_{};
    size_t history_index_{0};
    size_t phase_{0};
};

} // namespace audio
} // namespace noiselessx
