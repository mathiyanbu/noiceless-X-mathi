#pragma once

#if defined(__has_include)
  #if __has_include(<cstddef>)
    #include <cstddef>
  #endif
  #if __has_include(<cstdint>)
    #include <cstdint>
  #endif
  #if __has_include(<cmath>)
    #include <cmath>
  #endif
  #if __has_include(<algorithm>)
    #include <algorithm>
  #endif
#else
  #include <cstddef>
  #include <cstdint>
  #include <cmath>
  #include <algorithm>
#endif

// Standard C headers guaranteed across all compiler environments
#include <stddef.h>
#include <stdint.h>
#include <math.h>

namespace noiselessx {
namespace dsp {

namespace detail {
    template <typename T>
    constexpr const T& max_val(const T& a, const T& b) noexcept {
        return (a < b) ? b : a;
    }

    template <typename T>
    constexpr const T& min_val(const T& a, const T& b) noexcept {
        return (b < a) ? b : a;
    }

    template <typename T>
    constexpr const T& clamp_val(const T& val, const T& low, const T& high) noexcept {
        return (val < low) ? low : (high < val) ? high : val;
    }

    inline float sqrt_f(float x) noexcept {
#if defined(__has_include) && __has_include(<cmath>)
        return std::sqrt(x);
#else
        return ::sqrtf(x);
#endif
    }

    inline float log10_f(float x) noexcept {
#if defined(__has_include) && __has_include(<cmath>)
        return std::log10(x);
#else
        return ::log10f(x);
#endif
    }

    inline double log10_d(double x) noexcept {
#if defined(__has_include) && __has_include(<cmath>)
        return std::log10(x);
#else
        return ::log10(x);
#endif
    }
} // namespace detail

/**
 * @brief Live telemetry metrics computed by the real-time noise-floor tracker.
 */
struct LiveSnrMetrics {
    float estimated_input_snr_db{0.0f};
    float estimated_output_snr_db{0.0f};
    float estimated_snr_improvement_db{0.0f};
    float primary_level_dbfs{-96.0f};
    float reference_level_dbfs{-96.0f};
    float output_level_dbfs{-96.0f};
    bool is_estimated{true};
};

/**
 * @brief Real-time Noise-Floor Tracker & Live SNR Estimator.
 * 
 * In operational dual-mic deployment, ground-truth clean speech is physically unavailable.
 * This class monitors frame powers and recursively updates noise-floor variance during
 * non-speech/VAD-negative frames (p_vad < vad_threshold), computing live estimated SNR
 * and calibrated dBFS levels without any dynamic allocations (SCHED_FIFO safe).
 */
class SnrTracker {
public:
    explicit SnrTracker(
        uint32_t sample_rate = 16000,
        float vad_threshold = 0.30f,
        float noise_alpha = 0.95f,
        float signal_alpha = 0.85f,
        float eps = 1e-10f
    ) : sample_rate_(sample_rate),
        vad_threshold_(vad_threshold),
        noise_alpha_(noise_alpha),
        signal_alpha_(signal_alpha),
        eps_(eps) {
        reset();
    }

    void reset() noexcept {
        p_noise_in_ = 1e-4f;
        p_noise_out_ = 1e-4f;
        p_sig_in_ = 1e-4f;
        p_sig_out_ = 1e-4f;

        metrics_.estimated_input_snr_db = 0.0f;
        metrics_.estimated_output_snr_db = 0.0f;
        metrics_.estimated_snr_improvement_db = 0.0f;
        metrics_.primary_level_dbfs = -96.0f;
        metrics_.reference_level_dbfs = -96.0f;
        metrics_.output_level_dbfs = -96.0f;
        metrics_.is_estimated = true;
    }

    /**
     * @brief Process audio hop and update live estimated SNR and level meters.
     */
    LiveSnrMetrics update(
        const float* primary,
        size_t n_primary,
        const float* output,
        size_t n_output,
        float vad_prob,
        const float* reference = nullptr,
        size_t n_ref = 0
    ) noexcept {
        if (n_primary == 0 || n_output == 0 || primary == nullptr || output == nullptr) {
            return metrics_;
        }

        // 1. Calculate frame mean squared power
        float p_frame_in = 0.0f;
        for (size_t i = 0; i < n_primary; ++i) {
            p_frame_in += primary[i] * primary[i];
        }
        p_frame_in /= static_cast<float>(n_primary);

        float p_frame_out = 0.0f;
        for (size_t i = 0; i < n_output; ++i) {
            p_frame_out += output[i] * output[i];
        }
        p_frame_out /= static_cast<float>(n_output);

        float p_frame_ref = 0.0f;
        if (reference != nullptr && n_ref > 0) {
            for (size_t i = 0; i < n_ref; ++i) {
                p_frame_ref += reference[i] * reference[i];
            }
            p_frame_ref /= static_cast<float>(n_ref);
        }

        // 2. dBFS Level meters (20 * log10(RMS))
        const float rms_in = detail::sqrt_f(detail::max_val(p_frame_in, eps_));
        const float rms_out = detail::sqrt_f(detail::max_val(p_frame_out, eps_));
        const float rms_ref = detail::sqrt_f(detail::max_val(p_frame_ref, eps_));

        metrics_.primary_level_dbfs = detail::clamp_val(20.0f * detail::log10_f(rms_in), -96.0f, 0.0f);
        metrics_.output_level_dbfs = detail::clamp_val(20.0f * detail::log10_f(rms_out), -96.0f, 0.0f);
        metrics_.reference_level_dbfs = detail::clamp_val(20.0f * detail::log10_f(rms_ref), -96.0f, 0.0f);

        // 3. Update overall smoothed signal power
        p_sig_in_ = signal_alpha_ * p_sig_in_ + (1.0f - signal_alpha_) * p_frame_in;
        p_sig_out_ = signal_alpha_ * p_sig_out_ + (1.0f - signal_alpha_) * p_frame_out;

        // 4. Noise floor tracking during VAD-negative frames (non-speech)
        if (vad_prob < vad_threshold_) {
            p_noise_in_ = noise_alpha_ * p_noise_in_ + (1.0f - noise_alpha_) * p_frame_in;
            p_noise_out_ = noise_alpha_ * p_noise_out_ + (1.0f - noise_alpha_) * p_frame_out;
        } else {
            // Slow minimum tracking during speech to prevent divergence
            if (p_frame_in < p_noise_in_) {
                p_noise_in_ = 0.90f * p_noise_in_ + 0.10f * p_frame_in;
            }
            if (p_frame_out < p_noise_out_) {
                p_noise_out_ = 0.90f * p_noise_out_ + 0.10f * p_frame_out;
            }
        }

        p_noise_in_ = detail::max_val(p_noise_in_, eps_);
        p_noise_out_ = detail::max_val(p_noise_out_, eps_);

        // 5. Estimated SNR computation
        const float s_in_est = detail::max_val(p_sig_in_ - p_noise_in_, 1e-6f * p_noise_in_);
        const float s_out_est = detail::max_val(p_sig_out_ - p_noise_out_, 1e-6f * p_noise_out_);

        const float snr_in = 10.0f * detail::log10_f(s_in_est / p_noise_in_);
        const float snr_out = 10.0f * detail::log10_f(s_out_est / p_noise_out_);

        metrics_.estimated_input_snr_db = detail::clamp_val(snr_in, -20.0f, 45.0f);
        metrics_.estimated_output_snr_db = detail::clamp_val(snr_out, -20.0f, 45.0f);
        metrics_.estimated_snr_improvement_db = detail::clamp_val(metrics_.estimated_output_snr_db - metrics_.estimated_input_snr_db, -10.0f, 35.0f);
        metrics_.is_estimated = true;

        return metrics_;
    }

    [[nodiscard]] const LiveSnrMetrics& get_metrics() const noexcept {
        return metrics_;
    }

    /**
     * @brief Ground-truth SNR calculation for offline validation tests where clean s is known:
     *   SNR_in  = 10*log10( Σs^2 / (Σ(x-s)^2 + eps) )
     *   SNR_out = 10*log10( Σs^2 / (Σ(s_hat-s)^2 + eps) )
     *   ΔSNR = SNR_out - SNR_in
     */
    static void compute_true_snr(
        const float* clean,
        const float* noisy,
        const float* enhanced,
        size_t n,
        float& snr_in,
        float& snr_out,
        float& delta_snr,
        float eps = 1e-10f
    ) noexcept {
        if (n == 0 || clean == nullptr || noisy == nullptr || enhanced == nullptr) {
            snr_in = 0.0f;
            snr_out = 0.0f;
            delta_snr = 0.0f;
            return;
        }

        double s_power = 0.0;
        double in_noise_power = 0.0;
        double out_noise_power = 0.0;

        for (size_t i = 0; i < n; ++i) {
            const double s = clean[i];
            const double x = noisy[i];
            const double s_hat = enhanced[i];

            s_power += s * s;
            in_noise_power += (x - s) * (x - s);
            out_noise_power += (s_hat - s) * (s_hat - s);
        }

        in_noise_power += eps;
        out_noise_power += eps;

        snr_in = static_cast<float>(10.0 * detail::log10_d((s_power + eps) / in_noise_power));
        snr_out = static_cast<float>(10.0 * detail::log10_d((s_power + eps) / out_noise_power));
        delta_snr = snr_out - snr_in;
    }

private:
    uint32_t sample_rate_{16000};
    float vad_threshold_{0.30f};
    float noise_alpha_{0.95f};
    float signal_alpha_{0.85f};
    float eps_{1e-10f};

    float p_noise_in_{1e-4f};
    float p_noise_out_{1e-4f};
    float p_sig_in_{1e-4f};
    float p_sig_out_{1e-4f};

    LiveSnrMetrics metrics_;
};

} // namespace dsp
} // namespace noiselessx
