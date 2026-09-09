#pragma once

#include "nlms.hpp"
#include "mono_pipeline.hpp"
#include "embedded/audio/audio_types.hpp"
#include <span>
#include <vector>

namespace noiselessx::dsp {

struct DualMicStatus {
    bool nlms_enabled{true};
    bool nlms_active{false};
    bool drift_inhibited{false};
    bool fallback_to_ai_only{false};
    double measured_drift_ms{0.0};
    float nlms_residual_energy_db{-120.0f};
    VadDecision vad;
};

/**
 * @brief Dual-Microphone DSP Pipeline.
 * 
 * Integrates:
 * - Primary (headphone mic) + Reference (error mic)
 * - Drift-checked sample synchronization: Never pairs misaligned samples when
 *   drift exceeds tolerance; automatically flags fallback to AI-only.
 * - Active NLMS adaptive noise cancellation
 * - Single-channel STFT / VAD / iSTFT processing
 */
class DualMicPipeline {
public:
    explicit DualMicPipeline(
        float sample_rate = 16000.0f,
        int filter_length = 256,
        float learning_rate = 0.05f,
        float epsilon = 1e-6f,
        double max_allowable_drift_ms = 10.0,
        float hp_cutoff_hz = 80.0f,
        size_t fft_size = 512,
        size_t hop_size = 80
    );

    /**
     * @brief Process synchronized primary and reference hops.
     * @param primary_hop Input samples from headphone mic
     * @param reference_hop Input samples from error mic
     * @param measured_drift_ms Clock skew reported by Phase 2 DriftDetector
     * @param out_hop Cleaned output audio hop
     * @param status Status reporting NLMS activity, drift guard, and VAD
     * @param spectral_callback Optional callback for AI masking before iSTFT
     */
    void process_dual_hop(
        std::span<const float> primary_hop,
        std::span<const float> reference_hop,
        double measured_drift_ms,
        std::span<float> out_hop,
        DualMicStatus& status,
        const MonoPipeline::SpectralProcessorCallback& spectral_callback = nullptr
    );

    void reset();

    [[nodiscard]] NLMSFilter& get_nlms() noexcept { return nlms_; }
    [[nodiscard]] MonoPipeline& get_mono_pipeline() noexcept { return mono_chain_; }
    [[nodiscard]] bool is_enabled() const noexcept { return enabled_; }
    void set_enabled(bool enabled) noexcept { enabled_ = enabled; }

    [[nodiscard]] double get_max_allowable_drift_ms() const noexcept { return max_allowable_drift_ms_; }
    void set_max_allowable_drift_ms(double ms) noexcept { max_allowable_drift_ms_ = ms; }

private:
    float sample_rate_{16000.0f};
    size_t hop_size_{80};
    double max_allowable_drift_ms_{10.0};
    bool enabled_{true};

    NLMSFilter nlms_;
    MonoPipeline mono_chain_;

    // Preallocated scratch buffers (zero per-frame allocations)
    std::vector<float> nlms_out_scratch_;
};

} // namespace noiselessx::dsp
