#include "dual_mic_pipeline.hpp"
#include <cmath>
#include <algorithm>

namespace noiselessx {
namespace dsp {

DualMicPipeline::DualMicPipeline(
    float sample_rate,
    int filter_length,
    float learning_rate,
    float epsilon,
    double max_allowable_drift_ms,
    float hp_cutoff_hz,
    size_t fft_size,
    size_t hop_size
) : sample_rate_(sample_rate),
    hop_size_(hop_size),
    max_allowable_drift_ms_(max_allowable_drift_ms),
    enabled_(true),
    nlms_(filter_length, learning_rate, epsilon),
    mono_chain_(sample_rate, hp_cutoff_hz, fft_size, hop_size) {
    
    nlms_out_scratch_.resize(hop_size_);
}

void DualMicPipeline::reset() {
    nlms_.reset();
    mono_chain_.reset();
}

void DualMicPipeline::process_dual_hop(
    std::span<const float> primary_hop,
    std::span<const float> reference_hop,
    double measured_drift_ms,
    std::span<float> out_hop,
    DualMicStatus& status,
    const MonoPipeline::SpectralProcessorCallback& spectral_callback
) {
    const size_t count = std::min({primary_hop.size(), reference_hop.size(), hop_size_});
    status.measured_drift_ms = measured_drift_ms;
    status.nlms_enabled = enabled_;

    // 1. Check Drift-Checked Synchronization (Phase 2 DriftDetector)
    const bool drift_exceeded = std::abs(measured_drift_ms) > max_allowable_drift_ms_;
    status.drift_inhibited = drift_exceeded;

    if (enabled_ && !drift_exceeded && reference_hop.data()) {
        // Normal Dual-Microphone Operation: Samples are synchronized
        nlms_.process_block(primary_hop.data(), reference_hop.data(), nlms_out_scratch_.data(), count);
        status.nlms_active = true;
        status.fallback_to_ai_only = false;

        // Calculate residual energy
        double sum_sq = 0.0;
        for (size_t i = 0; i < count; ++i) {
            sum_sq += static_cast<double>(nlms_out_scratch_[i]) * static_cast<double>(nlms_out_scratch_[i]);
        }
        float mean_sq = static_cast<float>(sum_sq / static_cast<double>(count));
        status.nlms_residual_energy_db = (mean_sq > 1e-12f) ? 10.0f * std::log10(mean_sq) : -120.0f;

        // Feed NLMS error signal to STFT / VAD / iSTFT
        mono_chain_.process_hop(
            std::span<const float>(nlms_out_scratch_.data(), count),
            out_hop,
            status.vad,
            spectral_callback
        );
    } else {
        // Drift violation OR single-mic fallback:
        // Do NOT feed misaligned reference samples into NLMS (freezes weights, avoids corruption)
        // Fall back to AI-only path directly on primary headphone mic
        status.nlms_active = false;
        status.fallback_to_ai_only = true;

        mono_chain_.process_hop(
            std::span<const float>(primary_hop.data(), count),
            out_hop,
            status.vad,
            spectral_callback
        );
    }
}

} // namespace dsp
} // namespace noiselessx
