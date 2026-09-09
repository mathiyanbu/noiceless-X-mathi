#pragma once

#include "dc_blocker.hpp"
#include "biquad_filter.hpp"
#include "stft.hpp"
#include "vad.hpp"
#include <functional>
#include <span>

namespace noiselessx {
namespace dsp {

/**
 * @brief Complete Single-Channel (Mono) Preprocessing and STFT/iSTFT Pipeline.
 * 
 * Chains:
 * Raw Samples -> DC Blocker -> Biquad High-Pass (80Hz) -> STFT Analysis -> VAD
 *             -> [Spectral Modification Callback / Mask] -> iSTFT Synthesis -> Clean Audio
 * 
 * Guarantees zero heap allocation per frame.
 */
class MonoPipeline {
public:
    using SpectralProcessorCallback = std::function<void(
        SpectralFrame& frame,
        const VadDecision& vad
    )>;

    explicit MonoPipeline(
        float sample_rate = 16000.0f,
        float hp_cutoff_hz = 80.0f,
        size_t fft_size = 512,
        size_t hop_size = 80
    );

    /**
     * @brief Process an incoming hop of samples with optional spectral processing callback.
     * @param hop_in Input time-domain hop (length = hop_size)
     * @param hop_out Output synthesized time-domain hop (length = hop_size)
     * @param vad_decision Output VAD decision for this frame
     * @param callback Optional callback to modify spectral magnitude/phase (e.g. AI mask or identity)
     */
    void process_hop(
        std::span<const float> hop_in,
        std::span<float> hop_out,
        VadDecision& vad_decision,
        const SpectralProcessorCallback& callback = nullptr
    );

    void reset();

    [[nodiscard]] StftEngine& get_stft_engine() noexcept { return stft_; }
    [[nodiscard]] IVad& get_vad() noexcept { return vad_; }
    [[nodiscard]] BiquadFilter& get_high_pass() noexcept { return high_pass_; }
    [[nodiscard]] DcBlocker& get_dc_blocker() noexcept { return dc_blocker_; }

private:
    float sample_rate_{16000.0f};
    size_t fft_size_{512};
    size_t hop_size_{80};

    DcBlocker dc_blocker_;
    BiquadFilter high_pass_;
    StftEngine stft_;
    FeatureBasedVad vad_;

    // Preallocated scratch buffers
    std::vector<float> preprocessed_hop_;
    SpectralFrame spectral_frame_;
};

} // namespace dsp
} // namespace noiselessx
