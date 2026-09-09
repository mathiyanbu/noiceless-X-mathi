#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace noiselessx::dsp {

struct VadDecision {
    bool is_speech{false};
    float voice_probability{0.0f}; // [0.0, 1.0]
    float energy_db{-120.0f};
    float zcr{0.0f};
    float spectral_flatness{1.0f};
};

/**
 * @brief Abstract Voice Activity Detector (VAD) Interface.
 * 
 * Defines the contract for speech presence detection. Designed to allow seamless
 * replacement of feature-based heuristics with learned neural VAD models in later phases.
 */
class IVad {
public:
    virtual ~IVad() = default;

    /**
     * @brief Process an audio frame and its magnitude spectrum.
     * @param time_frame Time-domain samples of current analysis window
     * @param magnitude_spectrum Frequency magnitude bins from STFT
     * @return VadDecision containing speech probability and presence flag
     */
    virtual VadDecision process_frame(
        std::span<const float> time_frame,
        std::span<const float> magnitude_spectrum
    ) = 0;

    virtual void reset() = 0;
};

/**
 * @brief Real threshold-based VAD combining Short-Term Energy, Zero-Crossing Rate,
 *        and Spectral Flatness Measure.
 */
class FeatureBasedVad : public IVad {
public:
    explicit FeatureBasedVad(
        float sample_rate = 16000.0f,
        float energy_threshold_db = -45.0f,
        float speech_prob_threshold = 0.5f
    );

    VadDecision process_frame(
        std::span<const float> time_frame,
        std::span<const float> magnitude_spectrum
    ) override;

    void reset() override;

    void set_energy_threshold_db(float db) noexcept { energy_threshold_db_ = db; }
    void set_probability_threshold(float prob) noexcept { speech_prob_threshold_ = prob; }

private:
    float compute_energy_db(std::span<const float> samples) const noexcept;
    float compute_zcr(std::span<const float> samples) const noexcept;
    float compute_spectral_flatness(std::span<const float> mag) const noexcept;

    float sample_rate_{16000.0f};
    float energy_threshold_db_{-45.0f};
    float speech_prob_threshold_{0.5f};

    float noise_floor_db_{-65.0f};
    float smoothed_prob_{0.0f};
    int hangover_count_{0};
    int max_hangover_frames_{5};
};

} // namespace noiselessx::dsp
