#include "vad.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace noiselessx::dsp {

FeatureBasedVad::FeatureBasedVad(
    float sample_rate,
    float energy_threshold_db,
    float speech_prob_threshold
) : sample_rate_(sample_rate),
    energy_threshold_db_(energy_threshold_db),
    speech_prob_threshold_(speech_prob_threshold) {
    reset();
}

void FeatureBasedVad::reset() {
    noise_floor_db_ = -65.0f;
    smoothed_prob_ = 0.0f;
    hangover_count_ = 0;
}

float FeatureBasedVad::compute_energy_db(std::span<const float> samples) const noexcept {
    if (samples.empty()) return -120.0f;
    double sum = 0.0;
    for (float s : samples) {
        sum += static_cast<double>(s) * static_cast<double>(s);
    }
    double mean_energy = sum / static_cast<double>(samples.size());
    if (mean_energy < 1e-12) return -120.0f;
    return static_cast<float>(10.0 * std::log10(mean_energy));
}

float FeatureBasedVad::compute_zcr(std::span<const float> samples) const noexcept {
    if (samples.size() < 2) return 0.0f;
    size_t crossings = 0;
    for (size_t i = 1; i < samples.size(); ++i) {
        if ((samples[i] >= 0.0f && samples[i - 1] < 0.0f) ||
            (samples[i] < 0.0f && samples[i - 1] >= 0.0f)) {
            crossings++;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(samples.size() - 1);
}

float FeatureBasedVad::compute_spectral_flatness(std::span<const float> mag) const noexcept {
    if (mag.empty()) return 1.0f;

    double sum_power = 0.0;
    double sum_log_power = 0.0;
    const size_t count = mag.size();

    for (float m : mag) {
        double p = static_cast<double>(m) * static_cast<double>(m) + 1e-10;
        sum_power += p;
        sum_log_power += std::log(p);
    }

    double arithmetic_mean = sum_power / static_cast<double>(count);
    double geometric_mean = std::exp(sum_log_power / static_cast<double>(count));

    if (arithmetic_mean < 1e-12) return 1.0f;
    double sfm = geometric_mean / arithmetic_mean;
    return static_cast<float>(std::clamp(sfm, 0.0, 1.0));
}

VadDecision FeatureBasedVad::process_frame(
    std::span<const float> time_frame,
    std::span<const float> magnitude_spectrum
) {
    VadDecision dec;
    dec.energy_db = compute_energy_db(time_frame);
    dec.zcr = compute_zcr(time_frame);
    dec.spectral_flatness = compute_spectral_flatness(magnitude_spectrum);

    // Dynamic noise floor tracking (slow adaptation during low energy)
    if (dec.energy_db < noise_floor_db_ + 6.0f) {
        noise_floor_db_ = 0.98f * noise_floor_db_ + 0.02f * dec.energy_db;
    } else {
        noise_floor_db_ = 0.999f * noise_floor_db_ + 0.001f * dec.energy_db;
    }

    // 1. Energy metric relative to noise floor
    float snr_db = dec.energy_db - noise_floor_db_;
    float p_energy = 0.0f;
    if (snr_db > 3.0f && dec.energy_db > energy_threshold_db_) {
        p_energy = std::clamp((snr_db - 3.0f) / 12.0f, 0.0f, 1.0f);
    }

    // 2. Spectral flatness metric: harmonic/voiced speech has low flatness
    float p_flatness = std::clamp(1.0f - dec.spectral_flatness, 0.0f, 1.0f);

    // 3. Zero-crossing rate: typical speech range is 0.02 to 0.35
    float p_zcr = 0.0f;
    if (dec.zcr >= 0.02f && dec.zcr <= 0.35f) {
        p_zcr = 1.0f;
    } else if (dec.zcr < 0.02f) {
        p_zcr = dec.zcr / 0.02f;
    } else {
        p_zcr = std::clamp(1.0f - (dec.zcr - 0.35f) / 0.25f, 0.0f, 1.0f);
    }

    // Combine features
    float raw_prob = 0.50f * p_energy + 0.35f * p_flatness + 0.15f * p_zcr;

    // Hard gate on absolute silence (< -60 dBFS)
    if (dec.energy_db < -60.0f) {
        raw_prob = 0.0f;
    }

    // Hangover logic to prevent speech clipping at word boundaries
    if (raw_prob >= speech_prob_threshold_) {
        hangover_count_ = max_hangover_frames_;
    } else if (hangover_count_ > 0) {
        hangover_count_--;
        raw_prob = std::max(raw_prob, speech_prob_threshold_);
    }

    // Exponential probability smoothing
    smoothed_prob_ = 0.7f * smoothed_prob_ + 0.3f * raw_prob;
    dec.voice_probability = smoothed_prob_;
    dec.is_speech = (smoothed_prob_ >= speech_prob_threshold_);

    return dec;
}

} // namespace noiselessx::dsp
