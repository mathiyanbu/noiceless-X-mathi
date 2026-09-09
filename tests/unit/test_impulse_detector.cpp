#include <gtest/gtest.h>
#include "embedded/impulse/impulse_detector.hpp"

#include <vector>
#include <cmath>
#include <numbers>

namespace {

// Helper to generate a pure tone (steady state)
std::vector<float> generate_tone(size_t n_samples, float freq_hz, float sample_rate = 16000.0f) {
    std::vector<float> tone(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        tone[i] = 0.5f * std::sin(2.0f * 3.14159265f * freq_hz * t);
    }
    return tone;
}

// Helper to generate an impulsive signal (sharp click / delta spike)
std::vector<float> generate_impulse_click(size_t n_samples, size_t click_pos = 100) {
    std::vector<float> click(n_samples, 0.0f);
    if (click_pos < n_samples) {
        click[click_pos] = 0.95f;
    }
    if (click_pos + 1 < n_samples) {
        click[click_pos + 1] = -0.45f;
    }
    return click;
}

} // namespace

TEST(ImpulseDetectorTest, FeatureExtractionCalculations) {
    noiselessx::impulse::ImpulseDetector detector;
    constexpr size_t N = 512;
    constexpr size_t NUM_BINS = 257;

    // 1. DC / Constant signal: x[n] = 0.5
    std::vector<float> dc_frame(N, 0.5f);
    std::vector<float> zero_mag(NUM_BINS, 0.0f);

    auto feats = detector.extract_features(dc_frame, zero_mag);
    EXPECT_NEAR(feats.rms, 0.5f, 1e-4f);
    EXPECT_NEAR(feats.crest_factor, 1.0f, 1e-3f); // For DC, max|x| == RMS
    EXPECT_FLOAT_EQ(feats.zcr, 0.0f);             // No zero crossings in positive DC

    // 2. Square wave alternating +0.5 and -0.5: ZCR should be 1.0
    std::vector<float> square_frame(N);
    for (size_t i = 0; i < N; ++i) {
        square_frame[i] = (i % 2 == 0) ? 0.5f : -0.5f;
    }
    auto sq_feats = detector.extract_features(square_frame, zero_mag);
    EXPECT_NEAR(sq_feats.rms, 0.5f, 1e-4f);
    EXPECT_NEAR(sq_feats.zcr, 1.0f, 1e-3f);

    // 3. Spectral Flux: identical successive spectra must produce 0 flux
    std::vector<float> steady_mag(NUM_BINS, 1.5f);
    detector.extract_features(dc_frame, steady_mag); // first frame sets prev_magnitude
    auto flux_feats = detector.extract_features(dc_frame, steady_mag); // second frame identical
    EXPECT_FLOAT_EQ(flux_feats.spectral_flux, 0.0f);
}

TEST(ImpulseDetectorTest, ImpulseVsSteadyStateSeparation) {
    noiselessx::impulse::ImpulseDetector detector;
    constexpr size_t N = 512;
    constexpr size_t NUM_BINS = 257;

    // 1. Process several steady-state vowel/sinusoid frames
    auto steady_signal = generate_tone(N, 300.0f);
    std::vector<float> steady_mag(NUM_BINS, 0.1f);
    steady_mag[10] = 5.0f; // 300 Hz peak in band 0 (low freq)

    noiselessx::impulse::ImpulseState steady_state;
    for (int i = 0; i < 10; ++i) {
        steady_state = detector.detect(steady_signal, steady_mag);
    }
    EXPECT_LT(steady_state.probability, 0.35f);
    EXPECT_FALSE(steady_state.detected);
    EXPECT_GT(steady_state.attack_gain, 0.65f);

    // 2. Inject an abrupt high-amplitude impulse
    auto impulse_signal = generate_impulse_click(N, 200);
    std::vector<float> impulse_mag(NUM_BINS, 0.05f);
    // Broad high-frequency spectral burst
    for (size_t k = 100; k < 250; ++k) {
        impulse_mag[k] = 3.5f;
    }

    auto impulse_state = detector.detect(impulse_signal, impulse_mag);
    EXPECT_GT(impulse_state.probability, 0.65f);
    EXPECT_TRUE(impulse_state.detected);
    EXPECT_LT(impulse_state.attack_gain, 0.40f);
}

TEST(ImpulseDetectorTest, HysteresisPreventsBorderlineFlapping) {
    noiselessx::impulse::ImpulseConfig cfg;
    cfg.threshold_on = 0.65f;
    cfg.threshold_off = 0.35f;
    cfg.attack_alpha = 0.80f;
    cfg.release_alpha = 0.15f;
    noiselessx::impulse::ImpulseDetector detector(cfg);

    constexpr size_t N = 512;
    constexpr size_t NUM_BINS = 257;

    // First trigger an impulse to enter detected state
    auto impulse = generate_impulse_click(N, 150);
    std::vector<float> burst_mag(NUM_BINS, 2.0f);
    auto state = detector.detect(impulse, burst_mag);
    EXPECT_TRUE(state.detected);

    // Now present borderline signals where probability drops to ~0.50 (between 0.35 and 0.65)
    // In hysteresis, since it was already detected, it must stay detected until dropping below 0.35!
    std::vector<float> borderline_frame(N, 0.05f);
    borderline_frame[50] = 0.25f; // small transient
    std::vector<float> border_mag(NUM_BINS, 0.3f);

    auto border_state = detector.detect(borderline_frame, border_mag);
    // Even though probability is below threshold_on (0.65), it should not flap to false if above 0.35
    if (border_state.probability > cfg.threshold_off) {
        EXPECT_TRUE(border_state.detected);
    }
}

TEST(ImpulseDetectorTest, ResetClearsState) {
    noiselessx::impulse::ImpulseDetector detector;
    constexpr size_t N = 512;
    constexpr size_t NUM_BINS = 257;

    auto impulse = generate_impulse_click(N, 100);
    std::vector<float> burst_mag(NUM_BINS, 3.0f);
    detector.detect(impulse, burst_mag);

    EXPECT_GT(detector.get_smoothed_probability(), 0.0f);

    detector.reset();
    EXPECT_FLOAT_EQ(detector.get_smoothed_probability(), 0.0f);
    EXPECT_FALSE(detector.is_detected());
}
