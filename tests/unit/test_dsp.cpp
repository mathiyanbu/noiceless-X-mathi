#include <gtest/gtest.h>
#include <algorithm>
#include "embedded/dsp/dc_blocker.hpp"
#include "embedded/dsp/biquad_filter.hpp"
#include "embedded/dsp/stft.hpp"
#include "embedded/dsp/vad.hpp"
#include "embedded/dsp/mono_pipeline.hpp"
#include <vector>
#include <cmath>
#include <numeric>

using namespace noiselessx::dsp;

namespace {

constexpr double PI = 3.14159265358979323846;

std::vector<float> generate_sinusoid(double freq_hz, double sample_rate, size_t num_samples, float amplitude = 0.5f) {
    std::vector<float> signal(num_samples);
    for (size_t n = 0; n < num_samples; ++n) {
        signal[n] = amplitude * static_cast<float>(std::sin(2.0 * PI * freq_hz * static_cast<double>(n) / sample_rate));
    }
    return signal;
}

float compute_rms(const float* data, size_t count) {
    if (count == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(data[i]) * static_cast<double>(data[i]);
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(count)));
}

} // anonymous namespace

// 1. STFT/iSTFT Perfect Reconstruction Verification (< -80 dB)
TEST(DspTest, StftPerfectReconstructionUnderMinus80dB) {
    constexpr size_t fft_size = 512;
    constexpr size_t hop_size = 80; // 5ms @ 16kHz
    constexpr float sample_rate = 16000.0f;
    constexpr double test_freq = 440.0;
    constexpr size_t num_hops = 60; // 4800 samples = 300 ms

    StftEngine stft(fft_size, hop_size);

    std::vector<float> input_audio = generate_sinusoid(test_freq, sample_rate, num_hops * hop_size, 0.5f);
    std::vector<float> reconstructed_audio(input_audio.size(), 0.0f);

    SpectralFrame frame;
    std::vector<float> hop_in(hop_size);
    std::vector<float> hop_out(hop_size);

    for (size_t h = 0; h < num_hops; ++h) {
        std::copy_n(input_audio.data() + h * hop_size, hop_size, hop_in.data());

        // Analysis
        stft.process_analysis(hop_in, frame);

        // Identity mask (pass-through unchanged)
        stft.process_synthesis_complex(frame.real, frame.imag, hop_out);

        std::copy_n(hop_out.data(), hop_size, reconstructed_audio.data() + h * hop_size);
    }

    // Delay through STFT analysis window and synthesis overlap-add is exactly (fft_size - hop_size)
    const size_t latency_samples = fft_size - hop_size;
    const size_t warmup_samples = fft_size * 2; // Allow overlap-add to reach steady state

    ASSERT_GT(input_audio.size(), warmup_samples + latency_samples + 800);

    const size_t eval_start = warmup_samples;
    const size_t eval_count = input_audio.size() - eval_start - 200;

    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;

    for (size_t i = 0; i < eval_count; ++i) {
        float ref = input_audio[eval_start + i - latency_samples];
        float rec = reconstructed_audio[eval_start + i];
        float err = rec - ref;
        sum_sq_err += static_cast<double>(err) * static_cast<double>(err);
        sum_sq_ref += static_cast<double>(ref) * static_cast<double>(ref);
    }

    double rms_err = std::sqrt(sum_sq_err / static_cast<double>(eval_count));
    double rms_ref = std::sqrt(sum_sq_ref / static_cast<double>(eval_count));
    double snr_db = 20.0 * std::log10(rms_err / rms_ref);

    std::cout << "[Test: StftPerfectReconstruction] Measured reconstruction error: "
              << snr_db << " dB (Requirement: < -80.0 dB)\n";

    EXPECT_LT(snr_db, -80.0);
}

// 2. High-Pass Filter -3dB Cutoff Verification
TEST(DspTest, BiquadHighPassFrequencyResponse) {
    constexpr float sample_rate = 16000.0f;
    constexpr float cutoff_hz = 80.0f;
    BiquadFilter hp(sample_rate, cutoff_hz, 0.70710678f);

    constexpr size_t num_samples = 4000;

    // A. At cutoff frequency (80 Hz) -> expected gain ~ 0.707 (-3.01 dB)
    auto sig_80hz = generate_sinusoid(80.0, sample_rate, num_samples, 1.0f);
    hp.reset();
    hp.process(sig_80hz.data(), sig_80hz.size());
    // Evaluate steady state (last 2000 samples)
    float rms_in_80 = 1.0f / std::sqrt(2.0f);
    float rms_out_80 = compute_rms(sig_80hz.data() + 2000, 2000);
    float gain_80_db = 20.0f * std::log10(rms_out_80 / rms_in_80);

    std::cout << "[Test: HighPass] Gain at cutoff (80 Hz): " << gain_80_db << " dB\n";
    EXPECT_NEAR(gain_80_db, -3.01f, 0.6f); // within 0.6 dB of -3.01 dB

    // B. Well below cutoff (20 Hz) -> strong rumble attenuation (> 20 dB)
    auto sig_20hz = generate_sinusoid(20.0, sample_rate, num_samples, 1.0f);
    hp.reset();
    hp.process(sig_20hz.data(), sig_20hz.size());
    float rms_in_20 = 1.0f / std::sqrt(2.0f);
    float rms_out_20 = compute_rms(sig_20hz.data() + 2000, 2000);
    float gain_20_db = 20.0f * std::log10(rms_out_20 / rms_in_20);

    std::cout << "[Test: HighPass] Gain at rumble (20 Hz): " << gain_20_db << " dB\n";
    EXPECT_LT(gain_20_db, -20.0f);

    // C. Well above cutoff (1000 Hz) -> passband unity gain (~ 0.0 dB)
    auto sig_1khz = generate_sinusoid(1000.0, sample_rate, num_samples, 1.0f);
    hp.reset();
    hp.process(sig_1khz.data(), sig_1khz.size());
    float rms_in_1k = 1.0f / std::sqrt(2.0f);
    float rms_out_1k = compute_rms(sig_1khz.data() + 2000, 2000);
    float gain_1k_db = 20.0f * std::log10(rms_out_1k / rms_in_1k);

    std::cout << "[Test: HighPass] Gain at passband (1 kHz): " << gain_1k_db << " dB\n";
    EXPECT_NEAR(gain_1k_db, 0.0f, 0.2f);
}

// 3. VAD Silence vs Sinusoid Test
TEST(DspTest, VadSilenceVsSpeechSinusoid) {
    constexpr size_t fft_size = 512;
    constexpr size_t hop_size = 80;
    constexpr float sample_rate = 16000.0f;

    StftEngine stft(fft_size, hop_size);
    FeatureBasedVad vad(sample_rate);
    SpectralFrame frame;

    // A. Silent buffer (all zeros / ambient noise below -70 dBFS)
    std::vector<float> silence(hop_size, 0.0f);
    stft.reset();
    vad.reset();

    VadDecision silent_decision;
    for (int i = 0; i < 20; ++i) {
        stft.process_analysis(silence, frame);
        silent_decision = vad.process_frame(silence, frame.magnitude);
    }

    std::cout << "[Test: VAD] Silence probability: " << silent_decision.voice_probability
              << ", is_speech: " << (silent_decision.is_speech ? "TRUE" : "FALSE") << "\n";
    EXPECT_FALSE(silent_decision.is_speech);
    EXPECT_LT(silent_decision.voice_probability, 0.15f);

    // B. Active speech / sinusoid buffer (500 Hz tone at -6 dBFS)
    auto tone = generate_sinusoid(500.0, sample_rate, hop_size * 30, 0.5f);
    stft.reset();
    vad.reset();

    VadDecision speech_decision;
    std::vector<float> hop(hop_size);
    for (size_t h = 0; h < 30; ++h) {
        std::copy_n(tone.data() + h * hop_size, hop_size, hop.data());
        stft.process_analysis(hop, frame);
        speech_decision = vad.process_frame(hop, frame.magnitude);
    }

    std::cout << "[Test: VAD] Tone speech probability: " << speech_decision.voice_probability
              << ", is_speech: " << (speech_decision.is_speech ? "TRUE" : "FALSE") << "\n";
    EXPECT_TRUE(speech_decision.is_speech);
    EXPECT_GT(speech_decision.voice_probability, 0.70f);
}

// 4. DC Blocker Offset Removal Verification
TEST(DspTest, DcBlockerRemovesDCOffset) {
    constexpr float sample_rate = 16000.0f;
    DcBlocker blocker(sample_rate, 10.0f);

    constexpr size_t num_samples = 4000;
    std::vector<float> signal(num_samples);
    // Injected +0.5 DC bias with AC signal
    for (size_t n = 0; n < num_samples; ++n) {
        signal[n] = 0.5f + 0.2f * static_cast<float>(std::sin(2.0 * PI * 200.0 * n / sample_rate));
    }

    blocker.process(signal.data(), signal.size());

    // In steady state (last 1000 samples), mean should be very close to 0.0
    double steady_sum = 0.0;
    for (size_t n = 3000; n < num_samples; ++n) {
        steady_sum += signal[n];
    }
    double steady_mean = steady_sum / 1000.0;

    std::cout << "[Test: DcBlocker] Residual steady-state DC mean: " << steady_mean << "\n";
    EXPECT_NEAR(steady_mean, 0.0, 0.01);
}
