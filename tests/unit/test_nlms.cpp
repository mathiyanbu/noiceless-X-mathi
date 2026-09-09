#include <gtest/gtest.h>
#include "embedded/dsp/nlms.hpp"
#include "embedded/dsp/dual_mic_pipeline.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <random>

using namespace noiselessx::dsp;

namespace {

constexpr double PI = 3.14159265358979323846;

// FIR convolution to simulate acoustic transfer function from noise source to primary mic
std::vector<float> apply_fir(const std::vector<float>& in, const std::vector<float>& ir) {
    std::vector<float> out(in.size(), 0.0f);
    for (size_t n = 0; n < in.size(); ++n) {
        float sum = 0.0f;
        for (size_t k = 0; k < ir.size(); ++k) {
            if (n >= k) {
                sum += ir[k] * in[n - k];
            }
        }
        out[n] = sum;
    }
    return out;
}

float compute_power(const float* data, size_t count) {
    if (count == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(data[i]) * static_cast<double>(data[i]);
    }
    return static_cast<float>(sum / static_cast<double>(count));
}

} // anonymous namespace

// 1. Test NLMS Convergence on Correlated Acoustic Noise
TEST(NLMSTest, ConvergenceOnCorrelatedReference) {
    constexpr float sample_rate = 16000.0f;
    constexpr size_t filter_length = 64;
    constexpr float learning_rate = 0.15f; // Fast convergence for test
    constexpr float epsilon = 1e-6f;

    NLMSFilter filter(static_cast<int>(filter_length), learning_rate, epsilon);

    constexpr size_t total_samples = 4000;
    constexpr size_t block_size = 800; // 5 blocks of 800 samples

    // A. Desired speech signal (400 Hz tone)
    std::vector<float> speech(total_samples);
    for (size_t n = 0; n < total_samples; ++n) {
        speech[n] = 0.4f * static_cast<float>(std::sin(2.0 * PI * 400.0 * n / sample_rate));
    }

    // B. Acoustic noise source (multi-tone complex noise at reference mic)
    std::vector<float> reference_noise(total_samples);
    for (size_t n = 0; n < total_samples; ++n) {
        reference_noise[n] = 0.35f * static_cast<float>(std::sin(2.0 * PI * 850.0 * n / sample_rate)) +
                             0.25f * static_cast<float>(std::cos(2.0 * PI * 1350.0 * n / sample_rate)) +
                             0.15f * static_cast<float>(std::sin(2.0 * PI * 2200.0 * n / sample_rate));
    }

    // C. Acoustic channel impulse response: h = [0.7, -0.4, 0.25, -0.15, 0.08]
    std::vector<float> acoustic_channel = {0.7f, -0.4f, 0.25f, -0.15f, 0.08f};
    std::vector<float> primary_noise = apply_fir(reference_noise, acoustic_channel);

    // Primary mic receives speech + correlated acoustic noise
    std::vector<float> primary(total_samples);
    for (size_t n = 0; n < total_samples; ++n) {
        primary[n] = speech[n] + primary_noise[n];
    }

    // Process through NLMS
    std::vector<float> output_error(total_samples);
    std::vector<float> residual_noise(total_samples);

    for (size_t n = 0; n < total_samples; ++n) {
        output_error[n] = filter.process(primary[n], reference_noise[n]);
        residual_noise[n] = output_error[n] - speech[n]; // Error relative to clean speech
    }

    // Measure residual noise power per block to verify monotonic convergence
    std::vector<float> block_noise_powers;
    const size_t num_blocks = total_samples / block_size;

    std::cout << "\n[Test: NLMS Convergence] Tracking residual noise across blocks:\n";
    for (size_t b = 0; b < num_blocks; ++b) {
        float p = compute_power(residual_noise.data() + b * block_size, block_size);
        float p_db = 10.0f * std::log10(p + 1e-12f);
        block_noise_powers.push_back(p);
        std::cout << "  Block " << (b + 1) << " Residual Noise Power: " << p
                  << " (" << p_db << " dB)\n";
    }

    // 1. Monotonic decrease after initial adaptation transient (block 1 -> 2 -> 3 -> 4)
    EXPECT_GT(block_noise_powers[0], block_noise_powers[1]);
    EXPECT_GT(block_noise_powers[1], block_noise_powers[2]);
    EXPECT_GT(block_noise_powers[2], block_noise_powers[3]);

    // 2. Steady-state noise attenuation: initial noise vs final block residual
    float initial_noise_power = compute_power(primary_noise.data(), block_size);
    float final_residual_power = block_noise_powers.back();
    float attenuation_db = 10.0f * std::log10(initial_noise_power / (final_residual_power + 1e-12f));

    std::cout << "  Initial Noise Power:   " << initial_noise_power << "\n"
              << "  Final Residual Power: " << final_residual_power << "\n"
              << "  Total Noise Reduction: " << attenuation_db << " dB\n";

    // Expected steady-state noise reduction > 20 dB (or factor of 100 in power)
    EXPECT_GT(attenuation_db, 20.0f);
}

// 2. Test Decorrelated Reference (Speech Preservation — No False Convergence)
TEST(NLMSTest, DecorrelatedReferencePreservesSpeech) {
    constexpr float sample_rate = 16000.0f;
    constexpr size_t filter_length = 64;
    constexpr float learning_rate = 0.05f;
    constexpr float epsilon = 1e-6f;

    NLMSFilter filter(static_cast<int>(filter_length), learning_rate, epsilon);

    constexpr size_t total_samples = 3200;

    // Primary receives pure speech (500 Hz tone)
    std::vector<float> speech(total_samples);
    for (size_t n = 0; n < total_samples; ++n) {
        speech[n] = 0.5f * static_cast<float>(std::sin(2.0 * PI * 500.0 * n / sample_rate));
    }

    // Reference receives uncorrelated pseudo-random white noise
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.2f);
    std::vector<float> uncorrelated_noise(total_samples);
    for (size_t n = 0; n < total_samples; ++n) {
        uncorrelated_noise[n] = dist(rng);
    }

    std::vector<float> output_error(total_samples);
    for (size_t n = 0; n < total_samples; ++n) {
        output_error[n] = filter.process(speech[n], uncorrelated_noise[n]);
    }

    // In steady state (last 1600 samples), verify speech power is NOT attenuated
    float speech_power = compute_power(speech.data() + 1600, 1600);
    float output_power = compute_power(output_error.data() + 1600, 1600);
    float power_ratio = output_power / speech_power;

    std::cout << "\n[Test: Decorrelated Reference] Speech Preservation:\n"
              << "  Speech Power:   " << speech_power << "\n"
              << "  Output Power:   " << output_power << "\n"
              << "  Preserved Ratio: " << (power_ratio * 100.0f) << "%\n";

    // Speech must not be canceled: power ratio must be close to 1.0 (>= 0.90)
    EXPECT_GT(power_ratio, 0.90f);

    // Filter weights must remain small
    float max_weight = 0.0f;
    for (float w : filter.get_weights()) {
        max_weight = std::max(max_weight, std::abs(w));
    }
    std::cout << "  Max Filter Weight: " << max_weight << "\n";
    EXPECT_LT(max_weight, 0.15f);
}

// 3. Test Drift-Checked Synchronization Guard
TEST(NLMSTest, DriftCheckedSynchronizationGuard) {
    DualMicPipeline pipeline(
        16000.0f, // sample_rate
        128,      // filter_length
        0.05f,    // learning_rate
        1e-6f,    // epsilon
        10.0      // max_allowable_drift_ms
    );

    std::vector<float> primary_hop(80, 0.3f);
    std::vector<float> reference_hop(80, 0.1f);
    std::vector<float> out_hop(80, 0.0f);
    DualMicStatus status;

    // Case A: Drift within tolerance (2.0 ms <= 10.0 ms)
    pipeline.process_dual_hop(primary_hop, reference_hop, 2.0, out_hop, status);
    EXPECT_TRUE(status.nlms_active);
    EXPECT_FALSE(status.drift_inhibited);
    EXPECT_FALSE(status.fallback_to_ai_only);

    // Case B: Drift exceeds tolerance (14.5 ms > 10.0 ms)
    pipeline.process_dual_hop(primary_hop, reference_hop, 14.5, out_hop, status);
    EXPECT_FALSE(status.nlms_active);
    EXPECT_TRUE(status.drift_inhibited);
    EXPECT_TRUE(status.fallback_to_ai_only);

    // Case C: Drift recovers (1.5 ms <= 10.0 ms)
    pipeline.process_dual_hop(primary_hop, reference_hop, 1.5, out_hop, status);
    EXPECT_TRUE(status.nlms_active);
    EXPECT_FALSE(status.drift_inhibited);
    EXPECT_FALSE(status.fallback_to_ai_only);
}

// 4. Test Reset Functionality
TEST(NLMSTest, ResetClearsWeightsAndHistory) {
    NLMSFilter filter(64, 0.1f, 1e-6f);

    // Adapt filter
    for (int i = 0; i < 200; ++i) {
        filter.process(0.5f, 0.3f);
    }

    // Weights should be non-zero
    bool has_nonzero_weights = false;
    for (float w : filter.get_weights()) {
        if (std::abs(w) > 1e-4f) has_nonzero_weights = true;
    }
    EXPECT_TRUE(has_nonzero_weights);

    // Reset
    filter.reset();
    for (float w : filter.get_weights()) {
        EXPECT_FLOAT_EQ(w, 0.0f);
    }
    for (float h : filter.get_history()) {
        EXPECT_FLOAT_EQ(h, 0.0f);
    }
}
