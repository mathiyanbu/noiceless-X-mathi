#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <array>

namespace noiselessx::impulse {

/**
 * @brief Extracted feature vector for impulsive noise detection (8 features).
 */
struct ImpulseFeatures {
    float rms{0.0f};            // Time-domain Root Mean Square
    float spectral_flux{0.0f};  // Half-wave rectified positive spectral difference
    float crest_factor{0.0f};   // Peak-to-RMS ratio
    float zcr{0.0f};            // Zero Crossing Rate
    std::array<float, 4> band_energies{0.0f, 0.0f, 0.0f, 0.0f}; // Low, Mid-Low, Mid-High, High
};

/**
 * @brief Hand-tuned logistic regression weights for feature combination.
 */
struct ImpulseWeights {
    float w_rms{1.2f};
    float w_flux{0.001f};
    float w_crest{3.5f};
    float w_zcr{-2.0f};
    std::array<float, 4> w_bands{-0.5f, -0.3f, 1.0f, 3.0f};
    float bias{-14.0f};
};

/**
 * @brief Configuration parameters for impulsive noise detector.
 */
struct ImpulseConfig {
    bool enabled{true};
    bool use_onnx{false};
    std::string onnx_model_path{"models/onnx/impulse_detector.onnx"};
    
    // Dual thresholds for anti-chattering hysteresis
    float threshold_on{0.65f};  // Probability threshold to trigger detection
    float threshold_off{0.35f}; // Probability threshold to release detection

    // Asymmetric temporal smoothing
    float attack_alpha{0.80f};  // Fast attack response to rising transients
    float release_alpha{0.15f}; // Slow release to avoid audio flapping

    // Hand-tuned weights (used when use_onnx is false)
    ImpulseWeights weights;
};

/**
 * @brief Output state of the impulsive-noise detector.
 */
struct ImpulseState {
    float probability{0.0f};  // Smoothed probability in [0.0, 1.0]
    bool detected{false};      // Hysteresis-latched detection flag
    float attack_gain{1.0f};   // Instantaneous transient suppression gain in [0.0, 1.0]
    float release_gain{1.0f};  // Smoothed recovery gain envelope in [0.0, 1.0]
};

} // namespace noiselessx::impulse

// Global alias for seamless accessibility
using ImpulseState = noiselessx::impulse::ImpulseState;
