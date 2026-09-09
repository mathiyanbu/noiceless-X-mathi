#pragma once

#include "impulse_types.hpp"

#include <span>
#include <vector>
#include <memory>
#include <string>

namespace noiselessx::impulse {

/**
 * @brief Real-time single-channel impulsive noise detector.
 * 
 * Computes time-domain and spectral-domain features:
 * - RMS energy
 * - Spectral flux (half-wave rectified)
 * - Crest factor (peak / RMS)
 * - Zero Crossing Rate (ZCR)
 * - Subband energies (4 bands across 0-8000 Hz)
 * 
 * Evaluates impulse probability via either:
 * 1. Hand-tuned logistic combination (configurable via YAML, zero dependencies)
 * 2. Pretrained ONNX MLP classifier (via ONNX Runtime C++ API, Phase 7 infrastructure)
 * 
 * Applies asymmetric temporal smoothing (fast attack, slow release) and dual-threshold
 * hysteresis to prevent chattering and flapping on borderline transients.
 */
class ImpulseDetector {
public:
    explicit ImpulseDetector(const ImpulseConfig& config = ImpulseConfig{});
    ~ImpulseDetector();

    ImpulseDetector(const ImpulseDetector&) = delete;
    ImpulseDetector& operator=(const ImpulseDetector&) = delete;

    ImpulseDetector(ImpulseDetector&&) noexcept;
    ImpulseDetector& operator=(ImpulseDetector&&) noexcept;

    /**
     * @brief Optional initialization of ONNX model backend for trained classifier inference.
     * @param model_path Path to the exported impulse detector ONNX model.
     * @param num_threads Intra-op thread count (default: 1 for lightweight detector).
     * @return true if initialized successfully, false otherwise.
     */
    bool initialize_onnx(const std::string& model_path, int num_threads = 1);

    /**
     * @brief Extract raw feature vector from time-domain frame and spectral magnitude.
     * @param time_frame Time-domain samples of the frame (e.g. 512 samples)
     * @param spectral_magnitude Half-spectrum magnitudes (257 bins for N=512)
     */
    ImpulseFeatures extract_features(
        std::span<const float> time_frame,
        std::span<const float> spectral_magnitude
    );

    /**
     * @brief Perform full impulsive noise detection on a frame.
     * @param time_frame Time-domain audio frame samples
     * @param spectral_magnitude STFT magnitude spectrum of length 257
     * @return ImpulseState struct containing smoothed probability, detection flag, and gains.
     */
    ImpulseState detect(
        std::span<const float> time_frame,
        std::span<const float> spectral_magnitude
    );

    /**
     * @brief Reset temporal state (spectral history, smoothed probability, hysteresis state).
     */
    void reset();

    [[nodiscard]] const ImpulseConfig& get_config() const noexcept { return config_; }
    void set_config(const ImpulseConfig& config) { config_ = config; }
    [[nodiscard]] float get_smoothed_probability() const noexcept { return smoothed_prob_; }
    [[nodiscard]] bool is_detected() const noexcept { return detected_latched_; }

private:
    float compute_raw_probability(const ImpulseFeatures& features);

    ImpulseConfig config_;

    // Persistent state across consecutive frames
    std::vector<float> prev_magnitude_; // Stores |X(k, m-1)| for spectral flux computation
    float smoothed_prob_{0.0f};
    bool detected_latched_{false};
    float current_gain_{1.0f};

    class OnnxBackend;
    std::unique_ptr<OnnxBackend> onnx_backend_;
};

} // namespace noiselessx::impulse
