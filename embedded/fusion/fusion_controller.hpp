#pragma once

#include "fusion_types.hpp"
#include <string>
#include <chrono>
#include <functional>
#include <vector>

namespace noiselessx::fusion {

/**
 * @brief Dual-Microphone Real-Time Fusion Controller with Explicit State Machine.
 * 
 * Fuses:
 * - AI Speech Enhancement output (headphone-mic path)
 * - NLMS Adaptive Noise Cancellation residual (dual-mic headphone + error mic path)
 * 
 * Core Features:
 * 1. Explicit Enum-Based State Machine (NORMAL, LOW_CONFIDENCE, IMPULSE, DEGRADED,
 *    NLMS_FAULT, BYPASS, ERROR).
 * 2. Dynamically computed blending parameter lambda[n] based on AI confidence and VAD.
 * 3. Asymmetric fast-attack / slow-recovery impulse protection envelope with a non-zero floor.
 * 4. Sample-continuous envelope smoothing preventing frame boundary transients.
 * 5. Deterministic fallback paths:
 *    - Reference mic dropout / drift fault -> AI-only (NLMS_FAULT), fault logged.
 *    - AI inference failure -> NLMS-only (DEGRADED), never silence.
 *    - Hardware failure -> ERROR state cleanly reported.
 *    - Operator bypass -> Minimal trivial pass-through delay.
 */
class FusionController {
public:
    using FaultLoggerCallback = std::function<void(FusionMode mode, const std::string& reason, uint64_t frame_index)>;

    explicit FusionController(const FusionConfig& config = FusionConfig{});
    ~FusionController() = default;

    // Movable, non-copyable
    FusionController(const FusionController&) = delete;
    FusionController& operator=(const FusionController&) = delete;
    FusionController(FusionController&&) noexcept = default;
    FusionController& operator=(FusionController&&) noexcept = default;

    /**
     * @brief Process and fuse incoming audio streams for one frame.
     * Evaluates state machine transitions, computes dynamic lambda, updates the impulse
     * gain envelope, and produces the final enhanced AudioFrame.
     * 
     * @param input FusionInput containing audio streams, confidence, impulse, and health flags.
     * @return Fused and protected AudioFrame.
     */
    AudioFrame fuse(const FusionInput& input);

    /**
     * @brief Evaluates the next runtime mode from input signals and current state.
     * Explicit deterministic transition logic.
     */
    [[nodiscard]] FusionMode evaluate_mode(const FusionInput& input) const noexcept;

    /**
     * @brief Reset state machine, impulse envelope, and diagnostic counters.
     */
    void reset();

    // Mode & State Inspection
    [[nodiscard]] FusionMode get_current_mode() const noexcept { return current_mode_; }
    [[nodiscard]] FusionMode get_previous_mode() const noexcept { return previous_mode_; }
    [[nodiscard]] float get_current_lambda() const noexcept { return current_lambda_; }
    [[nodiscard]] float get_current_gain_envelope() const noexcept { return current_envelope_gain_; }
    [[nodiscard]] uint64_t get_fault_count() const noexcept { return fault_count_; }
    [[nodiscard]] uint64_t get_frame_count() const noexcept { return frame_count_; }
    [[nodiscard]] const std::string& get_last_fault_message() const noexcept { return last_fault_message_; }
    [[nodiscard]] double get_last_processing_time_us() const noexcept { return last_processing_time_us_; }

    // Manual overrides
    void set_bypass(bool bypass) noexcept { manual_bypass_ = bypass; }
    [[nodiscard]] bool is_bypass() const noexcept { return manual_bypass_; }

    void set_manual_error(bool error) noexcept { manual_error_ = error; }
    [[nodiscard]] bool has_manual_error() const noexcept { return manual_error_; }

    // Configuration
    [[nodiscard]] const FusionConfig& get_config() const noexcept { return config_; }
    void set_config(const FusionConfig& config) noexcept { config_ = config; }

    // Custom fault logger hook
    void set_fault_logger(FaultLoggerCallback callback) { fault_logger_ = std::move(callback); }

private:
    // Explicit state action execution handlers
    AudioFrame process_normal(const FusionInput& input, float target_gain);
    AudioFrame process_low_confidence(const FusionInput& input, float target_gain);
    AudioFrame process_impulse(const FusionInput& input, float target_gain);
    AudioFrame process_degraded(const FusionInput& input, float target_gain);
    AudioFrame process_nlms_fault(const FusionInput& input, float target_gain);
    AudioFrame process_bypass(const FusionInput& input);
    AudioFrame process_error(const FusionInput& input);

    // Helpers
    void log_fault_event(FusionMode mode, const std::string& reason);
    [[nodiscard]] float compute_dynamic_lambda(float ai_conf, float vad_prob, bool low_confidence_mode) const noexcept;
    void update_gain_envelope(float target_gain);
    void apply_gain_envelope(AudioFrame& frame, float prev_gain, float curr_gain);

private:
    FusionConfig config_;
    FusionMode current_mode_{FusionMode::NORMAL};
    FusionMode previous_mode_{FusionMode::NORMAL};

    float current_lambda_{0.5f};
    float current_envelope_gain_{1.0f};
    float previous_envelope_gain_{1.0f};

    bool manual_bypass_{false};
    bool manual_error_{false};

    uint64_t frame_count_{0};
    uint64_t fault_count_{0};
    std::string last_fault_message_;
    double last_processing_time_us_{0.0};

    FaultLoggerCallback fault_logger_{nullptr};
};

} // namespace noiselessx::fusion

using FusionController = noiselessx::fusion::FusionController;
