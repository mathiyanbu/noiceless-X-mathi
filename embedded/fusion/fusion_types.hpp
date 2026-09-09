#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <string>
#include <string_view>
#include <algorithm>
#include <initializer_list>

namespace noiselessx::fusion {

/**
 * @brief Fixed/variable size AudioFrame container representing a single processing hop/frame.
 * Standard hop in NOICELESSX is 80 samples (5ms @ 16kHz) or 160 samples (10ms @ 16kHz).
 */
struct AudioFrame {
    std::vector<float> samples;
    uint64_t frame_index{0};
    uint64_t timestamp_ns{0};

    AudioFrame() = default;
    explicit AudioFrame(size_t count, float val = 0.0f) : samples(count, val) {}
    AudioFrame(std::initializer_list<float> init) : samples(init) {}
    explicit AudioFrame(std::vector<float> s) : samples(std::move(s)) {}
    AudioFrame(const float* data, size_t count) : samples(data, data + count) {}
    explicit AudioFrame(std::span<const float> s) : samples(s.begin(), s.end()) {}

    [[nodiscard]] size_t size() const noexcept { return samples.size(); }
    [[nodiscard]] bool empty() const noexcept { return samples.empty(); }
    void resize(size_t new_size, float val = 0.0f) { samples.resize(new_size, val); }
    void clear() noexcept { samples.clear(); }

    float& operator[](size_t idx) noexcept { return samples[idx]; }
    const float& operator[](size_t idx) const noexcept { return samples[idx]; }

    float* data() noexcept { return samples.data(); }
    const float* data() const noexcept { return samples.data(); }

    auto begin() noexcept { return samples.begin(); }
    auto end() noexcept { return samples.end(); }
    auto begin() const noexcept { return samples.begin(); }
    auto end() const noexcept { return samples.end(); }
};

/**
 * @brief Explicit runtime modes for the Dual-Mic Fusion Controller state machine.
 */
enum class FusionMode {
    NORMAL = 0,      ///< AI + NLMS both active, impulse monitoring, dynamic lambda fusion
    LOW_CONFIDENCE,  ///< ai_confidence below threshold -> lean more heavily on NLMS output
    IMPULSE,         ///< impulse_probability above threshold -> apply full gain envelope
    DEGRADED,        ///< AI inference failing (Phase 7 error) -> output NLMS-only (never silence)
    NLMS_FAULT,      ///< Reference mic dropout / clock drift exceeded -> fall back to AI-only, log fault
    BYPASS,          ///< Operator-triggered or unrecoverable failure -> input routed directly to output
    ERROR            ///< Both mics/hardware unavailable -> no processing, clear error reported
};

[[nodiscard]] constexpr std::string_view to_string(FusionMode mode) noexcept {
    switch (mode) {
        case FusionMode::NORMAL: return "NORMAL";
        case FusionMode::LOW_CONFIDENCE: return "LOW_CONFIDENCE";
        case FusionMode::IMPULSE: return "IMPULSE";
        case FusionMode::DEGRADED: return "DEGRADED";
        case FusionMode::NLMS_FAULT: return "NLMS_FAULT";
        case FusionMode::BYPASS: return "BYPASS";
        case FusionMode::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Configuration parameters for the Fusion Controller.
 */
struct FusionConfig {
    float sample_rate{16000.0f};
    size_t default_frame_size{80};       // 5ms hop @ 16kHz
    
    // Dynamic lambda bounds
    float lambda_min{0.10f};             // Baseline minimum AI contribution during silence/noise
    float lambda_max{0.95f};             // Maximum AI contribution during active speech
    
    // Low confidence threshold & scaling
    float low_confidence_threshold{0.45f};
    float low_confidence_scale{0.25f};   // Factor reducing lambda in LOW_CONFIDENCE mode

    // Impulsive noise protection parameters
    float impulse_threshold{0.60f};      // Triggers IMPULSE mode
    float impulse_beta{0.90f};           // Gain attenuation strength: g_I = 1 - beta * P_impulse
    float impulse_gain_floor{0.10f};     // Floor > 0: audio NEVER fully cuts out / hard mutes
    
    // Gain envelope time smoothing
    float attack_alpha{0.85f};           // Fast attack response to transient onset
    float release_alpha{0.12f};          // Slower recovery to prevent flapping
    
    bool log_faults{true};               // Automatically log component faults
};

/**
 * @brief Input payload delivered to the fusion controller every frame.
 */
struct FusionInput {
    AudioFrame ai_output;                // AI enhanced output from headphone-mic path
    AudioFrame nlms_output;              // NLMS adaptive cancellation residual from dual-mic path
    float ai_confidence{1.0f};           // AI model confidence score in [0.0, 1.0]
    float impulse_probability{0.0f};     // Impulsive transient detection probability in [0.0, 1.0]
    float vad_probability{0.0f};         // Voice activity probability in [0.0, 1.0]
    bool nlms_available{true};           // false if Phase 2 drift check or Phase 4 NLMS reports fault
    
    // Extended diagnostic / system inputs
    AudioFrame raw_input{};              // Raw unenhanced input (used in BYPASS mode)
    bool ai_available{true};             // false if Phase 7 AI inference fails (DEGRADED mode)
    bool hardware_available{true};       // false if both mics/audio hardware drop (ERROR mode)
    bool bypass_requested{false};        // Operator-triggered bypass signal
};

} // namespace noiselessx::fusion

// Global aliases for seamless project-wide accessibility
using AudioFrame = noiselessx::fusion::AudioFrame;
using FusionMode = noiselessx::fusion::FusionMode;
using FusionConfig = noiselessx::fusion::FusionConfig;
using FusionInput = noiselessx::fusion::FusionInput;
