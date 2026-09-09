#include "fusion_controller.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

namespace noiselessx {
namespace fusion {

FusionController::FusionController(const FusionConfig& config)
    : config_(config),
      current_mode_(FusionMode::NORMAL),
      previous_mode_(FusionMode::NORMAL),
      current_lambda_(0.5f),
      current_envelope_gain_(1.0f),
      previous_envelope_gain_(1.0f)
{
}

void FusionController::reset() {
    current_mode_ = FusionMode::NORMAL;
    previous_mode_ = FusionMode::NORMAL;
    current_lambda_ = 0.5f;
    current_envelope_gain_ = 1.0f;
    previous_envelope_gain_ = 1.0f;
    manual_bypass_ = false;
    manual_error_ = false;
    frame_count_ = 0;
    fault_count_ = 0;
    last_fault_message_.clear();
    last_processing_time_us_ = 0.0;
}

FusionMode FusionController::evaluate_mode(const FusionInput& input) const noexcept {
    // 1. Unrecoverable hardware failure / both audio paths unavailable
    if (manual_error_ || !input.hardware_available || (!input.ai_available && !input.nlms_available)) {
        return FusionMode::ERROR;
    }

    // 2. Operator-triggered bypass or explicit bypass request
    if (manual_bypass_ || input.bypass_requested) {
        return FusionMode::BYPASS;
    }

    // 3. Component faults (high priority before signal quality heuristics)
    // AI inference failure -> DEGRADED (NLMS-only output)
    if (!input.ai_available) {
        return FusionMode::DEGRADED;
    }

    // Reference mic dropout or clock drift failure -> NLMS_FAULT (AI-only output)
    if (!input.nlms_available) {
        return FusionMode::NLMS_FAULT;
    }

    // 4. Signal condition heuristics (both AI and NLMS available)
    // Impulse noise detection takes precedence to guard listener hearing
    if (input.impulse_probability >= config_.impulse_threshold) {
        return FusionMode::IMPULSE;
    }

    // Low AI confidence -> lean more heavily on NLMS
    if (input.ai_confidence < config_.low_confidence_threshold) {
        return FusionMode::LOW_CONFIDENCE;
    }

    return FusionMode::NORMAL;
}

AudioFrame FusionController::fuse(const FusionInput& input) {
    const auto start_time = std::chrono::steady_clock::now();
    ++frame_count_;

    // 1. Explicit State Machine Transition Evaluation
    const FusionMode next_mode = evaluate_mode(input);
    if (next_mode != current_mode_) {
        previous_mode_ = current_mode_;
        current_mode_ = next_mode;

        // Log specific fault transitions
        if (current_mode_ == FusionMode::NLMS_FAULT) {
            log_fault_event(FusionMode::NLMS_FAULT,
                "Reference (error) mic dropped out or drift exceeded tolerance. Falling back to AI-only path.");
        } else if (current_mode_ == FusionMode::DEGRADED) {
            log_fault_event(FusionMode::DEGRADED,
                "AI speech enhancement inference failing. Falling back to NLMS residual-only path.");
        } else if (current_mode_ == FusionMode::ERROR) {
            log_fault_event(FusionMode::ERROR,
                "Audio hardware unavailable or dual microphone paths simultaneously failed.");
        }
    }

    // 2. Compute Target Impulse Gain (always calculated for protection)
    const float raw_prob = std::clamp(input.impulse_probability, 0.0f, 1.0f);
    float target_gain = 1.0f - (config_.impulse_beta * raw_prob);
    target_gain = std::max(target_gain, config_.impulse_gain_floor);

    // 3. Dispatch to explicit state handler
    AudioFrame out_frame;
    switch (current_mode_) {
        case FusionMode::NORMAL:
            out_frame = process_normal(input, target_gain);
            break;
        case FusionMode::LOW_CONFIDENCE:
            out_frame = process_low_confidence(input, target_gain);
            break;
        case FusionMode::IMPULSE:
            out_frame = process_impulse(input, target_gain);
            break;
        case FusionMode::DEGRADED:
            out_frame = process_degraded(input, target_gain);
            break;
        case FusionMode::NLMS_FAULT:
            out_frame = process_nlms_fault(input, target_gain);
            break;
        case FusionMode::BYPASS:
            out_frame = process_bypass(input);
            break;
        case FusionMode::ERROR:
            out_frame = process_error(input);
            break;
    }

    out_frame.frame_index = frame_count_;
    out_frame.timestamp_ns = input.ai_output.timestamp_ns != 0 
        ? input.ai_output.timestamp_ns 
        : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    // Record processing latency
    const auto end_time = std::chrono::steady_clock::now();
    last_processing_time_us_ = std::chrono::duration<double, std::micro>(end_time - start_time).count();

    return out_frame;
}

float FusionController::compute_dynamic_lambda(float ai_conf, float vad_prob, bool low_confidence_mode) const noexcept {
    const float conf = std::clamp(ai_conf, 0.0f, 1.0f);
    const float vad = std::clamp(vad_prob, 0.0f, 1.0f);
    
    // Dynamic lambda computation: lambda = clamp(ai_confidence * vad_probability, lambda_min, lambda_max)
    float lambda = std::clamp(conf * vad, config_.lambda_min, config_.lambda_max);

    if (low_confidence_mode) {
        // In LOW_CONFIDENCE mode, scale lambda down to lean heavily on NLMS residual output
        lambda = std::clamp(lambda * config_.low_confidence_scale, config_.lambda_min, 
                            config_.lambda_min + (config_.lambda_max - config_.lambda_min) * 0.25f);
    }

    return lambda;
}

void FusionController::update_gain_envelope(float target_gain) {
    previous_envelope_gain_ = current_envelope_gain_;

    if (target_gain < current_envelope_gain_) {
        // Fast attack phase: transient onset
        current_envelope_gain_ = (config_.attack_alpha * target_gain) + 
                                 ((1.0f - config_.attack_alpha) * current_envelope_gain_);
    } else {
        // Slower recovery / release phase: transient decaying
        current_envelope_gain_ = (config_.release_alpha * target_gain) + 
                                 ((1.0f - config_.release_alpha) * current_envelope_gain_);
    }

    // Guarantee configurable gain floor > 0: audio never hard mutes
    current_envelope_gain_ = std::clamp(current_envelope_gain_, config_.impulse_gain_floor, 1.0f);
}

void FusionController::apply_gain_envelope(AudioFrame& frame, float prev_gain, float curr_gain) {
    const size_t n_samples = frame.size();
    if (n_samples == 0) return;

    if (n_samples == 1) {
        frame[0] *= curr_gain;
        return;
    }

    // Linear sample-accurate interpolation across frame to avoid boundary clicks
    const float step = (curr_gain - prev_gain) / static_cast<float>(n_samples - 1);
    float gain = prev_gain;
    for (size_t n = 0; n < n_samples; ++n) {
        frame[n] *= gain;
        gain += step;
    }
}

AudioFrame FusionController::process_normal(const FusionInput& input, float target_gain) {
    const size_t n_samples = std::max(input.ai_output.size(), input.nlms_output.size());
    AudioFrame out(n_samples, 0.0f);

    // Compute dynamic lambda
    current_lambda_ = compute_dynamic_lambda(input.ai_confidence, input.vad_probability, false);

    // Fusion rule: s_hat[n] = lambda * s_AI[n] + (1 - lambda) * s_NLMS[n]
    for (size_t n = 0; n < n_samples; ++n) {
        const float s_ai = (n < input.ai_output.size()) ? input.ai_output[n] : 0.0f;
        const float s_nlms = (n < input.nlms_output.size()) ? input.nlms_output[n] : 0.0f;
        out[n] = (current_lambda_ * s_ai) + ((1.0f - current_lambda_) * s_nlms);
    }

    // Impulse protection gain envelope
    update_gain_envelope(target_gain);
    apply_gain_envelope(out, previous_envelope_gain_, current_envelope_gain_);

    return out;
}

AudioFrame FusionController::process_low_confidence(const FusionInput& input, float target_gain) {
    const size_t n_samples = std::max(input.ai_output.size(), input.nlms_output.size());
    AudioFrame out(n_samples, 0.0f);

    // Lean more heavily on NLMS output (lower lambda)
    current_lambda_ = compute_dynamic_lambda(input.ai_confidence, input.vad_probability, true);

    for (size_t n = 0; n < n_samples; ++n) {
        const float s_ai = (n < input.ai_output.size()) ? input.ai_output[n] : 0.0f;
        const float s_nlms = (n < input.nlms_output.size()) ? input.nlms_output[n] : 0.0f;
        out[n] = (current_lambda_ * s_ai) + ((1.0f - current_lambda_) * s_nlms);
    }

    update_gain_envelope(target_gain);
    apply_gain_envelope(out, previous_envelope_gain_, current_envelope_gain_);

    return out;
}

AudioFrame FusionController::process_impulse(const FusionInput& input, float target_gain) {
    const size_t n_samples = std::max(input.ai_output.size(), input.nlms_output.size());
    AudioFrame out(n_samples, 0.0f);

    current_lambda_ = compute_dynamic_lambda(input.ai_confidence, input.vad_probability, false);

    for (size_t n = 0; n < n_samples; ++n) {
        const float s_ai = (n < input.ai_output.size()) ? input.ai_output[n] : 0.0f;
        const float s_nlms = (n < input.nlms_output.size()) ? input.nlms_output[n] : 0.0f;
        if (input.nlms_available) {
            out[n] = (current_lambda_ * s_ai) + ((1.0f - current_lambda_) * s_nlms);
        } else {
            out[n] = s_ai;
        }
    }

    // Apply full impulse gain envelope
    update_gain_envelope(target_gain);
    apply_gain_envelope(out, previous_envelope_gain_, current_envelope_gain_);

    return out;
}

AudioFrame FusionController::process_degraded(const FusionInput& input, float target_gain) {
    // AI inference failing -> output NLMS-only (never silence, since NLMS is active)
    current_lambda_ = 0.0f;
    const size_t n_samples = !input.nlms_output.empty() ? input.nlms_output.size() : config_.default_frame_size;
    AudioFrame out(n_samples, 0.0f);

    for (size_t n = 0; n < n_samples; ++n) {
        out[n] = (n < input.nlms_output.size()) ? input.nlms_output[n] : 0.0f;
    }

    // Impulse protection remains active
    update_gain_envelope(target_gain);
    apply_gain_envelope(out, previous_envelope_gain_, current_envelope_gain_);

    return out;
}

AudioFrame FusionController::process_nlms_fault(const FusionInput& input, float target_gain) {
    // Reference mic dropped out / drift error -> fall back to AI-only
    current_lambda_ = 1.0f;
    const size_t n_samples = !input.ai_output.empty() ? input.ai_output.size() : config_.default_frame_size;
    AudioFrame out(n_samples, 0.0f);

    for (size_t n = 0; n < n_samples; ++n) {
        out[n] = (n < input.ai_output.size()) ? input.ai_output[n] : 0.0f;
    }

    // Impulse protection remains active on AI fallback path
    update_gain_envelope(target_gain);
    apply_gain_envelope(out, previous_envelope_gain_, current_envelope_gain_);

    return out;
}

AudioFrame FusionController::process_bypass(const FusionInput& input) {
    // Input routed directly to output with zero modification / trivial delay
    if (!input.raw_input.empty()) {
        return input.raw_input;
    }
    if (!input.nlms_output.empty()) {
        return input.nlms_output;
    }
    if (!input.ai_output.empty()) {
        return input.ai_output;
    }
    return AudioFrame(config_.default_frame_size, 0.0f);
}

AudioFrame FusionController::process_error(const FusionInput& input) {
    // Both hardware paths unavailable: clear error reported, produce silent buffer
    const size_t n_samples = !input.ai_output.empty() ? input.ai_output.size() 
                           : (!input.nlms_output.empty() ? input.nlms_output.size() : config_.default_frame_size);
    return AudioFrame(n_samples, 0.0f);
}

void FusionController::log_fault_event(FusionMode mode, const std::string& reason) {
    ++fault_count_;
    last_fault_message_ = reason;

    if (config_.log_faults) {
        std::cerr << "[FusionController][FAULT] Frame #" << frame_count_ 
                  << " Transition to " << to_string(mode) 
                  << " - Reason: " << reason << std::endl;
    }

    if (fault_logger_) {
        fault_logger_(mode, reason, frame_count_);
    }
}

} // namespace fusion
} // namespace noiselessx
