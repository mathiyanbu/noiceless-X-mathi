#include <gtest/gtest.h>
#include "embedded/fusion/fusion_controller.hpp"
#include <vector>
#include <cmath>
#include <chrono>

using namespace noiselessx::fusion;

namespace {

AudioFrame create_test_frame(size_t size, float start_val, float step = 0.01f) {
    AudioFrame frame(size);
    for (size_t i = 0; i < size; ++i) {
        frame[i] = start_val + static_cast<float>(i) * step;
    }
    return frame;
}

} // anonymous namespace

// 1. Normal Mode: AI + NLMS both active, dynamic lambda blending per rule
TEST(FusionControllerTest, NormalModeDynamicLambdaBlending) {
    FusionController controller;
    const size_t frame_size = 80;

    AudioFrame ai_frame = create_test_frame(frame_size, 1.0f, 0.0f);    // constant 1.0
    AudioFrame nlms_frame = create_test_frame(frame_size, 0.0f, 0.0f);  // constant 0.0

    // High confidence + active speech -> lambda should be ~0.80
    FusionInput input{
        .ai_output = ai_frame,
        .nlms_output = nlms_frame,
        .ai_confidence = 0.90f,
        .impulse_probability = 0.0f,
        .vad_probability = 0.88f,
        .nlms_available = true
    };

    AudioFrame out = controller.fuse(input);

    EXPECT_EQ(controller.get_current_mode(), FusionMode::NORMAL);
    EXPECT_NEAR(controller.get_current_lambda(), 0.90f * 0.88f, 0.02f);

    // With zero impulse, envelope gain is 1.0
    // Output = lambda * 1.0 + (1 - lambda) * 0.0 = lambda
    for (size_t i = 0; i < frame_size; ++i) {
        EXPECT_NEAR(out[i], controller.get_current_lambda(), 0.02f);
    }
}

// 2. Low Confidence Mode: Leans heavily on NLMS
TEST(FusionControllerTest, LowConfidenceModeLeansOnNlms) {
    FusionController controller;
    const size_t frame_size = 80;

    AudioFrame ai_frame = create_test_frame(frame_size, 1.0f, 0.0f);    // constant 1.0
    AudioFrame nlms_frame = create_test_frame(frame_size, 0.0f, 0.0f);  // constant 0.0

    // Low AI confidence (0.25 < 0.45 threshold)
    FusionInput input{
        .ai_output = ai_frame,
        .nlms_output = nlms_frame,
        .ai_confidence = 0.25f,
        .impulse_probability = 0.0f,
        .vad_probability = 0.90f,
        .nlms_available = true
    };

    AudioFrame out = controller.fuse(input);

    EXPECT_EQ(controller.get_current_mode(), FusionMode::LOW_CONFIDENCE);
    // Lambda should be scaled down significantly
    EXPECT_LT(controller.get_current_lambda(), 0.25f);
    // Output should be much closer to NLMS (0.0) than AI (1.0)
    for (size_t i = 0; i < frame_size; ++i) {
        EXPECT_LT(out[i], 0.25f);
    }
}

// 3. Impulse Mode: Fast attack attenuation, never hard mutes (floor > 0), slower recovery
TEST(FusionControllerTest, ImpulseModeAttenuatesWithoutHardMuteAndRecovers) {
    FusionController controller;
    const size_t frame_size = 80;

    AudioFrame ai_frame = create_test_frame(frame_size, 1.0f, 0.0f);
    AudioFrame nlms_frame = create_test_frame(frame_size, 1.0f, 0.0f);

    // Severe impulse noise (p = 1.0)
    FusionInput impulse_input{
        .ai_output = ai_frame,
        .nlms_output = nlms_frame,
        .ai_confidence = 0.8f,
        .impulse_probability = 1.0f,
        .vad_probability = 0.5f,
        .nlms_available = true
    };

    AudioFrame out_impulse = controller.fuse(impulse_input);

    EXPECT_EQ(controller.get_current_mode(), FusionMode::IMPULSE);

    // Gain must drop fast (fast attack)
    float gain_during_impulse = controller.get_current_gain_envelope();
    EXPECT_LT(gain_during_impulse, 0.5f);

    // Audio MUST NEVER hard mute: gain floor strictly > 0 (e.g. >= 0.10)
    EXPECT_GE(gain_during_impulse, controller.get_config().impulse_gain_floor);
    for (size_t i = 0; i < frame_size; ++i) {
        EXPECT_GE(out_impulse[i], controller.get_config().impulse_gain_floor * 0.99f);
    }

    // Next frames: impulse disappears (p = 0.0)
    FusionInput quiet_input{
        .ai_output = ai_frame,
        .nlms_output = nlms_frame,
        .ai_confidence = 0.8f,
        .impulse_probability = 0.0f,
        .vad_probability = 0.5f,
        .nlms_available = true
    };

    float previous_gain = gain_during_impulse;
    // Over 15 frames, verify gradual recovery (slow release)
    for (int frame = 0; frame < 15; ++frame) {
        controller.fuse(quiet_input);
        float current_gain = controller.get_current_gain_envelope();
        EXPECT_GT(current_gain, previous_gain); // Gain is monotonically recovering
        previous_gain = current_gain;
    }
    // Eventually approaches 1.0
    EXPECT_GT(controller.get_current_gain_envelope(), 0.85f);
}

// 4. Degraded Mode: AI inference failing -> outputs NLMS-only (never silence)
TEST(FusionControllerTest, DegradedModeOutputsNlmsOnly) {
    FusionController controller;
    const size_t frame_size = 80;

    AudioFrame ai_frame = create_test_frame(frame_size, 0.0f, 0.0f);    // Silence or corrupt
    AudioFrame nlms_frame = create_test_frame(frame_size, 0.42f, 0.0f); // NLMS residual signal

    FusionInput input{
        .ai_output = ai_frame,
        .nlms_output = nlms_frame,
        .ai_confidence = 0.0f,
        .impulse_probability = 0.0f,
        .vad_probability = 0.0f,
        .nlms_available = true,
        .ai_available = false // AI inference failing
    };

    AudioFrame out = controller.fuse(input);

    EXPECT_EQ(controller.get_current_mode(), FusionMode::DEGRADED);
    EXPECT_GT(controller.get_fault_count(), 0u);
    EXPECT_FALSE(controller.get_last_fault_message().empty());

    // Never silence: output MUST equal NLMS residual
    for (size_t i = 0; i < frame_size; ++i) {
        EXPECT_NEAR(out[i], 0.42f, 1e-4f);
    }
}

// 5. NLMS Fault Mode: Reference mic drops out -> falls back to AI-only, logs fault
TEST(FusionControllerTest, NlmsFaultModeFallsBackToAiOnlyAndLogsFault) {
    FusionController controller;
    const size_t frame_size = 80;

    AudioFrame ai_frame = create_test_frame(frame_size, 0.77f, 0.0f);
    AudioFrame nlms_frame = create_test_frame(frame_size, 0.0f, 0.0f);

    FusionInput input{
        .ai_output = ai_frame,
        .nlms_output = nlms_frame,
        .ai_confidence = 0.95f,
        .impulse_probability = 0.0f,
        .vad_probability = 0.90f,
        .nlms_available = false // Reference mic dropout or drift tolerance exceeded
    };

    AudioFrame out = controller.fuse(input);

    EXPECT_EQ(controller.get_current_mode(), FusionMode::NLMS_FAULT);
    EXPECT_EQ(controller.get_fault_count(), 1u);
    EXPECT_NE(controller.get_last_fault_message().find("Reference (error) mic"), std::string::npos);

    // Fallback to AI-only
    for (size_t i = 0; i < frame_size; ++i) {
        EXPECT_NEAR(out[i], 0.77f, 1e-4f);
    }
}

// 6. Bypass Mode: Routes input directly to output, minimal trivial latency
TEST(FusionControllerTest, BypassModeRoutesDirectlyWithTrivialLatency) {
    FusionController controller;
    const size_t frame_size = 80;

    AudioFrame raw_audio = create_test_frame(frame_size, 0.123f, 0.005f);
    AudioFrame dummy_ai(frame_size, 0.0f);
    AudioFrame dummy_nlms(frame_size, 0.0f);

    FusionInput input{
        .ai_output = dummy_ai,
        .nlms_output = dummy_nlms,
        .raw_input = raw_audio,
        .bypass_requested = true // Operator trigger
    };

    // Measure latency
    const auto t0 = std::chrono::high_resolution_clock::now();
    AudioFrame out = controller.fuse(input);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    EXPECT_EQ(controller.get_current_mode(), FusionMode::BYPASS);
    EXPECT_LT(elapsed_us, 100.0); // Trivial pass-through latency (< 100 us, typically < 10 us)

    // Bit-for-bit identical pass-through
    ASSERT_EQ(out.size(), frame_size);
    for (size_t i = 0; i < frame_size; ++i) {
        EXPECT_FLOAT_EQ(out[i], raw_audio[i]);
    }
}

// 7. Error Mode: Both mics/hardware unavailable -> no processing, clear error reported
TEST(FusionControllerTest, ErrorModeHardwareUnavailable) {
    FusionController controller;
    const size_t frame_size = 80;

    FusionInput input{
        .ai_output = AudioFrame(frame_size, 0.0f),
        .nlms_output = AudioFrame(frame_size, 0.0f),
        .nlms_available = false,
        .ai_available = false,
        .hardware_available = false // Total hardware failure
    };

    AudioFrame out = controller.fuse(input);

    EXPECT_EQ(controller.get_current_mode(), FusionMode::ERROR);
    EXPECT_GT(controller.get_fault_count(), 0u);
    EXPECT_NE(controller.get_last_fault_message().find("hardware unavailable"), std::string::npos);

    // Silent output
    for (size_t i = 0; i < frame_size; ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
}

// 8. Full Sequential State Machine Transitions
TEST(FusionControllerTest, FullStateTransitionsSequence) {
    FusionController controller;
    const size_t frame_size = 80;
    AudioFrame frame(frame_size, 0.5f);

    // Sequence of steps simulating real operational events:
    // 1. Start NORMAL
    FusionInput step1{.ai_output = frame, .nlms_output = frame, .ai_confidence = 0.9f, .impulse_probability = 0.1f, .vad_probability = 0.8f};
    controller.fuse(step1);
    EXPECT_EQ(controller.get_current_mode(), FusionMode::NORMAL);

    // 2. Confidence drops -> LOW_CONFIDENCE
    FusionInput step2{.ai_output = frame, .nlms_output = frame, .ai_confidence = 0.2f, .impulse_probability = 0.1f, .vad_probability = 0.8f};
    controller.fuse(step2);
    EXPECT_EQ(controller.get_current_mode(), FusionMode::LOW_CONFIDENCE);

    // 3. Transient slam -> IMPULSE
    FusionInput step3{.ai_output = frame, .nlms_output = frame, .ai_confidence = 0.9f, .impulse_probability = 0.95f, .vad_probability = 0.8f};
    controller.fuse(step3);
    EXPECT_EQ(controller.get_current_mode(), FusionMode::IMPULSE);

    // 4. Return to normal speech
    controller.fuse(step1);
    EXPECT_EQ(controller.get_current_mode(), FusionMode::NORMAL);

    // 5. Reference mic unplugged / drift -> NLMS_FAULT
    FusionInput step5{.ai_output = frame, .nlms_output = frame, .ai_confidence = 0.9f, .impulse_probability = 0.1f, .vad_probability = 0.8f, .nlms_available = false};
    controller.fuse(step5);
    EXPECT_EQ(controller.get_current_mode(), FusionMode::NLMS_FAULT);

    // 6. Reference mic reconnected -> NORMAL
    controller.fuse(step1);
    EXPECT_EQ(controller.get_current_mode(), FusionMode::NORMAL);

    // 7. AI process crashes -> DEGRADED
    FusionInput step7{.ai_output = frame, .nlms_output = frame, .ai_confidence = 0.0f, .impulse_probability = 0.0f, .vad_probability = 0.0f, .ai_available = false};
    controller.fuse(step7);
    EXPECT_EQ(controller.get_current_mode(), FusionMode::DEGRADED);

    // 8. Operator activates bypass
    controller.set_bypass(true);
    controller.fuse(step1);
    EXPECT_EQ(controller.get_current_mode(), FusionMode::BYPASS);
}
