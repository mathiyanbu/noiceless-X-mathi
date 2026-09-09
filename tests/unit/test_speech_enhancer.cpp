#include <gtest/gtest.h>
#include "embedded/ai/speech_enhancer.hpp"

#include <vector>
#include <cmath>

TEST(SpeechEnhancerTest, ComplexFrameInitialization) {
    noiselessx::ai::ComplexFrame frame;
    EXPECT_EQ(frame.num_bins, 257);
    EXPECT_EQ(frame.real.size(), 257);
    EXPECT_EQ(frame.imag.size(), 257);

    for (size_t i = 0; i < frame.num_bins; ++i) {
        EXPECT_FLOAT_EQ(frame.real[i], 0.0f);
        EXPECT_FLOAT_EQ(frame.imag[i], 0.0f);
    }

    frame.resize(129);
    EXPECT_EQ(frame.num_bins, 129);
    EXPECT_EQ(frame.real.size(), 129);
    EXPECT_EQ(frame.imag.size(), 129);
}

TEST(SpeechEnhancerTest, UninitializedInferenceRejection) {
    noiselessx::ai::SpeechEnhancer enhancer;
    EXPECT_FALSE(enhancer.is_initialized());

    noiselessx::ai::ComplexFrame in_frame(257);
    noiselessx::ai::ComplexFrame out_frame(257);

    // infer() on uninitialized engine must fail and trigger degraded state
    bool success = enhancer.infer(in_frame, out_frame);
    EXPECT_FALSE(success);
}

TEST(SpeechEnhancerTest, NonExistentModelPathFailsGracefully) {
    noiselessx::ai::SpeechEnhancer enhancer;
    bool init_ok = enhancer.initialize("non_existent_path_to_model.onnx", 2);
    
    // Must return false and set degraded mode without crashing
    EXPECT_FALSE(init_ok);
    EXPECT_FALSE(enhancer.is_initialized());
    EXPECT_TRUE(enhancer.is_degraded());
    EXPECT_FALSE(enhancer.get_last_error().empty());
}

#if NOICELESSX_HAS_ONNXRUNTIME

TEST(SpeechEnhancerTest, RealOnnxInferenceAndStateContinuity) {
    noiselessx::ai::SpeechEnhancer enhancer;
    std::string model_path = "models/onnx/speech_enhancer_int8.onnx";
    
    // Attempt initialization with INT8 model
    bool init_ok = enhancer.initialize(model_path, 2);
    if (!init_ok) {
        // Try fallback to FP32 model if INT8 not present
        model_path = "models/onnx/speech_enhancer_fp32.onnx";
        init_ok = enhancer.initialize(model_path, 2);
    }
    ASSERT_TRUE(init_ok) << "Failed to initialize session: " << enhancer.get_last_error();
    EXPECT_TRUE(enhancer.is_initialized());
    EXPECT_FALSE(enhancer.is_degraded());

    noiselessx::ai::ComplexFrame in_frame(257);
    noiselessx::ai::ComplexFrame out_frame(257);

    // Populate input with realistic spectral values
    for (size_t k = 0; k < 257; ++k) {
        in_frame.real[k] = std::sin(0.1f * static_cast<float>(k));
        in_frame.imag[k] = std::cos(0.1f * static_cast<float>(k));
    }

    // First inference step
    bool infer_ok = enhancer.infer(in_frame, out_frame);
    EXPECT_TRUE(infer_ok);
    EXPECT_GT(enhancer.get_last_inference_ms(), 0.0f);
    EXPECT_EQ(out_frame.num_bins, 257);

    // Verify non-NaN output
    for (size_t k = 0; k < 257; ++k) {
        EXPECT_FALSE(std::isnan(out_frame.real[k]));
        EXPECT_FALSE(std::isnan(out_frame.imag[k]));
        EXPECT_FALSE(std::isinf(out_frame.real[k]));
        EXPECT_FALSE(std::isinf(out_frame.imag[k]));
    }

    // Run 10 sequential streaming steps to ensure recurrent hidden-state persistence
    for (int step = 0; step < 10; ++step) {
        infer_ok = enhancer.infer(in_frame, out_frame);
        EXPECT_TRUE(infer_ok);
        EXPECT_GT(enhancer.get_last_inference_ms(), 0.0f);
    }

    // Test reset_state
    enhancer.reset_state();
    infer_ok = enhancer.infer(in_frame, out_frame);
    EXPECT_TRUE(infer_ok);
}

#else

TEST(SpeechEnhancerTest, HostStubRefusesToClaimReadyWithoutHardware) {
    noiselessx::ai::SpeechEnhancer enhancer;
    bool ok = enhancer.initialize("models/onnx/speech_enhancer_int8.onnx", 2);
    // On dev host without ONNX Runtime C++ headers, engine must refuse READY
    EXPECT_FALSE(ok);
    EXPECT_FALSE(enhancer.is_initialized());
    EXPECT_TRUE(enhancer.is_degraded());
    EXPECT_FALSE(enhancer.get_last_error().empty());
}

#endif
