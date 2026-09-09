#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace noiselessx::ai {

/**
 * @brief Complex spectral frame structure containing 257 bins (for N=512 FFT at 16kHz).
 * Matches real and imaginary spectral channels expected by ComplexCRN model.
 */
struct ComplexFrame {
    std::vector<float> real;
    std::vector<float> imag;
    size_t num_bins{257};

    ComplexFrame() : real(257, 0.0f), imag(257, 0.0f), num_bins(257) {}
    explicit ComplexFrame(size_t bins) : real(bins, 0.0f), imag(bins, 0.0f), num_bins(bins) {}

    void resize(size_t bins) {
        num_bins = bins;
        real.assign(bins, 0.0f);
        imag.assign(bins, 0.0f);
    }
};

/**
 * @brief Real-time C++ Speech Enhancement Inference Engine using ONNX Runtime C++ API.
 * 
 * Replaces the original TensorRT backend with a CPU Execution Provider tuned for
 * ARM NEON SIMD on Raspberry Pi 4/5 (Cortex-A72 / Cortex-A76).
 * 
 * Requirements:
 * - Load ONNX Runtime session ONCE at startup (Ort::Env, Ort::Session), never per-frame.
 * - Intra-op thread count tuned from config (e.g. 2 threads to leave headroom for audio/DSP).
 * - Persistent recurrent hidden state: carries output state across sequential streaming frames.
 * - Zero allocations in per-frame infer(): preallocated memory buffers and Ort::Value tensors.
 * - Non-crashing error recovery: catches ORT exceptions and signals DEGRADED mode.
 * - Accurate microsecond-precision latency measurement per infer() call.
 */
class SpeechEnhancer {
public:
    SpeechEnhancer();
    ~SpeechEnhancer();

    // Non-copyable to protect session and preallocated tensor handles
    SpeechEnhancer(const SpeechEnhancer&) = delete;
    SpeechEnhancer& operator=(const SpeechEnhancer&) = delete;

    // Movable
    SpeechEnhancer(SpeechEnhancer&&) noexcept;
    SpeechEnhancer& operator=(SpeechEnhancer&&) noexcept;

    /**
     * @brief Initialize ONNX Runtime session once at startup.
     * @param onnx_model_path Path to the .onnx model file (e.g. speech_enhancer_int8.onnx)
     * @param num_threads Intra-op thread count (defaults to 2 for Pi 4/5 headroom)
     * @return true if session initialized successfully, false otherwise
     */
    bool initialize(const std::string& onnx_model_path, int num_threads = 2);

    /**
     * @brief Run single-frame streaming inference.
     * @param input Input ComplexFrame (real and imaginary STFT bins, size 257)
     * @param output Output ComplexFrame (enhanced real and imaginary STFT bins, size 257)
     * @return true on success; false if inference failed (triggers DEGRADED mode, no crash)
     */
    bool infer(const ComplexFrame& input, ComplexFrame& output);

    /**
     * @brief Reset recurrent hidden state to zeros.
     * Call ONLY on new session, audio dropout recovery, or operator restart.
     */
    void reset_state();

    /**
     * @brief Latency of the most recent infer() call in milliseconds.
     */
    float get_last_inference_ms() const;

    /**
     * @brief Check if engine has entered degraded mode due to an inference error.
     */
    bool is_degraded() const noexcept;

    /**
     * @brief Check if session is successfully initialized and ready.
     */
    bool is_initialized() const noexcept;

    /**
     * @brief Returns the active execution provider (e.g. "XNNPACK" or "CPUExecutionProvider").
     */
    const std::string& get_execution_provider() const noexcept;

    /**
     * @brief Get last error or status message.
     */
    const std::string& get_last_error() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace noiselessx::ai

// Global aliases matching user request specification
using ComplexFrame = noiselessx::ai::ComplexFrame;
using SpeechEnhancer = noiselessx::ai::SpeechEnhancer;
