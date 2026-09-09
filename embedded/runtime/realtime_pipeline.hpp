#pragma once

#include "system_metrics.hpp"
#include "embedded/audio/audio_types.hpp"
#include "embedded/audio/alsa_backend.hpp"
#include "embedded/dsp/dc_blocker.hpp"
#include "embedded/dsp/biquad_filter.hpp"
#include "embedded/dsp/stft.hpp"
#include "embedded/dsp/vad.hpp"
#include "embedded/dsp/nlms.hpp"
#include "embedded/impulse/impulse_detector.hpp"
#include "embedded/ai/speech_enhancer.hpp"
#include "embedded/fusion/fusion_controller.hpp"

#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <memory>

namespace noiselessx::runtime {

/**
 * @brief Real-time performance metrics and latency breakdown.
 * All numbers are measured from real timers and kernel counters (never fabricated).
 */
struct PipelineTelemetry {
    // Stage Latencies (microseconds)
    double capture_us{0.0};
    double preprocessing_us{0.0};
    double stft_us{0.0};
    double ai_inference_us{0.0};
    double nlms_us{0.0};
    double fusion_us{0.0};
    double istft_us{0.0};
    double playback_queue_us{0.0};
    double total_processing_us{0.0};
    double end_to_end_latency_ms{0.0};

    // Real-Time Factor: processing_time / hop_duration (Target: < 0.50 on Raspberry Pi 4/5)
    double rtf{0.0};

    // Counters
    uint64_t processed_frames{0};
    uint64_t dropped_frames{0};
    uint64_t alsa_xruns_primary{0};
    uint64_t alsa_xruns_reference{0};
    uint64_t alsa_xruns_playback{0};

    // Clock skew / drift
    double drift_ms{0.0};
    int64_t drift_samples{0};
    bool drift_warning{false};

    // Fusion Controller state
    noiselessx::fusion::FusionMode fusion_mode{noiselessx::fusion::FusionMode::NORMAL};
    float current_lambda{0.5f};
    float impulse_envelope_gain{1.0f};
    float ai_confidence{1.0f};
    float impulse_probability{0.0f};
    float vad_probability{0.0f};

    // System metrics
    std::vector<float> cpu_per_core;
    float overall_cpu_pct{0.0f};
    float cpu_temperature_c{0.0f};
    bool temp_available{false};
};

/**
 * @brief Master Real-Time Audio Pipeline for Raspberry Pi Dual-Microphone Operation.
 * 
 * Orchestrates:
 * - Thread 1 & 2 (Capture): Real ALSA capture for Headphone Mic (Primary) + Error Mic (Reference)
 *   into bounded lock-free SPSC ring buffers.
 * - Thread 3 (Processing): Synchronized dequeue, drift check, DC removal, 80Hz high-pass,
 *   STFT analysis, parallel AI + NLMS branches, impulse detection, VAD, Phase 9 Fusion Controller,
 *   and iSTFT synthesis.
 * - Thread 4 (Playback): Real ALSA playback to Headphone output with XRUN management.
 * - Realtime thread priorities (SCHED_FIFO / SCHED_RR with graceful fallback).
 * - Comprehensive zero-mock telemetry monitoring.
 */
class RealtimePipeline {
public:
    struct PipelineConfig {
        noiselessx::audio::AudioConfig audio;
        noiselessx::impulse::ImpulseConfig impulse;
        noiselessx::fusion::FusionConfig fusion;
        std::string onnx_model_path{"models/onnx/speech_enhancer_int8.onnx"};
        int ai_threads{2};
        int nlms_filter_length{256};
        float nlms_learning_rate{0.05f};
        float nlms_epsilon{1e-6f};
        float highpass_cutoff_hz{80.0f};
        bool use_sched_fifo{true};
        int audio_thread_priority{80};
        int processing_thread_priority{75};
    };

    explicit RealtimePipeline(const PipelineConfig& config);
    ~RealtimePipeline();

    // Non-copyable, non-movable
    RealtimePipeline(const RealtimePipeline&) = delete;
    RealtimePipeline& operator=(const RealtimePipeline&) = delete;
    RealtimePipeline(RealtimePipeline&&) = delete;
    RealtimePipeline& operator=(RealtimePipeline&&) = delete;

    /**
     * @brief Initialize all audio hardware, DSP stages, ONNX Runtime session, and state machine.
     * @return true on success; prints descriptive error and returns false if devices/models fail.
     */
    bool initialize();

    /**
     * @brief Start all real-time audio capture, processing, and playback threads.
     */
    bool start();

    /**
     * @brief Stop all threads cleanly and flush audio queues.
     */
    void stop();

    /**
     * @brief Check if the pipeline is actively running.
     */
    [[nodiscard]] bool is_running() const noexcept { return is_running_.load(); }

    /**
     * @brief Query current live telemetry (zero fabrication).
     */
    [[nodiscard]] PipelineTelemetry get_telemetry();

    /**
     * @brief Set manual bypass mode.
     */
    void set_bypass(bool bypass);

    [[nodiscard]] const PipelineConfig& get_config() const noexcept { return config_; }

private:
    void processing_thread_loop();
    void configure_thread_priority(std::thread& th, int priority, const std::string& name);

    PipelineConfig config_;
    std::atomic<bool> is_running_{false};
    std::atomic<bool> should_stop_{false};

    // Subsystem components
    noiselessx::audio::AlsaBackend alsa_backend_;
    noiselessx::dsp::DcBlocker dc_blocker_;
    noiselessx::dsp::BiquadFilter high_pass_;
    noiselessx::dsp::StftEngine stft_primary_;
    noiselessx::dsp::FeatureBasedVad vad_;
    noiselessx::impulse::ImpulseDetector impulse_detector_;
    noiselessx::ai::SpeechEnhancer speech_enhancer_;
    noiselessx::dsp::NLMSFilter nlms_;
    noiselessx::fusion::FusionController fusion_controller_;
    SystemMetrics system_metrics_;

    // Threading
    std::thread processing_thread_;

    // Telemetry tracking state
    PipelineTelemetry telemetry_;
    std::chrono::steady_clock::time_point last_system_sample_time_;
};

} // namespace noiselessx::runtime
