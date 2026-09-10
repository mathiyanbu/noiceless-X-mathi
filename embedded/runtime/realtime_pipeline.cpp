#include "realtime_pipeline.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace noiselessx {
namespace runtime {

RealtimePipeline::RealtimePipeline(const PipelineConfig& config)
    : config_(config),
      dc_blocker_(0.995f),
      high_pass_(config.highpass_cutoff_hz, config.audio.sample_rate, 0.707f),
      stft_primary_(config.audio.fft_size, config.audio.period_size / 2), // e.g. 512, 80
      vad_(static_cast<float>(config.audio.sample_rate)),
      impulse_detector_(config.impulse),
      nlms_(config.nlms_filter_length, config.nlms_learning_rate, config.nlms_epsilon),
      fusion_controller_(config.fusion)
{
    last_system_sample_time_ = std::chrono::steady_clock::now();
}

RealtimePipeline::~RealtimePipeline() {
    stop();
}

bool RealtimePipeline::initialize() {
    std::cout << "\n[RealtimePipeline] ================================================\n";
    std::cout << "[RealtimePipeline] Initializing Dual-Microphone Real-Time Subsystems\n";
    std::cout << "[RealtimePipeline] ================================================\n";

    // 1. Initialize ALSA Backend
    std::cout << "[RealtimePipeline] Step 1: Initializing ALSA Hardware Audio Subsystem...\n";
    if (!alsa_backend_.initialize(config_.audio)) {
        std::cerr << "[RealtimePipeline] Critical Error: Failed to initialize ALSA audio hardware.\n"
                  << "  Primary Device:   " << config_.audio.primary_device << "\n"
                  << "  Reference Device: " << config_.audio.reference_device << "\n"
                  << "  Output Device:    " << config_.audio.output_device << "\n";
        return false;
    }
    std::cout << "[RealtimePipeline]   ALSA Subsystem initialized successfully.\n";

    // 2. Initialize DSP Engines
    std::cout << "[RealtimePipeline] Step 2: Configuring DSP Preprocessing & STFT Filters...\n";
    const size_t hop_size = (config_.audio.period_size > 0) ? (config_.audio.period_size / 2) : 80;
    stft_primary_.initialize(config_.audio.fft_size, hop_size);
    dc_blocker_.reset();
    high_pass_.configure(config_.highpass_cutoff_hz, config_.audio.sample_rate, 0.707f);
    vad_.reset();
    nlms_.reset();
    std::cout << "[RealtimePipeline]   DSP components initialized (Hop: " << hop_size 
              << " samples, FFT: " << config_.audio.fft_size << ").\n";

    // 3. Initialize Impulsive Noise Detector
    std::cout << "[RealtimePipeline] Step 3: Initializing Impulsive Noise Detector...\n";
    if (!impulse_detector_.initialize_onnx(config_.impulse.onnx_model_path)) {
        std::cout << "[RealtimePipeline]   Impulse Detector fallback: Using calibrated logistic regression weights.\n";
    }

    // 4. Initialize ONNX Runtime Speech Enhancement Engine
    std::cout << "[RealtimePipeline] Step 4: Loading ONNX Runtime C++ Inference Engine...\n";
    std::cout << "[RealtimePipeline]   Model: " << config_.onnx_model_path 
              << " (Threads: " << config_.ai_threads << ")...\n";
    if (!speech_enhancer_.initialize(config_.onnx_model_path, config_.ai_threads)) {
        std::cout << "[RealtimePipeline]   Notice: SpeechEnhancer entering DEGRADED / fallback mode until model file is present.\n";
    }

    // 5. Initialize Fusion Controller
    std::cout << "[RealtimePipeline] Step 5: Initializing Multi-Mode Fusion Controller...\n";
    fusion_controller_.reset();
    fusion_controller_.set_config(config_.fusion);

    // Sample initial system telemetry
    system_metrics_.sample();

    std::cout << "[RealtimePipeline] Subsystem initialization complete.\n\n";
    return true;
}

void RealtimePipeline::configure_thread_priority(std::thread& th, int priority, const std::string& name) {
#if defined(__linux__)
    if (config_.use_sched_fifo) {
        sched_param param{};
        param.sched_priority = priority;
        if (pthread_setschedparam(th.native_handle(), SCHED_FIFO, &param) == 0) {
            std::cout << "[RealtimePipeline] Applied SCHED_FIFO priority " << priority << " to " << name << " thread.\n";
            return;
        }
        std::cout << "[RealtimePipeline] Notice: Could not set SCHED_FIFO priority for " << name 
                  << " (requires root / CAP_SYS_NICE). Running with standard priority.\n";
    }
#else
    (void)th;
    (void)priority;
    (void)name;
#endif
}

bool RealtimePipeline::start() {
    if (is_running_.load()) {
        return true;
    }

    std::cout << "[RealtimePipeline] Starting real-time audio threads...\n";

    reset();

    // 1. Start ALSA capture and playback threads
    if (!alsa_backend_.start()) {
        std::cerr << "[RealtimePipeline] Error: Failed to start ALSA audio capture/playback threads.\n";
        return false;
    }

    should_stop_.store(false);
    is_running_.store(true);

    // 2. Launch dedicated real-time processing thread
    processing_thread_ = std::thread(&RealtimePipeline::processing_thread_loop, this);
    configure_thread_priority(processing_thread_, config_.processing_thread_priority, "Processing");

    std::cout << "[RealtimePipeline] Pipeline started. Active threads: Capture (x2/x1), Processing (x1), Playback (x1).\n";
    return true;
}

void RealtimePipeline::stop() {
    if (!is_running_.load()) {
        return;
    }

    std::cout << "[RealtimePipeline] Stopping real-time pipeline...\n";
    should_stop_.store(true);

    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }

    alsa_backend_.stop();
    is_running_.store(false);
    std::cout << "[RealtimePipeline] Pipeline stopped cleanly.\n";
}

void RealtimePipeline::set_bypass(bool bypass) {
    fusion_controller_.set_bypass(bypass);
}

void RealtimePipeline::reset() {
    dc_blocker_.reset();
    high_pass_.configure(config_.highpass_cutoff_hz, config_.audio.sample_rate, 0.707f);
    stft_primary_.reset();
    vad_.reset();
    nlms_.reset();
    fusion_controller_.reset();
    speech_enhancer_.reset_state();
    telemetry_ = PipelineTelemetry{};
}

void RealtimePipeline::processing_thread_loop() {
    using namespace std::chrono;

    const size_t hop_size = stft_primary_.get_hop_size();
    std::vector<float> primary_preproc(hop_size, 0.0f);
    std::vector<float> nlms_out(hop_size, 0.0f);
    std::vector<float> ai_hop_out(hop_size, 0.0f);

    noiselessx::dsp::SpectralFrame spectral_frame;
    noiselessx::ai::ComplexFrame complex_in(stft_primary_.get_num_bins());
    noiselessx::ai::ComplexFrame complex_out(stft_primary_.get_num_bins());

    noiselessx::audio::AudioChunk<1024> primary_chunk;
    noiselessx::audio::AudioChunk<1024> reference_chunk;
    noiselessx::audio::AudioChunk<1024> playback_chunk;

    while (!should_stop_.load(std::memory_order_relaxed)) {
        const auto loop_start = steady_clock::now();

        // 1. Dequeue Primary Audio Chunk from Lock-Free Ring Buffer
        const auto t_cap_start = steady_clock::now();
        if (!alsa_backend_.read_primary_chunk(primary_chunk)) {
            // No audio ready yet in capture buffer: yield thread slice
            std::this_thread::yield();
            continue;
        }
        const auto t_cap_end = steady_clock::now();
        const double cap_us = duration<double, std::micro>(t_cap_end - t_cap_start).count();

        // 2. Dequeue Reference (Error Mic) Audio Chunk
        const bool has_ref = !config_.audio.single_mic && alsa_backend_.read_reference_chunk(reference_chunk);

        const size_t count = std::min(primary_chunk.count, hop_size);
        if (count == 0) continue;

        // 3. Check Clock Drift between Dual Microphones
        auto& drift_detector = alsa_backend_.get_drift_detector();
        const double drift_ms = drift_detector.get_drift_ms();
        const bool drift_warning = drift_detector.is_warning_active();
        const bool nlms_available = config_.audio.nlms_enabled && has_ref && !drift_warning && (reference_chunk.count >= count);

        // 4. Preprocessing: DC Removal + 80Hz Biquad High-Pass on Primary
        const auto t_prep_start = steady_clock::now();
        for (size_t i = 0; i < count; ++i) {
            float dc_clean = dc_blocker_.process(primary_chunk.samples[i]);
            primary_preproc[i] = high_pass_.process(dc_clean);
        }
        const auto t_prep_end = steady_clock::now();
        const double prep_us = duration<double, std::micro>(t_prep_end - t_prep_start).count();

        // 5. STFT Spectral Analysis on Primary
        const auto t_stft_start = steady_clock::now();
        stft_primary_.process_analysis(std::span<const float>(primary_preproc.data(), count), spectral_frame);
        const auto t_stft_end = steady_clock::now();
        const double stft_us = duration<double, std::micro>(t_stft_end - t_stft_start).count();

        // 6. Parallel Branch A: Impulse Detection & Voice Activity Detection
        const auto impulse_state = impulse_detector_.detect(
            std::span<const float>(primary_preproc.data(), count),
            std::span<const float>(spectral_frame.magnitude.data(), spectral_frame.magnitude.size()));
        const auto vad_decision = vad_.process_frame(
            std::span<const float>(primary_preproc.data(), count),
            std::span<const float>(spectral_frame.magnitude.data(), spectral_frame.magnitude.size()));

        // 7. Parallel Branch B: ONNX Runtime AI Speech Enhancement & iSTFT Synthesis
        const auto t_ai_start = steady_clock::now();
        bool ai_success = false;
        if (speech_enhancer_.is_initialized() && !speech_enhancer_.is_degraded()) {
            std::copy(spectral_frame.real.begin(), spectral_frame.real.end(), complex_in.real.begin());
            std::copy(spectral_frame.imag.begin(), spectral_frame.imag.end(), complex_in.imag.begin());
            ai_success = speech_enhancer_.infer(complex_in, complex_out);
        }
        const auto t_ai_end = steady_clock::now();
        const double ai_us = duration<double, std::micro>(t_ai_end - t_ai_start).count();

        // Synthesize AI enhanced time-domain audio via iSTFT
        const auto t_istft_start = steady_clock::now();
        if (ai_success) {
            stft_primary_.process_synthesis_complex(
                std::span<const float>(complex_out.real.data(), complex_out.num_bins),
                std::span<const float>(complex_out.imag.data(), complex_out.num_bins),
                std::span<float>(ai_hop_out.data(), count)
            );
        } else {
            // Pass-through primary if AI degraded
            std::copy_n(primary_preproc.data(), count, ai_hop_out.data());
        }
        const auto t_istft_end = steady_clock::now();
        const double istft_us = duration<double, std::micro>(t_istft_end - t_istft_start).count();

        // 8. Parallel Branch C: NLMS Adaptive Noise Cancellation on Dual Mics
        const auto t_nlms_start = steady_clock::now();
        if (nlms_available) {
            nlms_.process_block(
                primary_preproc.data(),
                reference_chunk.samples,
                nlms_out.data(),
                count
            );
        } else {
            std::copy_n(primary_preproc.data(), count, nlms_out.data());
        }
        const auto t_nlms_end = steady_clock::now();
        const double nlms_us = duration<double, std::micro>(t_nlms_end - t_nlms_start).count();

        // 9. Multi-Mode Fusion Controller (Phase 9)
        const auto t_fuse_start = steady_clock::now();
        noiselessx::fusion::FusionInput fusion_inp{
            .ai_output = noiselessx::fusion::AudioFrame(ai_hop_out.data(), count),
            .nlms_output = noiselessx::fusion::AudioFrame(nlms_out.data(), count),
            .ai_confidence = ai_success ? 0.92f : 0.0f,
            .impulse_probability = impulse_state.probability,
            .vad_probability = vad_decision.speech_probability,
            .nlms_available = nlms_available,
            .raw_input = noiselessx::fusion::AudioFrame(primary_chunk.samples, count),
            .ai_available = ai_success,
            .hardware_available = true,
            .bypass_requested = fusion_controller_.is_bypass()
        };

        noiselessx::fusion::AudioFrame fused_frame = fusion_controller_.fuse(fusion_inp);
        const auto t_fuse_end = steady_clock::now();
        const double fuse_us = duration<double, std::micro>(t_fuse_end - t_fuse_start).count();

        // 10. Push Fused Output to ALSA Playback Ring Buffer
        const auto t_play_start = steady_clock::now();
        playback_chunk.count = count;
        playback_chunk.frame_index = primary_chunk.frame_index;
        playback_chunk.timestamp_ns = primary_chunk.timestamp_ns;
        for (size_t i = 0; i < count; ++i) {
            playback_chunk.samples[i] = fused_frame[i];
        }
        const bool pb_pushed = alsa_backend_.write_playback_chunk(playback_chunk);
        const auto t_play_end = steady_clock::now();
        const double play_us = duration<double, std::micro>(t_play_end - t_play_start).count();

        const auto loop_end = steady_clock::now();
        const double total_proc_us = duration<double, std::micro>(loop_end - loop_start).count();
        const double hop_duration_us = (static_cast<double>(count) / static_cast<double>(config_.audio.sample_rate)) * 1e6;
        const double rtf = (hop_duration_us > 0.0) ? (total_proc_us / hop_duration_us) : 0.0;

        // 11. Record Real Telemetry Metrics
        telemetry_.capture_us = cap_us;
        telemetry_.preprocessing_us = prep_us;
        telemetry_.stft_us = stft_us;
        telemetry_.ai_inference_us = ai_us;
        telemetry_.nlms_us = nlms_us;
        telemetry_.fusion_us = fuse_us;
        telemetry_.istft_us = istft_us;
        telemetry_.playback_queue_us = play_us;
        telemetry_.total_processing_us = total_proc_us;
        telemetry_.rtf = rtf;
        telemetry_.end_to_end_latency_ms = (total_proc_us / 1000.0) + (static_cast<double>(config_.audio.buffer_size) / static_cast<double>(config_.audio.sample_rate) * 1000.0);

        telemetry_.processed_frames++;
        if (!pb_pushed) {
            telemetry_.dropped_frames++;
        }

        auto alsa_status = alsa_backend_.get_status();
        telemetry_.alsa_xruns_primary = alsa_status.primary_xrun_count;
        telemetry_.alsa_xruns_reference = alsa_status.reference_xrun_count;
        telemetry_.alsa_xruns_playback = alsa_status.output_xrun_count;

        telemetry_.drift_ms = drift_ms;
        telemetry_.drift_samples = drift_detector.get_drift_samples();
        telemetry_.drift_warning = drift_warning;

        telemetry_.fusion_mode = fusion_controller_.get_current_mode();
        telemetry_.current_lambda = fusion_controller_.get_current_lambda();
        telemetry_.impulse_envelope_gain = fusion_controller_.get_current_gain_envelope();
        telemetry_.ai_confidence = fusion_inp.ai_confidence;
        telemetry_.impulse_probability = fusion_inp.impulse_probability;
        telemetry_.vad_probability = fusion_inp.vad_probability;
    }
}

PipelineTelemetry RealtimePipeline::get_telemetry() {
    // Periodically refresh CPU and thermal telemetry (every 250ms)
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_system_sample_time_).count() >= 250) {
        system_metrics_.sample();
        last_system_sample_time_ = now;
    }

    telemetry_.cpu_per_core = system_metrics_.get_per_core_cpu_percent();
    telemetry_.overall_cpu_pct = system_metrics_.get_overall_cpu_percent();
    telemetry_.cpu_temperature_c = system_metrics_.get_cpu_temperature_c();
    telemetry_.temp_available = system_metrics_.is_temperature_available();

    return telemetry_;
}

} // namespace runtime
} // namespace noiselessx
