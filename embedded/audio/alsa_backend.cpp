#include "alsa_backend.hpp"
#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>

namespace noiselessx {
namespace audio {

AlsaBackend::AlsaBackend() = default;

AlsaBackend::~AlsaBackend() {
    stop();
}

bool AlsaBackend::initialize(const AudioConfig& config) {
    config_ = config;
    drift_detector_ = DriftDetector(config.sample_rate, config.max_allowable_drift_ms);
    status_ = AudioDeviceStatus{};
    status_.primary_device_name = config.primary_device;
    status_.reference_device_name = config.reference_device;
    status_.output_device_name = config.output_device;

#if !NOICELESSX_HAS_ALSA
    std::cerr << "[AlsaBackend] Error: ALSA subsystem is disabled or unavailable on this host.\n"
              << "Real hardware operation requires Linux with ALSA (Raspberry Pi 4/5).\n";
    return false;
#else

    // 1. Detect device topology
    if (config.single_mic || config.reference_device.empty() || config.reference_device == "disabled" || config.reference_device == "null") {
        status_.topology = DeviceTopology::SINGLE_MIC;
        std::cout << "[AlsaBackend] Topology: SINGLE_MIC detected (left channel only).\n"
                  << "  Primary: " << config.primary_device << "\n";
    } else if (config.primary_device == config.reference_device && !config.primary_device.empty()) {
        status_.topology = DeviceTopology::SINGLE_DEVICE_MULTICHANNEL;
        std::cout << "[AlsaBackend] Topology: SINGLE_DEVICE_MULTICHANNEL detected ("
                  << config.primary_device << "). Ch0 = Primary, Ch1 = Reference.\n";
    } else {
        status_.topology = DeviceTopology::TWO_SEPARATE_DEVICES;
        std::cout << "[AlsaBackend] Topology: TWO_SEPARATE_DEVICES detected.\n"
                  << "  Primary:   " << config.primary_device << "\n"
                  << "  Reference: " << config.reference_device << "\n";
    }

    uint32_t negotiated_rate = config.hardware_sample_rate;
    snd_pcm_uframes_t negotiated_period = config.period_size;
    snd_pcm_uframes_t negotiated_buffer = config.buffer_size;
    if (config.hardware_sample_rate != config.sample_rate) {
        const uint32_t rate_ratio = config.hardware_sample_rate / config.sample_rate;
        if (rate_ratio == 0 || config.hardware_sample_rate % config.sample_rate != 0) {
            std::cerr << "[AlsaBackend] Unsupported non-integer hardware/internal sample-rate ratio.\n";
            return false;
        }
        negotiated_period *= rate_ratio;
        negotiated_buffer *= rate_ratio;
    }

    // 2. Open and configure Capture Device(s)
    if (status_.topology == DeviceTopology::SINGLE_MIC) {
        // VoiceHAT capture is hardware-rate, interleaved S32_LE stereo. Channel 0 is retained.
        if (!configure_pcm_handle(&pcm_primary_, config.primary_device, SND_PCM_STREAM_CAPTURE,
                                 config.hardware_channels, negotiated_rate, negotiated_period, negotiated_buffer,
                                 "Single microphone capture", SND_PCM_FORMAT_S32_LE)) {
            return false;
        }
    } else if (status_.topology == DeviceTopology::SINGLE_DEVICE_MULTICHANNEL) {
        // Open combined capture device with 2 channels
        if (!configure_pcm_handle(&pcm_primary_, config.primary_device, SND_PCM_STREAM_CAPTURE,
                                 2, negotiated_rate, negotiated_period, negotiated_buffer,
                                 "Combined Primary & Reference microphone")) {
            return false;
        }
    } else {
        // Open separate Primary (headphone) microphone
        if (!configure_pcm_handle(&pcm_primary_, config.primary_device, SND_PCM_STREAM_CAPTURE,
                                 1, negotiated_rate, negotiated_period, negotiated_buffer,
                                 "Primary (headphone) microphone")) {
            return false;
        }

        // Open separate Reference (error) microphone
        snd_pcm_uframes_t ref_period = config.period_size;
        snd_pcm_uframes_t ref_buffer = config.buffer_size;
        uint32_t ref_rate = config.sample_rate;
        if (!configure_pcm_handle(&pcm_reference_, config.reference_device, SND_PCM_STREAM_CAPTURE,
                                 1, ref_rate, ref_period, ref_buffer,
                                 "Reference (error) microphone")) {
            // Clean up primary handle on failure
            if (pcm_primary_) {
                snd_pcm_close(pcm_primary_);
                pcm_primary_ = nullptr;
            }
            return false;
        }
    }

    // 3. Open and configure Playback Device (headphone / USB DAC)
    snd_pcm_uframes_t out_period = config.period_size;
    snd_pcm_uframes_t out_buffer = config.buffer_size;
    uint32_t out_rate = config.sample_rate;
    if (!configure_pcm_handle(&pcm_playback_, config.output_device, SND_PCM_STREAM_PLAYBACK,
                             1, out_rate, out_period, out_buffer,
                             "Playback (headphone/DAC) device")) {
        if (pcm_primary_) { snd_pcm_close(pcm_primary_); pcm_primary_ = nullptr; }
        if (pcm_reference_) { snd_pcm_close(pcm_reference_); pcm_reference_ = nullptr; }
        return false;
    }

    status_.negotiated_sample_rate = negotiated_rate;
    status_.negotiated_period_size = negotiated_period;
    status_.negotiated_buffer_size = negotiated_buffer;

    std::cout << "[AlsaBackend] Successfully negotiated hardware parameters:\n"
              << "  Sample Rate: " << negotiated_rate << " Hz\n"
              << "  Period Size: " << negotiated_period << " frames ("
              << (static_cast<double>(negotiated_period) / negotiated_rate * 1000.0) << " ms)\n"
              << "  Buffer Size: " << negotiated_buffer << " frames\n";

    return true;
#endif
}

#if NOICELESSX_HAS_ALSA
bool AlsaBackend::configure_pcm_handle(
    snd_pcm_t** pcm,
    const std::string& dev_name,
    snd_pcm_stream_t stream_type,
    uint32_t channels,
    uint32_t& actual_rate,
    snd_pcm_uframes_t& actual_period,
    snd_pcm_uframes_t& actual_buffer,
    const std::string& role_label,
    snd_pcm_format_t format
) {
    int err = snd_pcm_open(pcm, dev_name.c_str(), stream_type, 0);
    if (err < 0) {
        std::cerr << role_label << " unavailable: " << snd_strerror(err)
                  << " (device: " << dev_name << ")\n";
        return false;
    }

    snd_pcm_hw_params_t* hw_params = nullptr;
    snd_pcm_hw_params_alloca(&hw_params);

    err = snd_pcm_hw_params_any(*pcm, hw_params);
    if (err < 0) {
        std::cerr << role_label << " error initializing hw params: " << snd_strerror(err) << "\n";
        snd_pcm_close(*pcm);
        *pcm = nullptr;
        return false;
    }

    err = snd_pcm_hw_params_set_access(*pcm, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        std::cerr << role_label << " interleaved access unsupported: " << snd_strerror(err) << "\n";
        snd_pcm_close(*pcm);
        *pcm = nullptr;
        return false;
    }

    err = snd_pcm_hw_params_set_format(*pcm, hw_params, format);
    if (err < 0) {
        std::cerr << role_label << " requested PCM format unsupported: " << snd_strerror(err) << "\n";
        snd_pcm_close(*pcm);
        *pcm = nullptr;
        return false;
    }

    err = snd_pcm_hw_params_set_channels(*pcm, hw_params, channels);
    if (err < 0) {
        std::cerr << role_label << " channels (" << channels << ") unsupported: " << snd_strerror(err) << "\n";
        snd_pcm_close(*pcm);
        *pcm = nullptr;
        return false;
    }

    unsigned int req_rate = actual_rate;
    err = snd_pcm_hw_params_set_rate_near(*pcm, hw_params, &req_rate, 0);
    if (err < 0) {
        std::cerr << role_label << " rate " << actual_rate << " unsupported: " << snd_strerror(err) << "\n";
        snd_pcm_close(*pcm);
        *pcm = nullptr;
        return false;
    }
    actual_rate = req_rate;

    snd_pcm_uframes_t req_period = actual_period;
    err = snd_pcm_hw_params_set_period_size_near(*pcm, hw_params, &req_period, 0);
    if (err < 0) {
        std::cerr << role_label << " period size unsupported: " << snd_strerror(err) << "\n";
        snd_pcm_close(*pcm);
        *pcm = nullptr;
        return false;
    }
    actual_period = req_period;

    snd_pcm_uframes_t req_buffer = actual_buffer;
    err = snd_pcm_hw_params_set_buffer_size_near(*pcm, hw_params, &req_buffer);
    if (err < 0) {
        std::cerr << role_label << " buffer size unsupported: " << snd_strerror(err) << "\n";
        snd_pcm_close(*pcm);
        *pcm = nullptr;
        return false;
    }
    actual_buffer = req_buffer;

    err = snd_pcm_hw_params(*pcm, hw_params);
    if (err < 0) {
        std::cerr << role_label << " failed to set hw params: " << snd_strerror(err) << "\n";
        snd_pcm_close(*pcm);
        *pcm = nullptr;
        return false;
    }

    err = snd_pcm_prepare(*pcm);
    if (err < 0) {
        std::cerr << role_label << " failed to prepare PCM stream: " << snd_strerror(err) << "\n";
        snd_pcm_close(*pcm);
        *pcm = nullptr;
        return false;
    }

    return true;
}
#endif

bool AlsaBackend::start() {
    if (is_running_.load()) return true;

#if !NOICELESSX_HAS_ALSA
    std::cerr << "[AlsaBackend] Cannot start: ALSA is not enabled.\n";
    return false;
#else
    if (!pcm_primary_ || !pcm_playback_) {
        if (!initialize(config_)) {
            return false;
        }
    }

    if (!pcm_primary_ || !pcm_playback_ ||
        (status_.topology == DeviceTopology::TWO_SEPARATE_DEVICES && !pcm_reference_)) {
        std::cerr << "[AlsaBackend] Cannot start: Devices not properly initialized.\n";
        return false;
    }

    should_stop_.store(false);
    is_running_.store(true);
    status_.is_running = true;

    // Launch realtime capture threads
    primary_decimator_.reset();
    if (status_.topology == DeviceTopology::SINGLE_MIC) {
        unified_capture_thread_ = std::thread(&AlsaBackend::single_microphone_capture_thread_loop, this, pcm_primary_);
    } else if (status_.topology == DeviceTopology::SINGLE_DEVICE_MULTICHANNEL) {
        unified_capture_thread_ = std::thread(&AlsaBackend::unified_multichannel_capture_thread_loop, this, pcm_primary_);
    } else {
        primary_capture_thread_ = std::thread(&AlsaBackend::capture_thread_loop, this, pcm_primary_, std::ref(primary_ring_buffer_), true);
        reference_capture_thread_ = std::thread(&AlsaBackend::capture_thread_loop, this, pcm_reference_, std::ref(reference_ring_buffer_), false);
    }

    // Launch realtime playback thread
    playback_thread_ = std::thread(&AlsaBackend::playback_thread_loop, this, pcm_playback_);

    return true;
#endif
}

void AlsaBackend::stop() {
    if (!is_running_.load()) return;

    should_stop_.store(true);

#if NOICELESSX_HAS_ALSA
    // Interrupt blocking read/write calls before joining their worker threads.
    if (pcm_primary_) snd_pcm_drop(pcm_primary_);
    if (pcm_reference_) snd_pcm_drop(pcm_reference_);
    if (pcm_playback_) snd_pcm_drop(pcm_playback_);
#endif

    if (primary_capture_thread_.joinable()) primary_capture_thread_.join();
    if (reference_capture_thread_.joinable()) reference_capture_thread_.join();
    if (unified_capture_thread_.joinable()) unified_capture_thread_.join();
    if (playback_thread_.joinable()) playback_thread_.join();

#if NOICELESSX_HAS_ALSA
    if (pcm_primary_) {
        snd_pcm_drop(pcm_primary_);
        snd_pcm_close(pcm_primary_);
        pcm_primary_ = nullptr;
    }
    if (pcm_reference_) {
        snd_pcm_drop(pcm_reference_);
        snd_pcm_close(pcm_reference_);
        pcm_reference_ = nullptr;
    }
    if (pcm_playback_) {
        snd_pcm_drain(pcm_playback_);
        snd_pcm_close(pcm_playback_);
        pcm_playback_ = nullptr;
    }
#endif

    is_running_.store(false);
    status_.is_running = false;
}

#if NOICELESSX_HAS_ALSA
void AlsaBackend::capture_thread_loop(snd_pcm_t* pcm, SpscRingBuffer<AudioChunk<1024>, 64>& ring_buf, bool is_primary) {
    const size_t period_frames = status_.negotiated_period_size > 0 ? status_.negotiated_period_size : 160;
    int16_t raw_buffer[1024];
    uint64_t frame_index = 0;

    snd_pcm_start(pcm);

    while (!should_stop_.load(std::memory_order_relaxed)) {
        snd_pcm_sframes_t frames = snd_pcm_readi(pcm, raw_buffer, period_frames);

        if (frames < 0) {
            // Overrun (XRUN) recovery
            int err = snd_pcm_recover(pcm, static_cast<int>(frames), 0);
            if (is_primary) {
                status_.primary_xrun_count++;
            } else {
                status_.reference_xrun_count++;
            }
            if (err < 0) {
                std::cerr << "[AlsaBackend] Unrecoverable XRUN on "
                          << (is_primary ? "Primary" : "Reference") << ": " << snd_strerror(err) << "\n";
            }
            continue;
        }

        if (frames > 0) {
            AudioChunk<1024> chunk;
            chunk.count = static_cast<size_t>(frames);
            chunk.frame_index = frame_index++;
            chunk.timestamp_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count()
            );

            // Convert S16_LE to float normalized to [-1.0f, 1.0f]
            for (size_t i = 0; i < chunk.count; ++i) {
                chunk.samples[i] = static_cast<float>(raw_buffer[i]) / 32768.0f;
            }

            ring_buf.push(chunk);

            if (is_primary) {
                status_.primary_frames_captured += static_cast<uint64_t>(frames);
                drift_detector_.record_primary(static_cast<uint64_t>(frames), chunk.timestamp_ns);
            } else {
                status_.reference_frames_captured += static_cast<uint64_t>(frames);
                drift_detector_.record_reference(static_cast<uint64_t>(frames), chunk.timestamp_ns);
            }
        }
    }
}

void AlsaBackend::unified_multichannel_capture_thread_loop(snd_pcm_t* pcm) {
    const size_t period_frames = status_.negotiated_period_size > 0 ? status_.negotiated_period_size : 160;
    int16_t interleaved_buffer[2048]; // 2 channels
    uint64_t frame_index = 0;

    snd_pcm_start(pcm);

    while (!should_stop_.load(std::memory_order_relaxed)) {
        snd_pcm_sframes_t frames = snd_pcm_readi(pcm, interleaved_buffer, period_frames);

        if (frames < 0) {
            int err = snd_pcm_recover(pcm, static_cast<int>(frames), 0);
            status_.primary_xrun_count++;
            status_.reference_xrun_count++;
            if (err < 0) {
                std::cerr << "[AlsaBackend] Unrecoverable XRUN on unified capture: " << snd_strerror(err) << "\n";
            }
            continue;
        }

        if (frames > 0) {
            uint64_t now_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count()
            );

            AudioChunk<1024> prim_chunk;
            AudioChunk<1024> ref_chunk;
            prim_chunk.count = static_cast<size_t>(frames);
            prim_chunk.frame_index = frame_index;
            prim_chunk.timestamp_ns = now_ns;

            ref_chunk.count = static_cast<size_t>(frames);
            ref_chunk.frame_index = frame_index++;
            ref_chunk.timestamp_ns = now_ns;

            for (size_t i = 0; i < static_cast<size_t>(frames); ++i) {
                prim_chunk.samples[i] = static_cast<float>(interleaved_buffer[2 * i]) / 32768.0f;
                ref_chunk.samples[i] = static_cast<float>(interleaved_buffer[2 * i + 1]) / 32768.0f;
            }

            primary_ring_buffer_.push(prim_chunk);
            reference_ring_buffer_.push(ref_chunk);

            status_.primary_frames_captured += static_cast<uint64_t>(frames);
            status_.reference_frames_captured += static_cast<uint64_t>(frames);
            drift_detector_.record_primary(static_cast<uint64_t>(frames), now_ns);
            drift_detector_.record_reference(static_cast<uint64_t>(frames), now_ns);
        }
    }
}

void AlsaBackend::single_microphone_capture_thread_loop(snd_pcm_t* pcm) {
    const size_t period_frames = status_.negotiated_period_size > 0 ? status_.negotiated_period_size : 480;
    int32_t interleaved_buffer[2048];
    float left_channel[1024];
    float resampled[1024];
    uint64_t frame_index = 0;

    if (snd_pcm_start(pcm) < 0) {
        std::cerr << "[AlsaBackend] Failed to start single-microphone capture.\n";
        return;
    }

    while (!should_stop_.load(std::memory_order_relaxed)) {
        snd_pcm_sframes_t frames = snd_pcm_readi(pcm, interleaved_buffer, period_frames);
        if (frames < 0) {
            const int err = snd_pcm_recover(pcm, static_cast<int>(frames), 0);
            status_.primary_xrun_count++;
            if (err < 0) {
                std::cerr << "[AlsaBackend] Unrecoverable single-mic XRUN: " << snd_strerror(err) << "\n";
            }
            continue;
        }
        if (frames == 0) continue;

        const size_t input_frames = static_cast<size_t>(frames);
        for (size_t i = 0; i < input_frames; ++i) {
            left_channel[i] = static_cast<float>(interleaved_buffer[2 * i]) / 2147483648.0f;
        }
        const size_t output_frames = primary_decimator_.process(
            left_channel, input_frames, resampled, sizeof(resampled) / sizeof(resampled[0]));
        if (output_frames == 0) continue;

        AudioChunk<1024> chunk;
        chunk.count = output_frames;
        chunk.frame_index = frame_index++;
        chunk.timestamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        std::memcpy(chunk.samples, resampled, output_frames * sizeof(float));
        primary_ring_buffer_.push(chunk);
        status_.primary_frames_captured += output_frames;
    }
}

void AlsaBackend::playback_thread_loop(snd_pcm_t* pcm) {
    AudioChunk<1024> chunk;
    int16_t out_buffer[1024];

    while (!should_stop_.load(std::memory_order_relaxed)) {
        if (playback_ring_buffer_.pop(chunk)) {
            for (size_t i = 0; i < chunk.count; ++i) {
                float clamped = std::clamp(chunk.samples[i], -1.0f, 1.0f);
                out_buffer[i] = static_cast<int16_t>(clamped * 32767.0f);
            }

            size_t offset = 0;
            while (offset < chunk.count && !should_stop_.load(std::memory_order_relaxed)) {
                snd_pcm_sframes_t written = snd_pcm_writei(pcm, out_buffer + offset, chunk.count - offset);
                if (written < 0) {
                    int err = snd_pcm_recover(pcm, static_cast<int>(written), 0);
                    status_.output_xrun_count++;
                    if (err < 0) {
                        std::cerr << "[AlsaBackend] Playback XRUN recovery error: " << snd_strerror(err) << "\n";
                        break;
                    }
                } else if (written > 0) {
                    offset += static_cast<size_t>(written);
                    status_.output_frames_written += static_cast<uint64_t>(written);
                }
            }
        } else {
            // Buffer underrun / waiting for DSP frames
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }
}
#endif

bool AlsaBackend::read_primary_chunk(AudioChunk<1024>& chunk) {
    return primary_ring_buffer_.pop(chunk);
}

bool AlsaBackend::read_reference_chunk(AudioChunk<1024>& chunk) {
    return reference_ring_buffer_.pop(chunk);
}

bool AlsaBackend::write_playback_chunk(const AudioChunk<1024>& chunk) {
    return playback_ring_buffer_.push(chunk);
}

AudioDeviceStatus AlsaBackend::get_status() const {
    status_.inter_stream_drift_ms = drift_detector_.get_drift_ms();
    status_.inter_stream_drift_samples = drift_detector_.get_drift_samples();
    status_.drift_warning_flag = drift_detector_.is_warning_active();
    return status_;
}

} // namespace audio
} // namespace noiselessx
