#pragma once

#include "audio_types.hpp"
#include "ring_buffer.hpp"
#include "drift_detector.hpp"
#include <memory>
#include <thread>
#include <atomic>
#include <string>

#if NOICELESSX_HAS_ALSA
#include <alsa/asoundlib.h>
#endif

namespace noiselessx::audio {

/**
 * @brief Real ALSA Audio Subsystem for Dual-Microphone Raspberry Pi 4/5 Operation.
 * 
 * Manages:
 * - Primary Mic (Headphone mic, speech + ambient)
 * - Reference Mic (Error mic, ambient noise / residual)
 * - Output Playback Device (Headphone / DAC)
 * - Automatic hardware topology detection (Two separate devices vs Single multichannel device)
 * - Lock-free SPSC ring buffers for zero-heap real-time audio pipeline
 * - Drift detection and XRUN recovery per stream
 * - Strict zero-mock enforcement: Aborts immediately if required hardware is missing
 */
class AlsaBackend {
public:
    AlsaBackend();
    ~AlsaBackend();

    // Non-copyable, non-movable
    AlsaBackend(const AlsaBackend&) = delete;
    AlsaBackend& operator=(const AlsaBackend&) = delete;
    AlsaBackend(AlsaBackend&&) = delete;
    AlsaBackend& operator=(AlsaBackend&&) = delete;

    /**
     * @brief Initialize ALSA devices according to config.
     * @return true on success; prints descriptive error and returns false if any device is unavailable.
     */
    bool initialize(const AudioConfig& config);

    /**
     * @brief Start real-time audio capture and playback threads.
     */
    bool start();

    /**
     * @brief Stop real-time audio threads and drain ALSA buffers.
     */
    void stop();

    /**
     * @brief Pop a captured audio chunk from the Primary microphone ring buffer.
     */
    bool read_primary_chunk(AudioChunk<1024>& chunk);

    /**
     * @brief Pop a captured audio chunk from the Reference microphone ring buffer.
     */
    bool read_reference_chunk(AudioChunk<1024>& chunk);

    /**
     * @brief Push an audio chunk to the Playback ring buffer for headphone output.
     */
    bool write_playback_chunk(const AudioChunk<1024>& chunk);

    /**
     * @brief Get instantaneous audio hardware and stream telemetry.
     */
    AudioDeviceStatus get_status() const;

    /**
     * @brief Access the active audio configuration.
     */
    const AudioConfig& get_config() const noexcept { return config_; }

    /**
     * @brief Access the inter-stream clock drift detector.
     */
    DriftDetector& get_drift_detector() noexcept { return drift_detector_; }

private:
#if NOICELESSX_HAS_ALSA
    bool configure_pcm_handle(
        snd_pcm_t** pcm,
        const std::string& dev_name,
        snd_pcm_stream_t stream_type,
        uint32_t channels,
        uint32_t& actual_rate,
        snd_pcm_uframes_t& actual_period,
        snd_pcm_uframes_t& actual_buffer,
        const std::string& role_label
    );

    void capture_thread_loop(snd_pcm_t* pcm, SpscRingBuffer<AudioChunk<1024>, 64>& ring_buf, bool is_primary);
    void unified_multichannel_capture_thread_loop(snd_pcm_t* pcm);
    void playback_thread_loop(snd_pcm_t* pcm);

    snd_pcm_t* pcm_primary_{nullptr};
    snd_pcm_t* pcm_reference_{nullptr};
    snd_pcm_t* pcm_playback_{nullptr};
#endif

    AudioConfig config_;
    mutable AudioDeviceStatus status_;
    DriftDetector drift_detector_;

    // Lock-free bounded ring buffers for audio streams (zero allocation in audio thread)
    SpscRingBuffer<AudioChunk<1024>, 64> primary_ring_buffer_;
    SpscRingBuffer<AudioChunk<1024>, 64> reference_ring_buffer_;
    SpscRingBuffer<AudioChunk<1024>, 64> playback_ring_buffer_;

    std::atomic<bool> is_running_{false};
    std::atomic<bool> should_stop_{false};

    std::thread primary_capture_thread_;
    std::thread reference_capture_thread_;
    std::thread unified_capture_thread_;
    std::thread playback_thread_;
};

} // namespace noiselessx::audio
