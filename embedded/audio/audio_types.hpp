#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <chrono>

namespace noiselessx {
namespace audio {

enum class DeviceTopology {
    UNKNOWN = 0,
    SINGLE_DEVICE_MULTICHANNEL,
    TWO_SEPARATE_DEVICES
};

struct AudioDeviceInfo {
    int card_index{-1};
    int device_index{-1};
    std::string card_name;
    std::string device_name;
    std::string alsa_identifier; // e.g. "hw:0,0" or "hw:CARD=Headset,DEV=0"
    bool is_capture{false};
    bool is_playback{false};
    std::vector<uint32_t> supported_rates;
    std::vector<uint32_t> supported_channels;
};

struct AudioConfig {
    uint32_t sample_rate{16000};
    uint32_t channels{1}; // per-stream channel count (typically 1 for mono speech)
    uint32_t frame_ms{10};
    uint32_t hop_ms{5};
    uint32_t period_size{160}; // 10ms at 16kHz
    uint32_t buffer_size{640}; // 4 periods
    std::string primary_device{"hw:CARD=Headset,DEV=0"};
    std::string reference_device{"hw:CARD=ErrorMic,DEV=0"};
    std::string output_device{"hw:CARD=Headset,DEV=0"};
    double max_allowable_drift_ms{10.0};
};

struct AudioDeviceStatus {
    bool is_running{false};
    DeviceTopology topology{DeviceTopology::UNKNOWN};
    std::string primary_device_name;
    std::string reference_device_name;
    std::string output_device_name;
    uint32_t negotiated_sample_rate{0};
    size_t negotiated_period_size{0};
    size_t negotiated_buffer_size{0};
    uint64_t primary_xrun_count{0};
    uint64_t reference_xrun_count{0};
    uint64_t output_xrun_count{0};
    uint64_t primary_frames_captured{0};
    uint64_t reference_frames_captured{0};
    uint64_t output_frames_written{0};
    double inter_stream_drift_ms{0.0};
    int64_t inter_stream_drift_samples{0};
    bool drift_warning_flag{false};
};

// Fixed-size sample buffer for real-time audio transfer (default max 1024 samples = 64ms @ 16kHz)
template <size_t MaxSamples = 1024>
struct AudioChunk {
    float samples[MaxSamples];
    size_t count{0};
    uint64_t frame_index{0};
    uint64_t timestamp_ns{0};
};

} // namespace audio
} // namespace noiselessx
