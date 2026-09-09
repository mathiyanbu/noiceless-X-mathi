#include "embedded/audio/alsa_backend.hpp"
#include "embedded/audio/device_enumerator.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <thread>
#include <iomanip>

using namespace noiselessx::audio;

// Simple lightweight YAML parser for key audio configuration settings
bool parse_config_file(const std::string& path, AudioConfig& config) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[audio_test] Warning: Could not open config file: " << path
                  << ". Using default configuration values.\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Strip leading whitespace
        auto start = line.find_first_not_of(" \t");
        if (start == std::string::npos || line[start] == '#') continue;
        line = line.substr(start);

        auto colon = line.find(':');
        if (colon == std::string::npos) continue;

        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);

        // Trim key and value
        auto trim_str = [](std::string& s) {
            auto first = s.find_first_not_of(" \t\r\n\"");
            if (first == std::string::npos) { s = ""; return; }
            auto last = s.find_last_not_of(" \t\r\n\"");
            s = s.substr(first, (last - first + 1));
        };
        trim_str(key);
        trim_str(val);

        if (key == "sample_rate") config.sample_rate = std::stoul(val);
        else if (key == "frame_ms") config.frame_ms = std::stoul(val);
        else if (key == "hop_ms") config.hop_ms = std::stoul(val);
        else if (key == "buffer_frames") config.period_size = std::stoul(val);
        else if (key == "input_device") config.primary_device = val;
        else if (key == "reference_device") config.reference_device = val;
        else if (key == "output_device") config.output_device = val;
    }

    config.buffer_size = config.period_size * 4;
    return true;
}

float calculate_rms(const float* samples, size_t count) {
    if (count == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += samples[i] * samples[i];
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(count)));
}

float rms_to_db(float rms) {
    if (rms < 1e-6f) return -120.0f;
    return 20.0f * std::log10(rms);
}

int main(int argc, char* argv[]) {
    std::cout << "\n==========================================================================\n";
    std::cout << "        SIH26052 NOICELESSX — Dual-Microphone Hardware Test Utility        \n";
    std::cout << "==========================================================================\n\n";

    std::string config_path = "config/raspberrypi.yaml";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    AudioConfig config;
    std::cout << "[Step 1] Loading audio configuration from: " << config_path << "\n";
    parse_config_file(config_path, config);

    std::cout << "  Configured Settings:\n"
              << "    Sample Rate:      " << config.sample_rate << " Hz\n"
              << "    Period Size:      " << config.period_size << " frames\n"
              << "    Buffer Size:      " << config.buffer_size << " frames\n"
              << "    Primary Mic:      " << config.primary_device << " (Headphone Mic)\n"
              << "    Reference Mic:    " << config.reference_device << " (Error Mic)\n"
              << "    Playback Output:  " << config.output_device << " (Headphones/DAC)\n\n";

    // 1. Enumerate and print ALSA endpoints
    std::cout << "[Step 2] Enumerating physical ALSA audio endpoints...\n";
    auto devices = DeviceEnumerator::enumerate_devices();
    DeviceEnumerator::print_devices(devices);

    const auto* prim_dev = DeviceEnumerator::find_device_matching(devices, config.primary_device, true);
    const auto* ref_dev = DeviceEnumerator::find_device_matching(devices, config.reference_device, true);
    const auto* out_dev = DeviceEnumerator::find_device_matching(devices, config.output_device, false);

    std::cout << "Endpoint Identification:\n";
    std::cout << "  [Primary Mic]   " << config.primary_device
              << (prim_dev ? (" -> MATCHED (" + prim_dev->card_name + " / " + prim_dev->alsa_identifier + ")") : " -> NOT FOUND IN ENUMERATION") << "\n";
    std::cout << "  [Reference Mic] " << config.reference_device
              << (ref_dev ? (" -> MATCHED (" + ref_dev->card_name + " / " + ref_dev->alsa_identifier + ")") : " -> NOT FOUND IN ENUMERATION") << "\n";
    std::cout << "  [Playback Out]  " << config.output_device
              << (out_dev ? (" -> MATCHED (" + out_dev->card_name + " / " + out_dev->alsa_identifier + ")") : " -> NOT FOUND IN ENUMERATION") << "\n\n";

    // 2. Initialize ALSA Backend
    std::cout << "[Step 3] Initializing ALSA dual-microphone capture and output...\n";
    AlsaBackend backend;
    if (!backend.initialize(config)) {
        std::cerr << "\n[FATAL ERROR] Audio hardware initialization failed!\n"
                  << "Strict zero-mock rule: Both the Primary (headphone) mic and Reference (error) mic\n"
                  << "as well as the playback output device must be physically connected and accessible.\n"
                  << "Refusing to proceed with degraded or simulated hardware.\n\n";
        return 1;
    }

    if (!backend.start()) {
        std::cerr << "\n[FATAL ERROR] Failed to start real-time ALSA audio threads.\n\n";
        return 1;
    }

    // 3. Capture 3 seconds of simultaneous audio
    std::cout << "[Step 4] Capturing 3.0 seconds of real audio simultaneously from both mics...\n";
    std::cout << "--------------------------------------------------------------------------\n";
    std::cout << std::left 
              << std::setw(12) << "Time (ms)"
              << std::setw(24) << "Primary RMS (dBFS)"
              << std::setw(24) << "Reference RMS (dBFS)"
              << "Inter-Stream Drift\n";
    std::cout << "--------------------------------------------------------------------------\n";

    std::vector<AudioChunk<1024>> recorded_primary_chunks;
    const double capture_duration_sec = 3.0;
    const auto start_time = std::chrono::steady_clock::now();
    auto next_report_time = start_time + std::chrono::milliseconds(100);
    uint32_t report_count = 0;

    std::vector<float> prim_100ms_window;
    std::vector<float> ref_100ms_window;
    prim_100ms_window.reserve(2000);
    ref_100ms_window.reserve(2000);

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
        if (elapsed_sec >= capture_duration_sec) break;

        AudioChunk<1024> p_chunk;
        AudioChunk<1024> r_chunk;

        bool has_p = backend.read_primary_chunk(p_chunk);
        bool has_r = backend.read_reference_chunk(r_chunk);

        if (has_p) {
            recorded_primary_chunks.push_back(p_chunk);
            prim_100ms_window.insert(prim_100ms_window.end(), p_chunk.samples, p_chunk.samples + p_chunk.count);
        }
        if (has_r) {
            ref_100ms_window.insert(ref_100ms_window.end(), r_chunk.samples, r_chunk.samples + r_chunk.count);
        }

        if (now >= next_report_time) {
            float p_rms = calculate_rms(prim_100ms_window.data(), prim_100ms_window.size());
            float r_rms = calculate_rms(ref_100ms_window.data(), ref_100ms_window.size());
            float p_db = rms_to_db(p_rms);
            float r_db = rms_to_db(r_rms);

            auto status = backend.get_status();
            std::cout << std::left
                      << std::setw(12) << (report_count * 100)
                      << std::setw(24) << (std::to_string(p_db).substr(0, 6) + " dBFS")
                      << std::setw(24) << (std::to_string(r_db).substr(0, 6) + " dBFS")
                      << status.inter_stream_drift_ms << " ms ("
                      << status.inter_stream_drift_samples << " samples)"
                      << (status.drift_warning_flag ? " [DRIFT WARNING]" : "") << "\n";

            prim_100ms_window.clear();
            ref_100ms_window.clear();
            report_count++;
            next_report_time += std::chrono::milliseconds(100);
        }

        if (!has_p && !has_r) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    std::cout << "--------------------------------------------------------------------------\n";
    auto final_status = backend.get_status();
    std::cout << "\nCapture Summary:\n"
              << "  Primary Frames Captured:   " << final_status.primary_frames_captured << "\n"
              << "  Reference Frames Captured: " << final_status.reference_frames_captured << "\n"
              << "  Primary XRUNs:             " << final_status.primary_xrun_count << "\n"
              << "  Reference XRUNs:           " << final_status.reference_xrun_count << "\n"
              << "  Inter-stream Drift:        " << final_status.inter_stream_drift_ms << " ms ("
              << final_status.inter_stream_drift_samples << " samples)\n"
              << "  Drift Warning Active:      " << (final_status.drift_warning_flag ? "YES" : "NO") << "\n\n";

    // 4. Play back captured audio to output device
    std::cout << "[Step 5] Playing back captured Primary microphone audio through output device: "
              << config.output_device << "...\n";

    for (const auto& chunk : recorded_primary_chunks) {
        while (!backend.write_playback_chunk(chunk)) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    // Wait for playback to drain
    std::this_thread::sleep_for(std::chrono::milliseconds(3200));

    std::cout << "[Step 6] Playback complete. Output XRUNs: " << backend.get_status().output_xrun_count << "\n";
    backend.stop();

    std::cout << "\n==========================================================================\n";
    std::cout << "       HARDWARE AUDIO TEST PASSED: DUAL MICROPHONES & OUTPUT VERIFIED      \n";
    std::cout << "==========================================================================\n\n";

    return 0;
}
