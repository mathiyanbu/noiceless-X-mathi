#include "realtime_pipeline.hpp"
#include "ipc_server.hpp"
#include "embedded/audio/device_enumerator.hpp"
#include "embedded/fusion/fusion_types.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>

using namespace noiselessx::runtime;
using namespace noiselessx::audio;
using namespace noiselessx::fusion;

namespace {

std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        g_shutdown_requested.store(true);
    }
}

void print_help(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --config <path>     Path to configuration YAML file (default: config/raspberrypi.yaml)\n"
              << "  --realtime          Start the active real-time dual-mic audio processing pipeline\n"
              << "  --test-devices      Enumerate and validate connected ALSA audio devices and exit\n"
              << "  --duration <sec>    Run duration in seconds (default: 0 = continuous until Ctrl+C)\n"
              << "  --bypass            Start pipeline with fusion engine in operator BYPASS mode\n"
              << "  --help              Display this help message and exit\n";
}

bool parse_yaml_config(const std::string& filepath, RealtimePipeline::PipelineConfig& config) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[sih26052] Warning: Could not open config file: " << filepath 
                  << ". Using default configuration.\n";
        return false;
    }

    std::string line;
    std::string current_section;
    auto parse_bool = [](const std::string& value, bool fallback) {
        if (value == "true" || value == "1" || value == "yes") return true;
        if (value == "false" || value == "0" || value == "no") return false;
        return fallback;
    };

    while (std::getline(file, line)) {
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        line = line.substr(first);

        auto colon = line.find(':');
        if (colon == std::string::npos) continue;

        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);

        auto trim = [](std::string& s) {
            auto f = s.find_first_not_of(" \t\r\n\"'");
            if (f == std::string::npos) { s = ""; return; }
            auto l = s.find_last_not_of(" \t\r\n\"'");
            s = s.substr(f, (l - f + 1));
        };
        trim(key);
        trim(val);

        if (val.empty() && colon == line.length() - 1) {
            current_section = key;
            continue;
        }

        try {
            if (current_section == "audio") {
                if (key == "sample_rate") config.audio.sample_rate = std::stoul(val);
                else if (key == "channels") config.audio.channels = std::stoul(val);
                else if (key == "hardware_sample_rate") config.audio.hardware_sample_rate = std::stoul(val);
                else if (key == "hardware_channels") config.audio.hardware_channels = std::stoul(val);
                else if (key == "frame_ms") config.audio.frame_ms = std::stoul(val);
                else if (key == "hop_ms") config.audio.hop_ms = std::stoul(val);
                else if (key == "fft_size") config.audio.fft_size = std::stoul(val);
                else if (key == "buffer_frames") config.audio.period_size = std::stoul(val);
                else if (key == "periods") config.audio.buffer_size = config.audio.period_size * std::stoul(val);
                else if (key == "input_device") config.audio.primary_device = val;
                else if (key == "reference_device") config.audio.reference_device = val;
                else if (key == "output_device") config.audio.output_device = val;
                else if (key == "single_mic") config.audio.single_mic = parse_bool(val, config.audio.single_mic);
            } else if (current_section == "ai") {
                if (key == "model_path") config.onnx_model_path = val;
                else if (key == "intra_op_num_threads") config.ai_threads = std::stoi(val);
            } else if (current_section == "nlms") {
                if (key == "enabled") config.audio.nlms_enabled = parse_bool(val, config.audio.nlms_enabled);
                else if (key == "filter_length") config.nlms_filter_length = std::stoi(val);
                else if (key == "learning_rate" || key == "step_size") config.nlms_learning_rate = std::stof(val);
                else if (key == "epsilon" || key == "eps") config.nlms_epsilon = std::stof(val);
            } else if (current_section == "impulse") {
                if (key == "threshold_on") config.impulse.threshold_on = std::stof(val);
                else if (key == "threshold_off") config.impulse.threshold_off = std::stof(val);
                else if (key == "attack_alpha") config.impulse.attack_alpha = std::stof(val);
                else if (key == "release_alpha") config.impulse.release_alpha = std::stof(val);
                else if (key == "model_path") config.impulse.onnx_model_path = val;
            }
        } catch (...) {
            // Keep default on parsing error
        }
    }

    if (config.audio.buffer_size == 0 && config.audio.period_size > 0) {
        config.audio.buffer_size = config.audio.period_size * 4;
    }

    return true;
}

void print_device_validation_status(const AudioConfig& config) {
    std::cout << "================================================================================\n";
    std::cout << "               ALSA HARDWARE DEVICE VALIDATION & ENUMERATION                   \n";
    std::cout << "================================================================================\n";

    auto devices = DeviceEnumerator::enumerate_devices();

    std::cout << std::left << std::setw(6)  << "Card"
              << std::setw(8)  << "Device"
              << std::setw(10) << "Type"
              << std::setw(24) << "ALSA Identifier"
              << std::setw(32) << "Hardware Name" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    for (const auto& dev : devices) {
        std::string type_str;
        if (dev.is_capture && dev.is_playback) type_str = "Duplex";
        else if (dev.is_capture) type_str = "Capture";
        else if (dev.is_playback) type_str = "Playback";
        else type_str = "Unknown";

        std::cout << std::left << std::setw(6)  << dev.card_index
                  << std::setw(8)  << dev.device_index
                  << std::setw(10) << type_str
                  << std::setw(24) << dev.alsa_identifier
                  << std::setw(32) << (dev.device_name.empty() ? dev.card_name : dev.device_name) << "\n";
    }
    std::cout << "--------------------------------------------------------------------------------\n";

    std::cout << "\nConfigured Target Audio Routes:\n";
    std::cout << "  [Primary Mic]   (Speech + Ambient): " << config.primary_device << "\n";
    std::cout << "  [Reference Mic] (Noise Reference):  " << config.reference_device << "\n";
    std::cout << "  [Playback DAC]  (Headphone Output): " << config.output_device << "\n";
    std::cout << "  [Sampling Rate] " << config.sample_rate << " Hz | Hop: " 
              << (config.period_size / 2) << " samples (" << config.hop_ms << " ms)\n";
    std::cout << "  [Mode]          " << (config.single_mic ? "Single microphone" : "Dual microphone") << "\n";
    std::cout << "================================================================================\n\n";
}

void render_telemetry_dashboard(const PipelineTelemetry& t) {
    // Clear screen and move cursor to top
    std::cout << "\033[H\033[J";

    std::cout << "================================================================================\n";
    std::cout << "       SIH26052 NOICELESSX — REALTIME DUAL-MIC TELEMETRY (Raspberry Pi)        \n";
    std::cout << "================================================================================\n";

    // Header Status
    std::cout << "  Engine State:        RUNNING [SCHED_FIFO Audio / Processing Threads]\n";
    std::cout << "  Processed Frames:    " << t.processed_frames 
              << " | Dropped: " << t.dropped_frames << "\n";
    std::cout << "  ALSA XRUNs:          Primary=" << t.alsa_xruns_primary 
              << " | Ref=" << t.alsa_xruns_reference 
              << " | Playback=" << t.alsa_xruns_playback << "\n";
    std::cout << "  Clock Skew (Drift):  " << std::fixed << std::setprecision(2) 
              << t.drift_ms << " ms (" << t.drift_samples << " samples) " 
              << (t.drift_warning ? "[WARNING: DRIFT EXCEEDED]" : "[OK]") << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Stage Latency Breakdown
    std::cout << "  STAGE LATENCY BREAKDOWN:\n";
    std::cout << "    [1] Audio Capture:     " << std::setw(7) << std::fixed << std::setprecision(1) << t.capture_us << " us\n";
    std::cout << "    [2] Preproc (DC+HPF):  " << std::setw(7) << t.preprocessing_us << " us\n";
    std::cout << "    [3] STFT Analysis:     " << std::setw(7) << t.stft_us << " us\n";
    std::cout << "    [4] AI Speech Enhance: " << std::setw(7) << t.ai_inference_us << " us\n";
    std::cout << "    [5] iSTFT Synthesis:   " << std::setw(7) << t.istft_us << " us\n";
    std::cout << "    [6] NLMS Adaptive Canc:" << std::setw(7) << t.nlms_us << " us\n";
    std::cout << "    [7] Multi-Mode Fusion: " << std::setw(7) << t.fusion_us << " us\n";
    std::cout << "    [8] Playback Output:   " << std::setw(7) << t.playback_queue_us << " us\n";
    std::cout << "    ------------------------------------\n";
    std::cout << "    Total Processing:      " << std::setw(7) << t.total_processing_us << " us\n";
    std::cout << "    End-to-End Latency:    " << std::setw(7) << std::setprecision(2) << t.end_to_end_latency_ms << " ms\n";
    std::cout << "    Real-Time Factor (RTF):" << std::setw(7) << std::setprecision(3) << t.rtf 
              << "x (Target < 0.50x) " << (t.rtf < 0.50 ? "[HEADROOM OK]" : "[LOAD HIGH]") << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Multi-Mode Fusion State
    std::cout << "  FUSION STATE MACHINE (Phase 9 Active Controller):\n";
    std::cout << "    Active Mode:           " << to_string(t.fusion_mode) << "\n";
    std::cout << "    Dynamic Lambda:        " << std::setprecision(3) << t.current_lambda 
              << " (AI: " << static_cast<int>(t.current_lambda * 100) << "% | NLMS: " 
              << static_cast<int>((1.0f - t.current_lambda) * 100) << "%)\n";
    std::cout << "    Impulse Gain Envelope: " << std::setprecision(3) << t.impulse_envelope_gain << "\n";
    std::cout << "    Signals: VAD=" << std::setprecision(2) << (t.vad_probability * 100.0f) << "%"
              << " | ImpulseProb=" << (t.impulse_probability * 100.0f) << "%"
              << " | AIConfidence=" << (t.ai_confidence * 100.0f) << "%\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // System Metrics (Real OS measurements)
    std::cout << "  SYSTEM HARDWARE TELEMETRY:\n";
    std::cout << "    Overall CPU Load:      " << std::setprecision(1) << t.overall_cpu_pct << "%\n";
    if (!t.cpu_per_core.empty()) {
        std::cout << "    Per-Core Utilization:  ";
        for (size_t i = 0; i < t.cpu_per_core.size(); ++i) {
            std::cout << "Core " << i << ": " << std::setw(4) << std::setprecision(1) << t.cpu_per_core[i] << "% ";
            if (i < t.cpu_per_core.size() - 1) std::cout << "| ";
        }
        std::cout << "\n";
    }
    if (t.temp_available) {
        std::cout << "    CPU Temperature:       " << std::setprecision(1) << t.cpu_temperature_c << " °C\n";
    } else {
        std::cout << "    CPU Temperature:       N/A (Host Development Environment)\n";
    }
    std::cout << "================================================================================\n";
    std::cout << "  Press Ctrl+C to stop real-time processing and safely flush hardware buffers.\n" << std::flush;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    std::string config_path = "config/raspberrypi.yaml";
    bool realtime_mode = false;
    bool test_devices_only = false;
    bool bypass_mode = false;
    uint32_t run_duration_sec = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--realtime") {
            realtime_mode = true;
        } else if (arg == "--test-devices") {
            test_devices_only = true;
        } else if (arg == "--bypass") {
            bypass_mode = true;
        } else if (arg == "--duration" && i + 1 < argc) {
            run_duration_sec = std::stoul(argv[++i]);
        }
    }

    std::cout << "\nSIH26052 — NOICELESSX Real-Time Dual-Mic Audio Subsystem\n";
    std::cout << "Loading configuration: " << config_path << "...\n";

    RealtimePipeline::PipelineConfig pipeline_config;
    parse_yaml_config(config_path, pipeline_config);

    if (test_devices_only) {
        print_device_validation_status(pipeline_config.audio);
        return 0;
    }

    print_device_validation_status(pipeline_config.audio);

    if (!realtime_mode) {
        std::cout << "[sih26052] Notice: Launching in hardware configuration check mode.\n"
                  << "To activate the live real-time audio pipeline, run with:\n"
                  << "  " << argv[0] << " --config " << config_path << " --realtime\n\n";
        return 0;
    }

    // Register signal handlers for clean shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    RealtimePipeline pipeline(pipeline_config);
    if (!pipeline.initialize()) {
        std::cerr << "[sih26052] Initialization failed. Exiting.\n";
        return 1;
    }

    if (bypass_mode) {
        pipeline.set_bypass(true);
    }

    // Launch Local IPC Server for FastAPI observation and control
    IpcServer ipc_server(9099, "/tmp/noiselessx.sock");
    ipc_server.set_request_handler([&pipeline, &pipeline_config](const std::string& cmd, const std::string& raw_request) -> std::string {
        if (cmd == "ping") {
            std::ostringstream ss;
            ss << "{\"status\":\"ok\",\"runtime\":\"" << (pipeline.is_running() ? "running" : "ready") << "\"}";
            return ss.str();
        } else if (cmd == "get_metrics" || cmd == "get_telemetry") {
            auto t = pipeline.get_telemetry();
            std::ostringstream ss;
            ss << "{\"status\":\"ok\",\"metrics\":{"
               << "\"latencies\":{"
               << "\"capture_us\":" << t.capture_us << ","
               << "\"preprocessing_us\":" << t.preprocessing_us << ","
               << "\"stft_us\":" << t.stft_us << ","
               << "\"ai_inference_us\":" << t.ai_inference_us << ","
               << "\"istft_us\":" << t.istft_us << ","
               << "\"nlms_us\":" << t.nlms_us << ","
               << "\"fusion_us\":" << t.fusion_us << ","
               << "\"playback_queue_us\":" << t.playback_queue_us << ","
               << "\"total_processing_us\":" << t.total_processing_us << ","
               << "\"end_to_end_latency_ms\":" << t.end_to_end_latency_ms
               << "},"
               << "\"rtf\":" << t.rtf << ","
               << "\"processed_frames\":" << t.processed_frames << ","
               << "\"dropped_frames\":" << t.dropped_frames << ","
               << "\"alsa_xruns\":{"
               << "\"primary\":" << t.alsa_xruns_primary << ","
               << "\"reference\":" << t.alsa_xruns_reference << ","
               << "\"playback\":" << t.alsa_xruns_playback
               << "},"
               << "\"drift_ms\":" << t.drift_ms << ","
               << "\"drift_samples\":" << t.drift_samples << ","
               << "\"drift_warning\":" << (t.drift_warning ? "true" : "false") << ","
               << "\"fusion_mode\":\"" << to_string(t.fusion_mode) << "\","
               << "\"current_lambda\":" << t.current_lambda << ","
               << "\"impulse_envelope_gain\":" << t.impulse_envelope_gain << ","
               << "\"ai_confidence\":" << t.ai_confidence << ","
               << "\"impulse_probability\":" << t.impulse_probability << ","
               << "\"vad_probability\":" << t.vad_probability
               << "}}";
            return ss.str();
        } else if (cmd == "get_system") {
            auto t = pipeline.get_telemetry();
            std::ostringstream ss;
            ss << "{\"status\":\"ok\",\"cpu\":{"
               << "\"overall_pct\":" << t.overall_cpu_pct << ","
               << "\"cores_pct\":[";
            for (size_t i = 0; i < t.cpu_per_core.size(); ++i) {
                ss << t.cpu_per_core[i];
                if (i + 1 < t.cpu_per_core.size()) ss << ",";
            }
            ss << "]},\"temperature_c\":" << t.cpu_temperature_c
               << ",\"temperature_available\":" << (t.temp_available ? "true" : "false") << "}";
            return ss.str();
        } else if (cmd == "start") {
            bool ok = pipeline.start();
            return ok ? "{\"status\":\"ok\",\"command\":\"start\",\"runtime_state\":\"running\"}"
                      : "{\"status\":\"error\",\"command\":\"start\",\"message\":\"Failed to start pipeline\"}";
        } else if (cmd == "stop") {
            pipeline.stop();
            return "{\"status\":\"ok\",\"command\":\"stop\",\"runtime_state\":\"stopped\"}";
        } else if (cmd == "bypass") {
            bool bp = (raw_request.find("\"bypass\": true") != std::string::npos ||
                       raw_request.find("\"bypass\":true") != std::string::npos);
            pipeline.set_bypass(bp);
            std::ostringstream ss;
            ss << "{\"status\":\"ok\",\"command\":\"bypass\",\"runtime_state\":\""
               << (pipeline.is_running() ? "running" : "ready") << "\",\"bypass\":" << (bp ? "true" : "false") << "}";
            return ss.str();
        } else if (cmd == "reset") {
            pipeline.reset();
            return "{\"status\":\"ok\",\"command\":\"reset\",\"runtime_state\":\"ready\"}";
        } else if (cmd == "get_devices") {
            std::ostringstream ss;
            ss << "{\"status\":\"ok\",\"config\":{"
               << "\"sample_rate\":" << pipeline_config.audio.sample_rate << ","
               << "\"channels\":" << pipeline_config.audio.channels << ","
               << "\"hardware_sample_rate\":" << pipeline_config.audio.hardware_sample_rate << ","
               << "\"hardware_channels\":" << pipeline_config.audio.hardware_channels << ","
               << "\"frame_ms\":" << pipeline_config.audio.frame_ms << ","
               << "\"hop_ms\":" << pipeline_config.audio.hop_ms << ","
               << "\"primary_device\":\"" << pipeline_config.audio.primary_device << "\","
               << "\"reference_device\":\"" << pipeline_config.audio.reference_device << "\","
               << "\"output_device\":\"" << pipeline_config.audio.output_device << "\","
               << "\"single_mic\":" << (pipeline_config.audio.single_mic ? "true" : "false") << ","
               << "\"nlms_enabled\":" << (pipeline_config.audio.nlms_enabled ? "true" : "false") << ","
               << "\"period_size\":" << pipeline_config.audio.period_size << ","
               << "\"buffer_size\":" << pipeline_config.audio.buffer_size
               << "}}";
            return ss.str();
        } else if (cmd == "get_model") {
            std::ostringstream ss;
            ss << "{\"status\":\"ok\",\"model\":{"
               << "\"model_path\":\"" << pipeline_config.onnx_model_path << "\","
               << "\"model_name\":\"ComplexCRN\","
               << "\"quantization\":\"INT8 (ARM NEON Optimized)\","
               << "\"execution_provider\":\"CPUExecutionProvider\","
               << "\"intra_op_threads\":" << pipeline_config.ai_threads << ","
               << "\"input_shape\":[\"noisy_stft: (B, 2, T, 257)\", \"hidden_in: (2, B, 128)\"],"
               << "\"output_shape\":[\"enhanced_stft: (B, 2, T, 257)\", \"mask: (B, 2, T, 257)\", \"hidden_out: (2, B, 128)\"]"
               << "}}";
            return ss.str();
        }
        return "{\"status\":\"error\",\"message\":\"Unknown command\"}";
    });
    ipc_server.start();

    if (!pipeline.start()) {
        std::cerr << "[sih26052] Failed to start real-time pipeline. Exiting.\n";
        ipc_server.stop();
        return 1;
    }

    const auto start_time = std::chrono::steady_clock::now();
    while (!g_shutdown_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        PipelineTelemetry t = pipeline.get_telemetry();
        render_telemetry_dashboard(t);

        if (run_duration_sec > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            if (static_cast<uint32_t>(elapsed) >= run_duration_sec) {
                break;
            }
        }
    }

    std::cout << "\n[sih26052] Shutdown signal received. Draining audio buffers...\n";
    ipc_server.stop();
    pipeline.stop();
    std::cout << "[sih26052] Real-time engine terminated cleanly.\n";

    return 0;
}
