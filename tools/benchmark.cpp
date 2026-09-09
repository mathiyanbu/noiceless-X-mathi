#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cstring>

#include "embedded/ai/speech_enhancer.hpp"
#include "embedded/dsp/stft.hpp"

namespace {

struct WavHeader {
    char riff_id[4];
    uint32_t riff_size;
    char wave_id[4];
    char fmt_id[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data_id[4];
    uint32_t data_size;
};

// Simple WAV reader for 16-bit PCM mono audio
bool load_wav_pcm16(const std::string& filepath, std::vector<float>& samples, uint32_t& sample_rate) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        return false;
    }

    if (std::memcmp(header.riff_id, "RIFF", 4) != 0 ||
        std::memcmp(header.wave_id, "WAVE", 4) != 0) {
        return false;
    }

    sample_rate = header.sample_rate;
    size_t num_samples = header.data_size / sizeof(int16_t);
    std::vector<int16_t> raw_pcm(num_samples);
    file.read(reinterpret_cast<char*>(raw_pcm.data()), header.data_size);

    samples.resize(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = static_cast<float>(raw_pcm[i]) / 32768.0f;
    }

    return true;
}

// Generate realistic harmonic speech-like test signal if no WAV is specified
std::vector<float> generate_test_speech_signal(size_t total_samples, float sample_rate = 16000.0f) {
    std::vector<float> signal(total_samples, 0.0f);
    // Fundamental frequency 150 Hz + speech formants at 500 Hz, 1500 Hz, 2500 Hz
    for (size_t n = 0; n < total_samples; ++n) {
        float t = static_cast<float>(n) / sample_rate;
        float s = 0.5f * std::sin(2.0f * 3.14159265f * 150.0f * t)
                + 0.3f * std::sin(2.0f * 3.14159265f * 500.0f * t)
                + 0.15f * std::sin(2.0f * 3.14159265f * 1500.0f * t)
                + 0.05f * std::sin(2.0f * 3.14159265f * 2500.0f * t);
        // Add subtle broadband acoustic noise
        float noise = 0.02f * (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f);
        signal[n] = s + noise;
    }
    return signal;
}

struct LatencyStats {
    float min_ms{0.0f};
    float mean_ms{0.0f};
    float median_ms{0.0f};
    float p90_ms{0.0f};
    float p95_ms{0.0f};
    float p99_ms{0.0f};
    float max_ms{0.0f};
    float stddev_ms{0.0f};
    float rtf{0.0f};
    float headroom_pct{0.0f};
    bool passed{false};
};

LatencyStats compute_stats(std::vector<float>& latencies, float hop_duration_ms = 5.0f) {
    LatencyStats stats;
    if (latencies.empty()) return stats;

    std::sort(latencies.begin(), latencies.end());
    size_t n = latencies.size();

    stats.min_ms = latencies.front();
    stats.max_ms = latencies.back();

    float sum = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
    stats.mean_ms = sum / static_cast<float>(n);

    stats.median_ms = latencies[n / 2];
    stats.p90_ms = latencies[static_cast<size_t>(n * 0.90f)];
    stats.p95_ms = latencies[static_cast<size_t>(n * 0.95f)];
    stats.p99_ms = latencies[static_cast<size_t>(n * 0.99f)];

    float var_sum = 0.0f;
    for (float v : latencies) {
        float diff = v - stats.mean_ms;
        var_sum += diff * diff;
    }
    stats.stddev_ms = std::sqrt(var_sum / static_cast<float>(n));

    stats.rtf = stats.mean_ms / hop_duration_ms;
    stats.headroom_pct = (hop_duration_ms - stats.mean_ms) / hop_duration_ms * 100.0f;
    stats.passed = (stats.p95_ms < hop_duration_ms);

    return stats;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path = "models/onnx/speech_enhancer_int8.onnx";
    std::string wav_path = "tests/fixtures/speech_test.wav";
    int num_threads = 2;
    int warmup_frames = 30;
    int benchmark_frames = 200;

    // Parse command-line flags
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--wav" && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            benchmark_frames = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_frames = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --model <path>    Path to ONNX model (default: " << model_path << ")\n"
                      << "  --wav <path>      Path to 16kHz WAV file (default: " << wav_path << ")\n"
                      << "  --threads <N>     Intra-op thread count for Pi CPU EP (default: 2)\n"
                      << "  --frames <N>      Number of benchmark frames to infer (default: 200)\n"
                      << "  --warmup <N>      Number of warmup frames (default: 30)\n";
            return 0;
        }
    }

    std::cout << "================================================================================\n";
    std::cout << "       SIH26052 NOICELESSX — ONNX Runtime C++ Inference Engine Benchmark        \n";
    std::cout << "       Target: Raspberry Pi 4/5 (ARM NEON SIMD) | Zero-Mocks Hardware Eval     \n";
    std::cout << "================================================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Model Path:            " << model_path << "\n";
    std::cout << "  Intra-Op Threads:      " << num_threads << " (tuning for 4-core Pi headroom)\n";
    std::cout << "  Warmup Frames:         " << warmup_frames << "\n";
    std::cout << "  Benchmark Frames:      " << benchmark_frames << "\n";
    std::cout << "  Hop Size:              80 samples (5.0 ms @ 16kHz)\n";
    std::cout << "  FFT Size:              512 samples (257 frequency bins)\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // 1. Initialize SpeechEnhancer engine
    std::cout << "[Step 1] Initializing SpeechEnhancer session...\n";
    noiselessx::ai::SpeechEnhancer enhancer;
    bool init_ok = enhancer.initialize(model_path, num_threads);
    if (!init_ok) {
        std::cerr << "[Benchmark] FAILED to initialize engine: " << enhancer.get_last_error() << "\n";
        return 1;
    }
    std::cout << "  Provider: " << enhancer.get_execution_provider() << "\n";
    std::cout << "  Status:   SESSION READY\n\n";

    // 2. Load audio samples
    std::cout << "[Step 2] Preparing test audio samples...\n";
    std::vector<float> audio_samples;
    uint32_t sample_rate = 16000;
    if (load_wav_pcm16(wav_path, audio_samples, sample_rate)) {
        std::cout << "  Loaded WAV: " << wav_path << " (" << audio_samples.size() 
                  << " samples, " << sample_rate << " Hz)\n";
    } else {
        std::cout << "  WAV file '" << wav_path << "' not found; generating multi-harmonic test signal.\n";
        audio_samples = generate_test_speech_signal((warmup_frames + benchmark_frames + 10) * 80, 16000.0f);
    }

    // 3. Initialize STFT engine
    constexpr size_t FFT_SIZE = 512;
    constexpr size_t HOP_SIZE = 80;
    noiselessx::dsp::StftEngine stft_engine(FFT_SIZE, HOP_SIZE);
    noiselessx::dsp::SpectralFrame spectral_frame;
    spectral_frame.real.resize(257);
    spectral_frame.imag.resize(257);
    spectral_frame.magnitude.resize(257);
    spectral_frame.phase.resize(257);
    spectral_frame.num_bins = 257;

    noiselessx::ai::ComplexFrame input_frame(257);
    noiselessx::ai::ComplexFrame output_frame(257);
    std::vector<float> synthesized_hop(HOP_SIZE, 0.0f);

    size_t total_hops_available = (audio_samples.size() >= FFT_SIZE) ? 
        ((audio_samples.size() - FFT_SIZE) / HOP_SIZE) : (audio_samples.size() / HOP_SIZE);

    size_t actual_warmup = std::min(static_cast<size_t>(warmup_frames), total_hops_available / 4);
    size_t actual_bench = std::min(static_cast<size_t>(benchmark_frames), total_hops_available - actual_warmup);
    if (actual_bench == 0) {
        actual_bench = benchmark_frames;
    }

    // 4. Warmup phase
    std::cout << "[Step 3] Running " << actual_warmup << " warmup frames...\n";
    for (size_t i = 0; i < actual_warmup; ++i) {
        size_t offset = (i * HOP_SIZE) % (audio_samples.size() - HOP_SIZE);
        std::span<const float> hop_span(&audio_samples[offset], HOP_SIZE);
        stft_engine.process_analysis(hop_span, spectral_frame);

        std::memcpy(input_frame.real.data(), spectral_frame.real.data(), 257 * sizeof(float));
        std::memcpy(input_frame.imag.data(), spectral_frame.imag.data(), 257 * sizeof(float));

        if (!enhancer.infer(input_frame, output_frame)) {
            std::cerr << "[Benchmark] Warmup inference error: " << enhancer.get_last_error() << "\n";
            return 1;
        }
    }
    std::cout << "  Warmup completed. Persistent hidden state initialized.\n\n";

    // 5. Benchmark phase
    std::cout << "[Step 4] Measuring inference latency across " << actual_bench << " consecutive frames...\n";
    std::vector<float> latencies_ms;
    latencies_ms.reserve(actual_bench);

    for (size_t i = 0; i < actual_bench; ++i) {
        size_t offset = ((actual_warmup + i) * HOP_SIZE) % (audio_samples.size() - HOP_SIZE);
        std::span<const float> hop_span(&audio_samples[offset], HOP_SIZE);
        stft_engine.process_analysis(hop_span, spectral_frame);

        std::memcpy(input_frame.real.data(), spectral_frame.real.data(), 257 * sizeof(float));
        std::memcpy(input_frame.imag.data(), spectral_frame.imag.data(), 257 * sizeof(float));

        bool ok = enhancer.infer(input_frame, output_frame);
        if (!ok) {
            std::cerr << "[Benchmark] CRITICAL: Frame " << i << " failed with error: " 
                      << enhancer.get_last_error() << "\n";
            return 1;
        }

        latencies_ms.push_back(enhancer.get_last_inference_ms());

        // Perform iSTFT synthesis to verify complete audio pipeline roundtrip
        stft_engine.process_synthesis_complex(output_frame.real, output_frame.imag, synthesized_hop);
    }

    // 6. Compute and display statistics
    LatencyStats stats = compute_stats(latencies_ms, 5.0f);

    std::cout << "\n================================================================================\n";
    std::cout << "                          LATENCY BENCHMARK RESULTS                             \n";
    std::cout << "================================================================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Measured Iterations:       " << latencies_ms.size() << " frames\n";
    std::cout << "  Hop Time Budget:           5.000 ms (Real-Time Deadline)\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  Average (Mean) Latency:    " << std::setw(8) << stats.mean_ms << " ms\n";
    std::cout << "  Median (50th Percentile):  " << std::setw(8) << stats.median_ms << " ms\n";
    std::cout << "  90th Percentile:           " << std::setw(8) << stats.p90_ms << " ms\n";
    std::cout << "  95th Percentile:           " << std::setw(8) << stats.p95_ms << " ms\n";
    std::cout << "  99th Percentile:           " << std::setw(8) << stats.p99_ms << " ms\n";
    std::cout << "  Min Latency:               " << std::setw(8) << stats.min_ms << " ms\n";
    std::cout << "  Max Latency:               " << std::setw(8) << stats.max_ms << " ms\n";
    std::cout << "  Jitter (Std Deviation):    " << std::setw(8) << stats.stddev_ms << " ms\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  Real-Time Factor (RTF):    " << std::setw(8) << stats.rtf << "  (< 1.0 is Real-Time)\n";
    std::cout << "  CPU Headroom:              " << std::setw(8) << stats.headroom_pct << " %\n";
    std::cout << "  Real-Time Target Status:   " << (stats.passed ? "[PASSED - REAL-TIME READY]" : "[FAILED - EXCEEDS 5ms DEADLINE]") << "\n";
    std::cout << "================================================================================\n\n";

    return stats.passed ? 0 : 2;
}
