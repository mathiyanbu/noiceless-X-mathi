#include "speech_enhancer.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstring>
#include <chrono>
#include <cmath>

#if NOICELESSX_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
static std::wstring string_to_wstring(const std::string& s) {
    if (s.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &s[0], (int)s.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &s[0], (int)s.size(), &wstr[0], size_needed);
    return wstr;
}
#define TO_ORT_PATH(p) string_to_wstring(p).c_str()
#else
#define TO_ORT_PATH(p) (p).c_str()
#endif

#endif // NOICELESSX_HAS_ONNXRUNTIME

namespace noiselessx::ai {

static constexpr size_t NUM_BINS = 257;
static constexpr size_t STFT_CHANNELS = 2; // 0 = Real, 1 = Imag
static constexpr size_t STFT_FRAME_SIZE = STFT_CHANNELS * NUM_BINS; // 514 floats
static constexpr size_t GRU_LAYERS = 2;
static constexpr size_t GRU_HIDDEN_DIM = 256;
static constexpr size_t HIDDEN_STATE_SIZE = GRU_LAYERS * 1 * GRU_HIDDEN_DIM; // 512 floats

class SpeechEnhancer::Impl {
public:
    Impl() = default;
    ~Impl() = default;

#if NOICELESSX_HAS_ONNXRUNTIME
    bool initialize(const std::string& onnx_model_path, int num_threads) {
        last_error_.clear();
        degraded_mode_ = false;
        initialized_ = false;

        // Verify model file existence
        std::ifstream check_file(onnx_model_path, std::ios::binary);
        if (!check_file.is_open()) {
            last_error_ = "Model file not found or inaccessible: " + onnx_model_path;
            degraded_mode_ = true;
            std::cerr << "[SpeechEnhancer] ERROR: " << last_error_ << "\n";
            return false;
        }
        check_file.close();

        try {
            // 1. Initialize ORT Environment once
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NoiselessX_SpeechEnhancer");

            // 2. Configure Session Options
            session_options_ = Ort::SessionOptions();
            
            // Intra-op threads: tune for Pi's 4 cores (e.g. 2 threads to leave headroom for audio/DSP)
            session_options_.SetIntraOpNumThreads(num_threads > 0 ? num_threads : 2);
            session_options_.SetInterOpNumThreads(1);

            // Enable all ORT graph optimizations
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // 3. Execution Provider configuration
            execution_provider_ = "CPUExecutionProvider";
            
            // Attempt XNNPACK provider if available for ARM
            try {
                std::unordered_map<std::string, std::string> xnnpack_options;
                xnnpack_options["intra_op_num_threads"] = std::to_string(num_threads > 0 ? num_threads : 2);
                session_options_.AppendExecutionProvider("XNNPACK", xnnpack_options);
                execution_provider_ = "XNNPACK";
                std::cout << "[SpeechEnhancer] Enabled XNNPACK Execution Provider with " << num_threads << " threads.\n";
            } catch (...) {
                // Graceful fallback to default CPU Execution Provider (with ARM NEON SIMD)
                execution_provider_ = "CPUExecutionProvider";
                std::cout << "[SpeechEnhancer] Defaulting to CPUExecutionProvider (ARM NEON SIMD).\n";
            }

            // 4. Load ONNX Runtime session ONCE at startup
            session_ = std::make_unique<Ort::Session>(*env_, TO_ORT_PATH(onnx_model_path), session_options_);

            // 5. Preallocate input/output buffers
            input_stft_buffer_.assign(STFT_FRAME_SIZE, 0.0f);
            hidden_in_buffer_.assign(HIDDEN_STATE_SIZE, 0.0f);

            output_stft_buffer_.assign(STFT_FRAME_SIZE, 0.0f);
            output_mask_buffer_.assign(STFT_FRAME_SIZE, 0.0f);
            output_hidden_buffer_.assign(HIDDEN_STATE_SIZE, 0.0f);

            // Preallocate Ort::MemoryInfo for CPU arena
            memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // Define static tensor dimensions
            input_stft_dims_ = {1, 2, 1, static_cast<int64_t>(NUM_BINS)};
            hidden_dims_ = {static_cast<int64_t>(GRU_LAYERS), 1, static_cast<int64_t>(GRU_HIDDEN_DIM)};

            // Preallocate persistent input Ort::Value tensors pointing directly to class buffers
            input_tensors_.clear();
            input_tensors_.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                input_stft_buffer_.data(),
                input_stft_buffer_.size(),
                input_stft_dims_.data(),
                input_stft_dims_.size()
            ));
            input_tensors_.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                hidden_in_buffer_.data(),
                hidden_in_buffer_.size(),
                hidden_dims_.data(),
                hidden_dims_.size()
            ));

            // Preallocate persistent output Ort::Value tensors
            output_tensors_.clear();
            output_tensors_.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                output_stft_buffer_.data(),
                output_stft_buffer_.size(),
                input_stft_dims_.data(),
                input_stft_dims_.size()
            ));
            output_tensors_.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                output_mask_buffer_.data(),
                output_mask_buffer_.size(),
                input_stft_dims_.data(),
                input_stft_dims_.size()
            ));
            output_tensors_.push_back(Ort::Value::CreateTensor<float>(
                memory_info_,
                output_hidden_buffer_.data(),
                output_hidden_buffer_.size(),
                hidden_dims_.data(),
                hidden_dims_.size()
            ));

            input_node_names_ = {"noisy_stft", "hidden_in"};
            output_node_names_ = {"enhanced_stft", "mask", "hidden_out"};

            initialized_ = true;
            std::cout << "[SpeechEnhancer] Initialized session successfully. (Provider: " 
                      << execution_provider_ << ", Model: " << onnx_model_path << ")\n";
            return true;

        } catch (const Ort::Exception& oe) {
            last_error_ = std::string("ONNX Runtime Init Exception: ") + oe.what();
            degraded_mode_ = true;
            initialized_ = false;
            std::cerr << "[SpeechEnhancer] ERROR: " << last_error_ << "\n";
            return false;
        } catch (const std::exception& e) {
            last_error_ = std::string("Initialization Exception: ") + e.what();
            degraded_mode_ = true;
            initialized_ = false;
            std::cerr << "[SpeechEnhancer] ERROR: " << last_error_ << "\n";
            return false;
        }
    }

    bool infer(const ComplexFrame& input, ComplexFrame& output) {
        if (!initialized_ || degraded_mode_) {
            return false;
        }

        if (input.real.size() < NUM_BINS || input.imag.size() < NUM_BINS) {
            last_error_ = "Input ComplexFrame has insufficient bins (< 257).";
            degraded_mode_ = true;
            return false;
        }

        // Copy input spectral frame into preallocated buffer (zero heap allocations)
        // Channel 0 = Real, Channel 1 = Imag
        std::memcpy(input_stft_buffer_.data(), input.real.data(), NUM_BINS * sizeof(float));
        std::memcpy(input_stft_buffer_.data() + NUM_BINS, input.imag.data(), NUM_BINS * sizeof(float));

        if (output.real.size() != NUM_BINS) output.real.resize(NUM_BINS);
        if (output.imag.size() != NUM_BINS) output.imag.resize(NUM_BINS);
        output.num_bins = NUM_BINS;

        try {
            const auto start_time = std::chrono::high_resolution_clock::now();

            // Run inference with preallocated tensors (no allocations per frame)
            session_->Run(
                run_options_,
                input_node_names_.data(),
                input_tensors_.data(),
                input_tensors_.size(),
                output_node_names_.data(),
                output_tensors_.data(),
                output_tensors_.size()
            );

            const auto end_time = std::chrono::high_resolution_clock::now();
            last_inference_ms_ = std::chrono::duration<float, std::milli>(end_time - start_time).count();

            // Copy enhanced spectral components to output frame
            std::memcpy(output.real.data(), output_stft_buffer_.data(), NUM_BINS * sizeof(float));
            std::memcpy(output.imag.data(), output_stft_buffer_.data() + NUM_BINS, NUM_BINS * sizeof(float));

            // Persist recurrent hidden-state tensors between calls:
            // Feed the previous call's output hidden state as the next call's input hidden state
            std::memcpy(hidden_in_buffer_.data(), output_hidden_buffer_.data(), HIDDEN_STATE_SIZE * sizeof(float));

            return true;

        } catch (const Ort::Exception& oe) {
            degraded_mode_ = true;
            last_error_ = std::string("ONNX Runtime Inference Exception: ") + oe.what();
            std::cerr << "[SpeechEnhancer] CRITICAL: Inference failed, entering DEGRADED mode: " 
                      << last_error_ << "\n";
            return false;
        } catch (const std::exception& e) {
            degraded_mode_ = true;
            last_error_ = std::string("Inference Exception: ") + e.what();
            std::cerr << "[SpeechEnhancer] CRITICAL: Inference failed, entering DEGRADED mode: " 
                      << last_error_ << "\n";
            return false;
        } catch (...) {
            degraded_mode_ = true;
            last_error_ = "Unknown catastrophic exception during inference.";
            std::cerr << "[SpeechEnhancer] CRITICAL: " << last_error_ << "\n";
            return false;
        }
    }

    void reset_state() {
        if (hidden_in_buffer_.size() == HIDDEN_STATE_SIZE) {
            std::fill(hidden_in_buffer_.begin(), hidden_in_buffer_.end(), 0.0f);
        }
        if (output_hidden_buffer_.size() == HIDDEN_STATE_SIZE) {
            std::fill(output_hidden_buffer_.begin(), output_hidden_buffer_.end(), 0.0f);
        }
    }

    [[nodiscard]] float get_last_inference_ms() const noexcept { return last_inference_ms_; }
    [[nodiscard]] bool is_degraded() const noexcept { return degraded_mode_; }
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
    [[nodiscard]] const std::string& get_execution_provider() const noexcept { return execution_provider_; }
    [[nodiscard]] const std::string& get_last_error() const noexcept { return last_error_; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    Ort::RunOptions run_options_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    std::vector<int64_t> input_stft_dims_;
    std::vector<int64_t> hidden_dims_;

    std::vector<float> input_stft_buffer_;
    std::vector<float> hidden_in_buffer_;

    std::vector<float> output_stft_buffer_;
    std::vector<float> output_mask_buffer_;
    std::vector<float> output_hidden_buffer_;

    std::vector<Ort::Value> input_tensors_;
    std::vector<Ort::Value> output_tensors_;

    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    std::string execution_provider_{"CPUExecutionProvider"};
    std::string last_error_;
    float last_inference_ms_{0.0f};
    bool initialized_{false};
    bool degraded_mode_{false};

#else // !NOICELESSX_HAS_ONNXRUNTIME
    // Dev host fallback stub adhering strictly to zero-mock rule:
    // Refuses to claim READY or fabricate fake outputs without real ONNX Runtime engine.
    bool initialize(const std::string& onnx_model_path, int /*num_threads*/) {
        last_error_ = "ONNX Runtime C++ headers/library not present on this host platform. "
                      "Engine refuses to claim READY without real hardware backend. Path: " + onnx_model_path;
        initialized_ = false;
        degraded_mode_ = true;
        std::cerr << "[SpeechEnhancer] HOST WARNING: " << last_error_ << "\n";
        return false;
    }

    bool infer(const ComplexFrame& /*input*/, ComplexFrame& /*output*/) {
        last_error_ = "Inference requested on host without ONNX Runtime backend. Failing cleanly.";
        degraded_mode_ = true;
        return false;
    }

    void reset_state() {}
    [[nodiscard]] float get_last_inference_ms() const noexcept { return 0.0f; }
    [[nodiscard]] bool is_degraded() const noexcept { return true; }
    [[nodiscard]] bool is_initialized() const noexcept { return false; }
    [[nodiscard]] const std::string& get_execution_provider() const noexcept { return host_provider_; }
    [[nodiscard]] const std::string& get_last_error() const noexcept { return last_error_; }

private:
    std::string host_provider_{"None (Dev Host Stub)"};
    std::string last_error_{"ONNX Runtime not compiled on dev host"};
    bool initialized_{false};
    bool degraded_mode_{true};
#endif
};

SpeechEnhancer::SpeechEnhancer() : impl_(std::make_unique<Impl>()) {}
SpeechEnhancer::~SpeechEnhancer() = default;
SpeechEnhancer::SpeechEnhancer(SpeechEnhancer&&) noexcept = default;
SpeechEnhancer& SpeechEnhancer::operator=(SpeechEnhancer&&) noexcept = default;

bool SpeechEnhancer::initialize(const std::string& onnx_model_path, int num_threads) {
    return impl_->initialize(onnx_model_path, num_threads);
}

bool SpeechEnhancer::infer(const ComplexFrame& input, ComplexFrame& output) {
    return impl_->infer(input, output);
}

void SpeechEnhancer::reset_state() {
    impl_->reset_state();
}

float SpeechEnhancer::get_last_inference_ms() const {
    return impl_->get_last_inference_ms();
}

bool SpeechEnhancer::is_degraded() const noexcept {
    return impl_->is_degraded();
}

bool SpeechEnhancer::is_initialized() const noexcept {
    return impl_->is_initialized();
}

const std::string& SpeechEnhancer::get_execution_provider() const noexcept {
    return impl_->get_execution_provider();
}

const std::string& SpeechEnhancer::get_last_error() const noexcept {
    return impl_->get_last_error();
}

} // namespace noiselessx::ai
