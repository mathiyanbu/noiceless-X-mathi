#include "impulse_detector.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

#if NOICELESSX_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
static std::wstring to_wstr(const std::string& s) {
    if (s.empty()) return std::wstring();
    int size = MultiByteToWideChar(CP_UTF8, 0, &s[0], (int)s.size(), NULL, 0);
    std::wstring ws(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, &s[0], (int)s.size(), &ws[0], size);
    return ws;
}
#define ORT_PATH(p) to_wstr(p).c_str()
#else
#define ORT_PATH(p) (p).c_str()
#endif

#endif

namespace noiselessx {
namespace impulse {

class ImpulseDetector::OnnxBackend {
public:
    OnnxBackend() = default;
    ~OnnxBackend() = default;

#if NOICELESSX_HAS_ONNXRUNTIME
    bool initialize(const std::string& model_path, int num_threads) {
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ImpulseDetectorEnv");
            session_options_ = Ort::SessionOptions();
            session_options_.SetIntraOpNumThreads(num_threads > 0 ? num_threads : 1);
            session_options_.SetInterOpNumThreads(1);
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            session_ = std::make_unique<Ort::Session>(*env_, ORT_PATH(model_path), session_options_);
            memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            input_buffer_.assign(8, 0.0f);
            output_buffer_.assign(1, 0.0f);

            input_shape_ = {1, 8};
            output_shape_ = {1, 1};

            input_names_ = {"features"};
            output_names_ = {"probability"};

            ready_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[ImpulseDetector] Failed to initialize ONNX model: " << e.what() << "\n";
            ready_ = false;
            return false;
        }
    }

    bool predict(const ImpulseFeatures& feats, float& out_prob) {
        if (!ready_ || !session_) return false;

        input_buffer_[0] = feats.rms;
        input_buffer_[1] = feats.spectral_flux;
        input_buffer_[2] = feats.crest_factor;
        input_buffer_[3] = feats.zcr;
        input_buffer_[4] = feats.band_energies[0];
        input_buffer_[5] = feats.band_energies[1];
        input_buffer_[6] = feats.band_energies[2];
        input_buffer_[7] = feats.band_energies[3];

        try {
            std::vector<Ort::Value> in_tensors;
            in_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, input_buffer_.data(), input_buffer_.size(),
                input_shape_.data(), input_shape_.size()
            ));

            std::vector<Ort::Value> out_tensors;
            out_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, output_buffer_.data(), output_buffer_.size(),
                output_shape_.data(), output_shape_.size()
            ));

            session_->Run(
                Ort::RunOptions{nullptr},
                input_names_.data(), in_tensors.data(), 1,
                output_names_.data(), out_tensors.data(), 1
            );

            out_prob = std::clamp(output_buffer_[0], 0.0f, 1.0f);
            return true;
        } catch (...) {
            return false;
        }
    }

    [[nodiscard]] bool is_ready() const noexcept { return ready_; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_{nullptr};

    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    bool ready_{false};

#else
    bool initialize(const std::string&, int) { return false; }
    bool predict(const ImpulseFeatures&, float&) { return false; }
    [[nodiscard]] bool is_ready() const noexcept { return false; }
#endif
};

ImpulseDetector::ImpulseDetector(const ImpulseConfig& config)
    : config_(config),
      onnx_backend_(std::make_unique<OnnxBackend>()) {
    prev_magnitude_.assign(257, 0.0f);
}

ImpulseDetector::~ImpulseDetector() = default;
ImpulseDetector::ImpulseDetector(ImpulseDetector&&) noexcept = default;
ImpulseDetector& ImpulseDetector::operator=(ImpulseDetector&&) noexcept = default;

bool ImpulseDetector::initialize_onnx(const std::string& model_path, int num_threads) {
    config_.onnx_model_path = model_path;
    config_.use_onnx = onnx_backend_->initialize(model_path, num_threads);
    return config_.use_onnx;
}

ImpulseFeatures ImpulseDetector::extract_features(
    std::span<const float> time_frame,
    std::span<const float> spectral_magnitude
) {
    ImpulseFeatures feats;
    const size_t N = time_frame.size();
    if (N == 0) return feats;

    // 1. RMS: sqrt((1/N) * sum(x[n]^2))
    float sum_sq = 0.0f;
    float max_abs = 0.0f;
    for (size_t n = 0; n < N; ++n) {
        float val = time_frame[n];
        sum_sq += val * val;
        float abs_val = std::abs(val);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    feats.rms = std::sqrt(sum_sq / static_cast<float>(N));

    // 2. Crest Factor: max|x[n]| / (RMS + eps)
    feats.crest_factor = max_abs / (feats.rms + 1e-7f);

    // 3. Zero-Crossing Rate (ZCR): (1/(N-1)) * sum |sign(x[n]) - sign(x[n-1])| / 2
    if (N > 1) {
        uint32_t zero_crossings = 0;
        int prev_sign = (time_frame[0] >= 0.0f) ? 1 : -1;
        for (size_t n = 1; n < N; ++n) {
            int curr_sign = (time_frame[n] >= 0.0f) ? 1 : -1;
            if (curr_sign != prev_sign) {
                zero_crossings++;
                prev_sign = curr_sign;
            }
        }
        feats.zcr = static_cast<float>(zero_crossings) / static_cast<float>(N - 1);
    }

    // 4. Spectral Flux (Half-Wave Rectified): sum_k (max(0, |X(k, m)| - |X(k, m-1)|))^2
    const size_t num_bins = std::min(spectral_magnitude.size(), prev_magnitude_.size());
    float flux = 0.0f;
    for (size_t k = 0; k < num_bins; ++k) {
        float diff = spectral_magnitude[k] - prev_magnitude_[k];
        if (diff > 0.0f) {
            flux += diff * diff;
        }
    }
    feats.spectral_flux = flux;

    // Update persistent previous magnitude buffer
    if (spectral_magnitude.size() == prev_magnitude_.size()) {
        std::copy(spectral_magnitude.begin(), spectral_magnitude.end(), prev_magnitude_.begin());
    } else {
        prev_magnitude_.assign(spectral_magnitude.begin(), spectral_magnitude.end());
    }

    // 5. Subband Energies: sum_{k in band} |X(k)|^2 (averaged by band bin count)
    // At 16kHz with N=512, bin width is 31.25 Hz
    // Band 0: Low (0 - 500 Hz): bins 0..16 (16 bins)
    // Band 1: Mid-Low (500 - 2000 Hz): bins 16..64 (48 bins)
    // Band 2: Mid-High (2000 - 4000 Hz): bins 64..128 (64 bins)
    // Band 3: High (4000 - 8000 Hz): bins 128..256 (128 bins)
    static constexpr size_t band_splits[5] = {0, 16, 64, 128, 256};
    for (size_t b = 0; b < 4; ++b) {
        size_t start = band_splits[b];
        size_t end = std::min(band_splits[b + 1], spectral_magnitude.size());
        float band_sum = 0.0f;
        if (end > start) {
            for (size_t k = start; k < end; ++k) {
                float mag = spectral_magnitude[k];
                band_sum += mag * mag;
            }
            feats.band_energies[b] = band_sum / static_cast<float>(end - start);
        }
    }

    return feats;
}

float ImpulseDetector::compute_raw_probability(const ImpulseFeatures& feats) {
    if (config_.use_onnx && onnx_backend_->is_ready()) {
        float onnx_prob = 0.0f;
        if (onnx_backend_->predict(feats, onnx_prob)) {
            return onnx_prob;
        }
    }

    // Hand-tuned logistic regression combination
    const auto& w = config_.weights;
    float z = w.w_rms * feats.rms
            + w.w_flux * feats.spectral_flux
            + w.w_crest * (feats.crest_factor - 1.0f)
            + w.w_zcr * feats.zcr
            + w.w_bands[0] * feats.band_energies[0]
            + w.w_bands[1] * feats.band_energies[1]
            + w.w_bands[2] * feats.band_energies[2]
            + w.w_bands[3] * feats.band_energies[3]
            + w.bias;

    // Numerical clamp to prevent exp overflow
    z = std::clamp(z, -30.0f, 30.0f);
    return 1.0f / (1.0f + std::exp(-z));
}

ImpulseState ImpulseDetector::detect(
    std::span<const float> time_frame,
    std::span<const float> spectral_magnitude
) {
    ImpulseState state;
    if (!config_.enabled) {
        state.probability = 0.0f;
        state.detected = false;
        state.attack_gain = 1.0f;
        state.release_gain = 1.0f;
        return state;
    }

    // 1. Compute features
    ImpulseFeatures feats = extract_features(time_frame, spectral_magnitude);

    // 2. Compute instantaneous raw probability
    float raw_prob = compute_raw_probability(feats);

    // 3. Asymmetric temporal smoothing:
    // Fast attack when probability rises, slower release when it falls
    if (raw_prob > smoothed_prob_) {
        smoothed_prob_ = config_.attack_alpha * raw_prob 
                       + (1.0f - config_.attack_alpha) * smoothed_prob_;
    } else {
        smoothed_prob_ = config_.release_alpha * raw_prob 
                       + (1.0f - config_.release_alpha) * smoothed_prob_;
    }
    smoothed_prob_ = std::clamp(smoothed_prob_, 0.0f, 1.0f);

    // 4. Dual-threshold hysteresis state machine
    if (!detected_latched_) {
        if (smoothed_prob_ >= config_.threshold_on) {
            detected_latched_ = true;
        }
    } else {
        if (smoothed_prob_ <= config_.threshold_off) {
            detected_latched_ = false;
        }
    }

    // 5. Gain calculation for transient suppression
    // Instantaneous attack gain attenuates impulse
    float target_gain = 1.0f - smoothed_prob_;
    target_gain = std::max(target_gain, 0.05f); // 5% minimum floor to avoid absolute mute

    if (target_gain < current_gain_) {
        // Fast attack gain reduction
        current_gain_ = config_.attack_alpha * target_gain + (1.0f - config_.attack_alpha) * current_gain_;
    } else {
        // Slower recovery release
        current_gain_ = config_.release_alpha * target_gain + (1.0f - config_.release_alpha) * current_gain_;
    }

    state.probability = smoothed_prob_;
    state.detected = detected_latched_;
    state.attack_gain = target_gain;
    state.release_gain = current_gain_;

    return state;
}

void ImpulseDetector::reset() {
    std::fill(prev_magnitude_.begin(), prev_magnitude_.end(), 0.0f);
    smoothed_prob_ = 0.0f;
    detected_latched_ = false;
    current_gain_ = 1.0f;
}

} // namespace impulse
} // namespace noiselessx
