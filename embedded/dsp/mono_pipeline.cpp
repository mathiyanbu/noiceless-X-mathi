#include "mono_pipeline.hpp"
#include <algorithm>

namespace noiselessx {
namespace dsp {

MonoPipeline::MonoPipeline(
    float sample_rate,
    float hp_cutoff_hz,
    size_t fft_size,
    size_t hop_size
) : sample_rate_(sample_rate),
    fft_size_(fft_size),
    hop_size_(hop_size),
    dc_blocker_(sample_rate, 5.0f),
    high_pass_(sample_rate, hp_cutoff_hz, 0.70710678f),
    stft_(fft_size, hop_size),
    vad_(sample_rate) {
    
    preprocessed_hop_.resize(hop_size_);
    spectral_frame_.magnitude.resize(stft_.get_num_bins());
    spectral_frame_.phase.resize(stft_.get_num_bins());
    spectral_frame_.real.resize(stft_.get_num_bins());
    spectral_frame_.imag.resize(stft_.get_num_bins());
    spectral_frame_.num_bins = stft_.get_num_bins();
}

void MonoPipeline::reset() {
    dc_blocker_.reset();
    high_pass_.reset();
    stft_.reset();
    vad_.reset();
}

void MonoPipeline::process_hop(
    std::span<const float> hop_in,
    std::span<float> hop_out,
    VadDecision& vad_decision,
    const SpectralProcessorCallback& callback
) {
    const size_t count = std::min(hop_in.size(), hop_size_);

    // 1. Copy to preallocated buffer
    std::copy_n(hop_in.data(), count, preprocessed_hop_.data());

    // 2. DC offset removal (running mean subtraction)
    dc_blocker_.process(preprocessed_hop_.data(), count);

    // 3. High-pass filter (80 Hz biquad)
    high_pass_.process(preprocessed_hop_.data(), count);

    // 4. STFT Analysis
    stft_.process_analysis(
        std::span<const float>(preprocessed_hop_.data(), count),
        spectral_frame_
    );

    // 5. Voice Activity Detection
    vad_decision = vad_.process_frame(
        std::span<const float>(preprocessed_hop_.data(), count),
        std::span<const float>(spectral_frame_.magnitude.data(), spectral_frame_.num_bins)
    );

    // 6. Optional spectral modification callback (e.g. AI mask, NLMS suppression)
    if (callback) {
        callback(spectral_frame_, vad_decision);
    }

    // 7. iSTFT Synthesis (WOLA overlap-add)
    stft_.process_synthesis_complex(
        std::span<const float>(spectral_frame_.real.data(), spectral_frame_.num_bins),
        std::span<const float>(spectral_frame_.imag.data(), spectral_frame_.num_bins),
        hop_out
    );
}

} // namespace dsp
} // namespace noiselessx
