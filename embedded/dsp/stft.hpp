#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <complex>

struct PFFFT_Setup;

namespace noiselessx {
namespace dsp {

/**
 * @brief Complex spectral frame structure (257 frequency bins for N=512).
 */
struct SpectralFrame {
    std::vector<float> magnitude; // |X(k)| for k in [0, N/2]
    std::vector<float> phase;     // arg(X(k)) for k in [0, N/2]
    std::vector<float> real;      // Re{X(k)} for k in [0, N/2]
    std::vector<float> imag;      // Im{X(k)} for k in [0, N/2]
    size_t num_bins{0};
};

/**
 * @brief Short-Time Fourier Transform (STFT) and Inverse STFT (iSTFT) Engine.
 * 
 * Implements:
 * - Analysis: X(k, m) = sum_{n=0}^{N-1} x[n + mH] * w[n] * exp(-j 2*pi*k*n / N)
 * - Synthesis: y[n + mH] += (1/norm[n]) * Re{ IFFT(S_hat(., m)) }[n] * w[n]
 * - Accelerated with PFFFT (SIMD ARM NEON / SSE / fallback)
 * - Persistent overlap-add state buffer across calls
 * - Weighted Overlap-Add (WOLA) exact normalization achieving < -80 dB reconstruction error
 * - Zero heap allocations in per-frame processing methods
 */
class StftEngine {
public:
    explicit StftEngine(size_t fft_size = 512, size_t hop_size = 80);
    ~StftEngine();

    // Non-copyable, movable
    StftEngine(const StftEngine&) = delete;
    StftEngine& operator=(const StftEngine&) = delete;
    StftEngine(StftEngine&& other) noexcept;
    StftEngine& operator=(StftEngine&& other) noexcept;

    /**
     * @brief Initialize or reconfigure STFT dimensions.
     */
    void initialize(size_t fft_size, size_t hop_size);

    /**
     * @brief Perform STFT analysis on a hop of new input samples.
     * @param hop_in New time-domain samples of length hop_size
     * @param out_frame Preallocated SpectralFrame populated with magnitude, phase, real, imag
     */
    void process_analysis(std::span<const float> hop_in, SpectralFrame& out_frame);

    /**
     * @brief Perform iSTFT synthesis with overlap-add from modified spectral magnitude and phase.
     * @param magnitude Magnitude spectrum of length num_bins (N/2 + 1)
     * @param phase Phase spectrum of length num_bins (N/2 + 1)
     * @param hop_out Output buffer of length hop_size populated with synthesized audio
     */
    void process_synthesis(
        std::span<const float> magnitude,
        std::span<const float> phase,
        std::span<float> hop_out
    );

    /**
     * @brief Perform iSTFT synthesis directly from real and imaginary spectral bins.
     */
    void process_synthesis_complex(
        std::span<const float> real,
        std::span<const float> imag,
        std::span<float> hop_out
    );

    /**
     * @brief Reset persistent overlap-add and history buffers to zero.
     */
    void reset();

    [[nodiscard]] size_t get_fft_size() const noexcept { return fft_size_; }
    [[nodiscard]] size_t get_hop_size() const noexcept { return hop_size_; }
    [[nodiscard]] size_t get_num_bins() const noexcept { return num_bins_; }

private:
    void cleanup();
    void compute_window();

    size_t fft_size_{512};
    size_t hop_size_{80};
    size_t num_bins_{257};

    PFFFT_Setup* pffft_setup_{nullptr};

    // Aligned buffers for PFFFT SIMD execution
    float* window_{nullptr};
    float* fft_in_{nullptr};
    float* fft_out_{nullptr};
    float* fft_work_{nullptr};

    // Persistent state buffers
    std::vector<float> history_buffer_;      // Size N
    std::vector<float> accum_audio_buffer_;  // Size 2*N for overlap-add
    std::vector<float> accum_weight_buffer_; // Size 2*N for WOLA normalization
};

} // namespace dsp
} // namespace noiselessx
