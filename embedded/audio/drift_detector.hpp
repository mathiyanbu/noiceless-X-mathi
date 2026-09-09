#pragma once

#include <cstdint>
#include <cmath>
#include <atomic>
#include <chrono>

namespace noiselessx::audio {

/**
 * @brief Inter-Stream Clock Drift Detector and Synchronizer.
 * 
 * Two independent USB audio interfaces operate on separate crystal oscillators,
 * inevitably drifting over time. This class compares cumulative frames and
 * arrival timestamps to quantify clock skew and alert when the drift exceeds
 * real-time DSP tolerance.
 */
class DriftDetector {
public:
    explicit DriftDetector(uint32_t sample_rate = 16000, double max_allowable_drift_ms = 10.0)
        : sample_rate_(sample_rate),
          max_allowable_drift_ms_(max_allowable_drift_ms) {}

    /**
     * @brief Record captured frames and timestamp from Primary stream.
     * @param frames Number of frames captured in this chunk
     * @param timestamp_ns Hardware/system timestamp in nanoseconds (0 to use steady_clock)
     */
    void record_primary(uint64_t frames, uint64_t timestamp_ns = 0) {
        if (timestamp_ns == 0) {
            timestamp_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count()
            );
        }
        primary_frames_.fetch_add(frames, std::memory_order_relaxed);
        primary_timestamp_ns_.store(timestamp_ns, std::memory_order_relaxed);
        update_metrics();
    }

    /**
     * @brief Record captured frames and timestamp from Reference stream.
     * @param frames Number of frames captured in this chunk
     * @param timestamp_ns Hardware/system timestamp in nanoseconds (0 to use steady_clock)
     */
    void record_reference(uint64_t frames, uint64_t timestamp_ns = 0) {
        if (timestamp_ns == 0) {
            timestamp_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count()
            );
        }
        reference_frames_.fetch_add(frames, std::memory_order_relaxed);
        reference_timestamp_ns_.store(timestamp_ns, std::memory_order_relaxed);
        update_metrics();
    }

    /**
     * @brief Difference in captured samples: Primary - Reference.
     */
    [[nodiscard]] int64_t get_drift_samples() const noexcept {
        const int64_t prim = static_cast<int64_t>(primary_frames_.load(std::memory_order_relaxed));
        const int64_t ref = static_cast<int64_t>(reference_frames_.load(std::memory_order_relaxed));
        return prim - ref;
    }

    /**
     * @brief Clock skew converted to milliseconds: (drift_samples / sample_rate) * 1000.
     */
    [[nodiscard]] double get_drift_ms() const noexcept {
        if (sample_rate_ == 0) return 0.0;
        return (static_cast<double>(get_drift_samples()) / static_cast<double>(sample_rate_)) * 1000.0;
    }

    /**
     * @brief Raw arrival timestamp difference in milliseconds: (Primary_ts - Ref_ts).
     */
    [[nodiscard]] double get_timestamp_drift_ms() const noexcept {
        const uint64_t prim_ns = primary_timestamp_ns_.load(std::memory_order_relaxed);
        const uint64_t ref_ns = reference_timestamp_ns_.load(std::memory_order_relaxed);
        if (prim_ns == 0 || ref_ns == 0) return 0.0;
        return static_cast<double>(static_cast<int64_t>(prim_ns) - static_cast<int64_t>(ref_ns)) / 1e6;
    }

    /**
     * @brief Returns true if inter-stream drift exceeds max allowable threshold.
     */
    [[nodiscard]] bool is_warning_active() const noexcept {
        return warning_active_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] double get_max_drift_ms() const noexcept {
        return max_allowable_drift_ms_;
    }

    void set_max_drift_ms(double ms) noexcept {
        max_allowable_drift_ms_ = ms;
        update_metrics();
    }

    [[nodiscard]] uint64_t get_primary_total_frames() const noexcept {
        return primary_frames_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t get_reference_total_frames() const noexcept {
        return reference_frames_.load(std::memory_order_relaxed);
    }

    void reset() noexcept {
        primary_frames_.store(0, std::memory_order_relaxed);
        reference_frames_.store(0, std::memory_order_relaxed);
        primary_timestamp_ns_.store(0, std::memory_order_relaxed);
        reference_timestamp_ns_.store(0, std::memory_order_relaxed);
        warning_active_.store(false, std::memory_order_relaxed);
    }

private:
    void update_metrics() noexcept {
        const double drift_ms = std::abs(get_drift_ms());
        warning_active_.store(drift_ms > max_allowable_drift_ms_, std::memory_order_relaxed);
    }

    uint32_t sample_rate_{16000};
    double max_allowable_drift_ms_{10.0};

    std::atomic<uint64_t> primary_frames_{0};
    std::atomic<uint64_t> reference_frames_{0};
    std::atomic<uint64_t> primary_timestamp_ns_{0};
    std::atomic<uint64_t> reference_timestamp_ns_{0};
    std::atomic<bool> warning_active_{false};
};

} // namespace noiselessx::audio
