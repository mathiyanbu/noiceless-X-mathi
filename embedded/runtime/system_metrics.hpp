#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <chrono>

namespace noiselessx {
namespace runtime {

struct CpuSnapshot {
    uint64_t user{0};
    uint64_t nice{0};
    uint64_t system{0};
    uint64_t idle{0};
    uint64_t iowait{0};
    uint64_t irq{0};
    uint64_t softirq{0};
    uint64_t steal{0};

    [[nodiscard]] uint64_t get_idle_time() const noexcept {
        return idle + iowait;
    }

    [[nodiscard]] uint64_t get_total_time() const noexcept {
        return user + nice + system + idle + iowait + irq + softirq + steal;
    }
};

/**
 * @brief System Telemetry Monitor for Raspberry Pi 4/5 (ARM64 Linux).
 * 
 * Accurately reads:
 * - Real CPU utilization per core from /proc/stat
 * - Real CPU thermal metrics from /sys/class/thermal/thermal_zone0/temp or vcgencmd measure_temp
 * - Never fabricates telemetry: reports genuine OS/kernel values or indicates host unavailability.
 */
class SystemMetrics {
public:
    SystemMetrics();
    ~SystemMetrics() = default;

    /**
     * @brief Sample /proc/stat and thermal zone to update CPU usage and temperatures.
     */
    void sample();

    [[nodiscard]] const std::vector<float>& get_per_core_cpu_percent() const noexcept {
        return per_core_cpu_percent_;
    }

    [[nodiscard]] float get_overall_cpu_percent() const noexcept {
        return overall_cpu_percent_;
    }

    [[nodiscard]] float get_cpu_temperature_c() const noexcept {
        return cpu_temperature_c_;
    }

    [[nodiscard]] bool is_temperature_available() const noexcept {
        return temp_available_;
    }

    [[nodiscard]] const std::string& get_temperature_source() const noexcept {
        return temp_source_;
    }

    [[nodiscard]] size_t get_num_cores() const noexcept {
        return per_core_cpu_percent_.size();
    }

private:
    void read_proc_stat();
    void read_cpu_temp();

    std::vector<CpuSnapshot> prev_core_snapshots_;
    CpuSnapshot prev_total_snapshot_;
    std::chrono::steady_clock::time_point last_sample_time_;

    std::vector<float> per_core_cpu_percent_;
    float overall_cpu_percent_{0.0f};
    float cpu_temperature_c_{0.0f};
    bool temp_available_{false};
    std::string temp_source_{"none"};
    bool initialized_{false};
};

} // namespace runtime
} // namespace noiselessx
