#include "system_metrics.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace noiselessx::runtime {

SystemMetrics::SystemMetrics() {
    last_sample_time_ = std::chrono::steady_clock::now();
    sample();
}

void SystemMetrics::sample() {
    read_proc_stat();
    read_cpu_temp();
    last_sample_time_ = std::chrono::steady_clock::now();
}

void SystemMetrics::read_proc_stat() {
    std::ifstream stat_file("/proc/stat");
    if (!stat_file.is_open()) {
        // Not running on Linux (e.g. development host)
        if (per_core_cpu_percent_.empty()) {
            per_core_cpu_percent_ = {0.0f, 0.0f, 0.0f, 0.0f}; // Placeholder quad-core on host
        }
        return;
    }

    std::string line;
    std::vector<CpuSnapshot> current_cores;
    CpuSnapshot current_total;
    bool found_total = false;

    while (std::getline(stat_file, line)) {
        if (line.rfind("cpu", 0) != 0) {
            // Lines after CPU entries can be skipped
            continue;
        }

        std::istringstream ss(line);
        std::string cpu_label;
        ss >> cpu_label;

        CpuSnapshot snap;
        ss >> snap.user >> snap.nice >> snap.system >> snap.idle
           >> snap.iowait >> snap.irq >> snap.softirq >> snap.steal;

        if (cpu_label == "cpu") {
            current_total = snap;
            found_total = true;
        } else if (cpu_label.rfind("cpu", 0) == 0 && cpu_label.length() > 3) {
            // Per-core line, e.g. cpu0, cpu1...
            current_cores.push_back(snap);
        }
    }

    if (!initialized_) {
        prev_total_snapshot_ = current_total;
        prev_core_snapshots_ = current_cores;
        per_core_cpu_percent_.assign(current_cores.size(), 0.0f);
        overall_cpu_percent_ = 0.0f;
        initialized_ = true;
        return;
    }

    // Compute overall CPU utilization %
    if (found_total) {
        const uint64_t total_delta = current_total.get_total_time() - prev_total_snapshot_.get_total_time();
        const uint64_t idle_delta = current_total.get_idle_time() - prev_total_snapshot_.get_idle_time();
        if (total_delta > 0) {
            const float active_delta = static_cast<float>(total_delta - idle_delta);
            overall_cpu_percent_ = std::clamp((active_delta / static_cast<float>(total_delta)) * 100.0f, 0.0f, 100.0f);
        }
        prev_total_snapshot_ = current_total;
    }

    // Compute per-core CPU utilization %
    if (current_cores.size() == prev_core_snapshots_.size()) {
        per_core_cpu_percent_.resize(current_cores.size());
        for (size_t i = 0; i < current_cores.size(); ++i) {
            const uint64_t total_delta = current_cores[i].get_total_time() - prev_core_snapshots_[i].get_total_time();
            const uint64_t idle_delta = current_cores[i].get_idle_time() - prev_core_snapshots_[i].get_idle_time();
            if (total_delta > 0) {
                const float active_delta = static_cast<float>(total_delta - idle_delta);
                per_core_cpu_percent_[i] = std::clamp((active_delta / static_cast<float>(total_delta)) * 100.0f, 0.0f, 100.0f);
            }
        }
        prev_core_snapshots_ = current_cores;
    } else {
        prev_core_snapshots_ = current_cores;
        per_core_cpu_percent_.assign(current_cores.size(), 0.0f);
    }
}

void SystemMetrics::read_cpu_temp() {
    // 1. Primary path: /sys/class/thermal/thermal_zone0/temp (Linux sysfs standard)
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
    if (temp_file.is_open()) {
        int64_t raw_temp_milli = 0;
        if (temp_file >> raw_temp_milli) {
            cpu_temperature_c_ = static_cast<float>(raw_temp_milli) / 1000.0f;
            temp_available_ = true;
            temp_source_ = "/sys/class/thermal";
            return;
        }
    }

    // 2. Secondary path: vcgencmd measure_temp (Raspberry Pi VideoCore tool)
#if defined(__linux__)
    FILE* pipe = popen("vcgencmd measure_temp 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            // Typical format: temp=48.2'C
            char* eq = strchr(buffer, '=');
            if (eq) {
                float temp = 0.0f;
                if (sscanf(eq + 1, "%f", &temp) == 1) {
                    cpu_temperature_c_ = temp;
                    temp_available_ = true;
                    temp_source_ = "vcgencmd";
                    pclose(pipe);
                    return;
                }
            }
        }
        pclose(pipe);
    }
#endif

    // Host dev environment or unsupported platform
    cpu_temperature_c_ = 0.0f;
    temp_available_ = false;
    temp_source_ = "N/A";
}

} // namespace noiselessx::runtime
