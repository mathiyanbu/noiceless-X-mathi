#pragma once

#include "audio_types.hpp"
#include <vector>
#include <string>
#include <iostream>

namespace noiselessx::audio {

class DeviceEnumerator {
public:
    /**
     * @brief Query ALSA directly to find all physical and logical audio endpoints.
     * @return List of discovered AudioDeviceInfo records.
     */
    static std::vector<AudioDeviceInfo> enumerate_devices();

    /**
     * @brief Print enumerated audio devices to stdout in a clean, human-readable table.
     */
    static void print_devices(const std::vector<AudioDeviceInfo>& devices, std::ostream& out = std::cout);

    /**
     * @brief Attempt to resolve a configuration string (e.g. "hw:CARD=Headset,DEV=0" or "Headset")
     *        to a specific AudioDeviceInfo.
     */
    static const AudioDeviceInfo* find_device_matching(
        const std::vector<AudioDeviceInfo>& devices,
        const std::string& pattern,
        bool is_capture
    );
};

} // namespace noiselessx::audio
