#include "device_enumerator.hpp"
#include <iomanip>
#include <algorithm>

#if NOICELESSX_HAS_ALSA
#include <alsa/asoundlib.h>
#endif

namespace noiselessx::audio {

std::vector<AudioDeviceInfo> DeviceEnumerator::enumerate_devices() {
    std::vector<AudioDeviceInfo> devices;

#if NOICELESSX_HAS_ALSA
    int card = -1;
    while (snd_card_next(&card) == 0 && card >= 0) {
        snd_ctl_t* ctl = nullptr;
        char card_str[32];
        std::snprintf(card_str, sizeof(card_str), "hw:%d", card);

        if (snd_ctl_open(&ctl, card_str, 0) < 0) {
            continue;
        }

        snd_ctl_card_info_t* cinfo = nullptr;
        snd_ctl_card_info_alloca(&cinfo);
        if (snd_ctl_card_info(ctl, cinfo) < 0) {
            snd_ctl_close(ctl);
            continue;
        }

        const char* raw_card_name = snd_ctl_card_info_get_name(cinfo);
        const char* raw_card_id = snd_ctl_card_info_get_id(cinfo);
        std::string card_name = raw_card_name ? raw_card_name : "Unknown Card";
        std::string card_id = raw_card_id ? raw_card_id : "";

        int dev = -1;
        while (snd_ctl_pcm_next_device(ctl, &dev) == 0 && dev >= 0) {
            snd_pcm_info_t* pinfo = nullptr;
            snd_pcm_info_alloca(&pinfo);
            snd_pcm_info_set_device(pinfo, dev);
            snd_pcm_info_set_subdevice(pinfo, 0);

            // Probe Capture capability
            snd_pcm_info_set_stream(pinfo, SND_PCM_STREAM_CAPTURE);
            if (snd_ctl_pcm_info(ctl, pinfo) >= 0) {
                AudioDeviceInfo info;
                info.card_index = card;
                info.device_index = dev;
                info.card_name = card_name;
                const char* dname = snd_pcm_info_get_name(pinfo);
                info.device_name = dname ? dname : "Capture Device";
                info.is_capture = true;
                info.is_playback = false;
                info.alsa_identifier = "hw:" + std::to_string(card) + "," + std::to_string(dev);
                devices.push_back(info);
            }

            // Probe Playback capability
            snd_pcm_info_set_stream(pinfo, SND_PCM_STREAM_PLAYBACK);
            if (snd_ctl_pcm_info(ctl, pinfo) >= 0) {
                AudioDeviceInfo info;
                info.card_index = card;
                info.device_index = dev;
                info.card_name = card_name;
                const char* dname = snd_pcm_info_get_name(pinfo);
                info.device_name = dname ? dname : "Playback Device";
                info.is_capture = false;
                info.is_playback = true;
                info.alsa_identifier = "hw:" + std::to_string(card) + "," + std::to_string(dev);
                devices.push_back(info);
            }
        }
        snd_ctl_close(ctl);
    }
#endif

    return devices;
}

void DeviceEnumerator::print_devices(const std::vector<AudioDeviceInfo>& devices, std::ostream& out) {
    out << "\n========================== ALSA Audio Endpoints ==========================\n";
    out << std::left 
        << std::setw(6)  << "Card"
        << std::setw(6)  << "Dev"
        << std::setw(10) << "Type"
        << std::setw(14) << "ALSA ID"
        << std::setw(24) << "Card Name"
        << "Device Name\n";
    out << "--------------------------------------------------------------------------\n";

    if (devices.empty()) {
        out << "  [No ALSA audio devices detected on this host / ALSA disabled]\n";
    } else {
        for (const auto& dev : devices) {
            std::string type_str = dev.is_capture ? "CAPTURE" : "PLAYBACK";
            out << std::left
                << std::setw(6)  << dev.card_index
                << std::setw(6)  << dev.device_index
                << std::setw(10) << type_str
                << std::setw(14) << dev.alsa_identifier
                << std::setw(24) << dev.card_name.substr(0, 22)
                << dev.device_name << "\n";
        }
    }
    out << "==========================================================================\n\n";
}

const AudioDeviceInfo* DeviceEnumerator::find_device_matching(
    const std::vector<AudioDeviceInfo>& devices,
    const std::string& pattern,
    bool is_capture
) {
    if (pattern.empty()) return nullptr;

    for (const auto& dev : devices) {
        if (dev.is_capture != is_capture) continue;

        // Exact match on ALSA identifier (e.g. "hw:0,0")
        if (dev.alsa_identifier == pattern) {
            return &dev;
        }

        // Substring match on card name or device name (case-insensitive)
        std::string pattern_lower = pattern;
        std::string card_lower = dev.card_name;
        std::string dev_lower = dev.device_name;
        std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
        std::transform(card_lower.begin(), card_lower.end(), card_lower.begin(), ::tolower);
        std::transform(dev_lower.begin(), dev_lower.end(), dev_lower.begin(), ::tolower);

        if (card_lower.find(pattern_lower) != std::string::npos ||
            dev_lower.find(pattern_lower) != std::string::npos) {
            return &dev;
        }
    }
    return nullptr;
}

} // namespace noiselessx::audio
