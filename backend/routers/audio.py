"""
Audio devices and hardware routing endpoint.
"""

import os
import yaml
from typing import List
from fastapi import APIRouter
from backend.schemas import AudioDevicesResponse, AudioConfigInfo, AudioDeviceItem
from backend.config import settings
from backend.ipc_client import ipc_client, RuntimeUnavailableError

router = APIRouter(prefix="/api/audio", tags=["audio"])


def _load_audio_config() -> AudioConfigInfo:
    if os.path.exists(settings.config_yaml_path):
        try:
            with open(settings.config_yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            audio = data.get("audio", {})
            return AudioConfigInfo(
                sample_rate=audio.get("sample_rate", 16000),
                channels=audio.get("channels", 1),
                hardware_sample_rate=audio.get("hardware_sample_rate", audio.get("sample_rate", 16000)),
                hardware_channels=audio.get("hardware_channels", audio.get("channels", 1)),
                frame_ms=audio.get("frame_ms", 10),
                hop_ms=audio.get("hop_ms", 5),
                primary_device=audio.get("input_device", "hw:CARD=Headset,DEV=0"),
                reference_device=audio.get("reference_device", "hw:CARD=ErrorMic,DEV=0"),
                output_device=audio.get("output_device", "hw:CARD=Headset,DEV=0"),
                single_mic=audio.get("single_mic", False),
                nlms_enabled=data.get("nlms", {}).get("enabled", True),
                period_size=audio.get("buffer_frames", 160),
                buffer_size=audio.get("buffer_frames", 160) * audio.get("periods", 4)
            )
        except Exception:
            pass
    return AudioConfigInfo()


def _enumerate_system_devices() -> List[AudioDeviceItem]:
    devices = []
    asound_cards = "/proc/asound/cards"
    if os.path.exists(asound_cards):
        try:
            with open(asound_cards, "r") as f:
                for line in f:
                    if "[" in line and "]" in line:
                        parts = line.strip().split()
                        card_idx = int(parts[0])
                        name = line.split("[")[1].split("]")[0].strip()
                        devices.append(AudioDeviceItem(
                            card_index=card_idx,
                            device_index=0,
                            device_type="Duplex",
                            identifier=f"hw:{card_idx},0",
                            name=name
                        ))
            return devices
        except Exception:
            pass

    # Default configured devices for demonstration
    devices.append(AudioDeviceItem(
        card_index=0,
        device_index=0,
        device_type="Duplex",
        identifier="hw:CARD=Headset,DEV=0",
        name="USB Headphone / Headset Audio"
    ))
    devices.append(AudioDeviceItem(
        card_index=1,
        device_index=0,
        device_type="Capture",
        identifier="hw:CARD=ErrorMic,DEV=0",
        name="Reference Error Microphone"
    ))
    return devices


@router.get("/devices", response_model=AudioDevicesResponse)
async def get_audio_devices():
    """
    Returns the configured audio parameters and enumerated ALSA hardware devices.
    """
    try:
        res = ipc_client.send_command("get_devices")
        if res.get("status") == "ok":
            cfg_dict = res.get("config", {})
            dev_list = res.get("devices", [])
            return AudioDevicesResponse(
                config=AudioConfigInfo(**cfg_dict) if cfg_dict else _load_audio_config(),
                devices=[AudioDeviceItem(**d) for d in dev_list] if dev_list else _enumerate_system_devices()
            )
    except RuntimeUnavailableError:
        pass

    return AudioDevicesResponse(
        config=_load_audio_config(),
        devices=_enumerate_system_devices()
    )
