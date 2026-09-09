from pathlib import Path
import yaml
import pytest

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

def test_raspberrypi_yaml_schema():
    """Verify config/raspberrypi.yaml contains all required dual-mic fields."""
    config_file = CONFIG_DIR / "raspberrypi.yaml"
    assert config_file.exists(), f"Missing config file: {config_file}"

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "audio" in cfg, "Missing 'audio' section in config"
    audio = cfg["audio"]
    assert audio["sample_rate"] == 16000
    assert audio["channels"] == 1
    assert audio["frame_ms"] == 10
    assert audio["hop_ms"] == 5
    assert audio["fft_size"] == 512
    assert "input_device" in audio and audio["input_device"] != ""
    assert "reference_device" in audio and audio["reference_device"] != ""
    assert "output_device" in audio and audio["output_device"] != ""

    assert "nlms" in cfg, "Missing 'nlms' section in config"
    assert cfg["nlms"]["enabled"] is True, "NLMS dual-mic mode must be active by default"

def test_development_yaml_schema():
    """Verify config/development.yaml contains matching schema for host dev."""
    config_file = CONFIG_DIR / "development.yaml"
    assert config_file.exists(), f"Missing config file: {config_file}"

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "audio" in cfg
    assert "nlms" in cfg
    assert cfg["nlms"]["enabled"] is True
