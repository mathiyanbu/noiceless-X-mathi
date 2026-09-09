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


def test_config_parser_corrupt_yaml_raises():
    """Verify that malformed YAML syntax raises YAMLError with informative trace."""
    corrupted_yaml = """
    audio:
      sample_rate: 16000
      channels: [unclosed list
    nlms:
      enabled: true
    """
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(corrupted_yaml)


def test_config_parser_boundary_and_cola_constraints():
    """Verify acoustic and algorithmic boundary constraints across configuration."""
    for cfg_name in ["raspberrypi.yaml", "development.yaml"]:
        cfg_file = CONFIG_DIR / cfg_name
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        audio = cfg["audio"]
        assert audio["sample_rate"] in [16000, 48000], "Sample rate must be standard voice rate"
        assert audio["hop_ms"] <= audio["frame_ms"], "Hop duration must be <= frame duration"
        fft_size = audio["fft_size"]
        assert (fft_size & (fft_size - 1)) == 0, f"FFT size {fft_size} must be a power of 2"

        nlms = cfg["nlms"]
        assert int(nlms["filter_length"]) >= 32, "NLMS filter length must be at least 32 taps"
        assert 0.0 < float(nlms["learning_rate"]) <= 1.0, "NLMS learning rate mu must be in (0, 1]"
        assert float(nlms["epsilon"]) > 0.0, "NLMS regularization epsilon must be strictly positive"

        imp = cfg["impulse"]
        assert float(imp["threshold_on"]) > float(imp["threshold_off"]), "Hysteresis threshold_on must be > threshold_off"
        assert 0.0 < float(imp["attack_alpha"]) <= 1.0
        assert 0.0 < float(imp["release_alpha"]) <= 1.0
        assert float(imp["release_alpha"]) < float(imp["attack_alpha"]), "Fast attack, slow release required"


def test_config_parser_default_fallback_function():
    """Verify safe fallback behavior when optional fields are missing."""
    def parse_audio_config_with_defaults(raw_dict: dict) -> dict:
        audio = raw_dict.get("audio", {})
        return {
            "sample_rate": int(audio.get("sample_rate", 16000)),
            "channels": int(audio.get("channels", 1)),
            "frame_ms": int(audio.get("frame_ms", 10)),
            "hop_ms": int(audio.get("hop_ms", 5)),
            "fft_size": int(audio.get("fft_size", 512)),
            "input_device": str(audio.get("input_device", "default")),
            "reference_device": str(audio.get("reference_device", "default")),
            "output_device": str(audio.get("output_device", "default")),
            "nlms_enabled": bool(raw_dict.get("nlms", {}).get("enabled", True)),
            "ai_threads": int(raw_dict.get("ai", {}).get("intra_op_num_threads", 2)),
        }

    # Test empty dict
    defaults = parse_audio_config_with_defaults({})
    assert defaults["sample_rate"] == 16000
    assert defaults["frame_ms"] == 10
    assert defaults["hop_ms"] == 5
    assert defaults["fft_size"] == 512
    assert defaults["nlms_enabled"] is True
    assert defaults["ai_threads"] == 2

    # Test partial overrides
    partial = parse_audio_config_with_defaults({"audio": {"sample_rate": 48000, "hop_ms": 10}})
    assert partial["sample_rate"] == 48000
    assert partial["hop_ms"] == 10
    assert partial["fft_size"] == 512

