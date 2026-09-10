"""
Unit tests for NOICELESSX Audio Mixer, Mathematical Formulations, and Split Discipline.
Validates:
1. Exact round-trip SNR calculation matching target SNR (< 0.001 dB error).
2. Strict split enforcement (zero cross-contamination between train/val/test).
3. RIR convolution direct-path alignment and overflow prevention.
4. Impulsive event injection and clipping simulation.
"""

import csv
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from ai.preprocessing.mixer import AudioMixer, ManifestAudioMixer


@pytest.fixture
def mock_manifest_corpus(tmp_path):
    """Creates a temporary audio corpus and manifest covering train, val, test splits."""
    corpus = tmp_path / "audio_corpus"
    speech = corpus / "speech"
    noise = corpus / "noise"
    impulse = corpus / "impulse"
    rirs = corpus / "rirs"

    for d in [speech, noise, impulse, rirs]:
        d.mkdir(parents=True)

    sr = 16000
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)

    records = []

    # Clean speech: p225 (train), p226 (val), p227 (test)
    split_map = {"p225": "train", "p226": "val", "p227": "test"}
    for spk, split in split_map.items():
        for i in range(1, 3):
            fpath = speech / f"{spk}_{i:02d}.wav"
            sig = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
            sf.write(str(fpath), sig, sr)
            records.append({
                "filepath": str(fpath.resolve()),
                "dataset_source": "voicebank",
                "category": "clean_speech",
                "duration_sec": 2.0,
                "sample_rate": sr,
                "class_label": spk,
                "split": split,
                "speaker_id": spk,
                "channels": 1,
                "rms_dbfs": -12.0,
            })

    # Noise: train, val, test splits
    for split in ["train", "val", "test"]:
        for i in range(1, 3):
            fpath = noise / f"noise_{split}_{i:02d}.wav"
            sig = (0.2 * np.random.randn(int(2.0 * sr))).astype(np.float32)
            sf.write(str(fpath), sig, sr)
            records.append({
                "filepath": str(fpath.resolve()),
                "dataset_source": "demand",
                "category": "noise_stationary",
                "duration_sec": 2.0,
                "sample_rate": sr,
                "class_label": "office",
                "split": split,
                "speaker_id": None,
                "channels": 1,
                "rms_dbfs": -18.0,
            })

    # Impulses
    for i in range(1, 3):
        fpath = impulse / f"gunshot_{i:02d}.wav"
        sig = np.zeros(int(0.5 * sr), dtype=np.float32)
        sig[100] = 0.95
        sig[100:300] = 0.5 * np.exp(-np.linspace(0, 5, 200))
        sf.write(str(fpath), sig, sr)
        records.append({
            "filepath": str(fpath.resolve()),
            "dataset_source": "esc50",
            "category": "noise_impulsive",
            "duration_sec": 0.5,
            "sample_rate": sr,
            "class_label": "gun_shot",
            "split": "train",
            "speaker_id": None,
            "channels": 1,
            "rms_dbfs": -10.0,
        })

    # RIRs
    fpath = rirs / "test_rir.wav"
    rir = np.zeros(int(0.2 * sr), dtype=np.float32)
    rir[10] = 1.0
    rir[50:] = 0.2 * np.exp(-np.linspace(0, 5, len(rir) - 50))
    sf.write(str(fpath), rir, sr)
    records.append({
        "filepath": str(fpath.resolve()),
        "dataset_source": "rirs_noises",
        "category": "rir",
        "duration_sec": 0.2,
        "sample_rate": sr,
        "class_label": "room_medium",
        "split": "train",
        "speaker_id": None,
        "channels": 1,
        "rms_dbfs": -20.0,
    })

    manifest_csv = tmp_path / "test_manifest.csv"
    fieldnames = list(records[0].keys())
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    return manifest_csv


def test_round_trip_snr_exact_formula():
    mixer = AudioMixer(sample_rate=16000)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (0.2 * np.random.randn(sr)).astype(np.float32)

    # Test across multiple target SNRs
    for target_snr in [-5.0, 0.0, 5.0, 10.0, 15.0]:
        res = mixer.mix(
            clean_speech=speech,
            noise=noise,
            target_snr_db=target_snr,
            force_reverb=False,
            force_impulse=False,
            force_clipping=False,
        )
        assert abs(res.measured_snr_db - target_snr) < 1e-4
        assert res.has_reverb is False
        assert res.has_impulse is False
        assert res.is_clipped is False


def test_strict_split_disjointness(mock_manifest_corpus):
    train_mixer = ManifestAudioMixer(manifest_path=str(mock_manifest_corpus), split="train", seed=42)
    val_mixer = ManifestAudioMixer(manifest_path=str(mock_manifest_corpus), split="val", seed=42)
    test_mixer = ManifestAudioMixer(manifest_path=str(mock_manifest_corpus), split="test", seed=42)

    # Train split must only use p225 speech and train noise
    for _ in range(10):
        mix = train_mixer.generate_mixture()
        assert mix.metadata["speaker_id"] == "p225"
        assert "train" in mix.metadata["noise_file"]

    # Val split must only use p226 speech and val noise
    for _ in range(10):
        mix = val_mixer.generate_mixture()
        assert mix.metadata["speaker_id"] == "p226"
        assert "val" in mix.metadata["noise_file"]

    # Test split must only use p227 speech and test noise
    for _ in range(10):
        mix = test_mixer.generate_mixture()
        assert mix.metadata["speaker_id"] == "p227"
        assert "test" in mix.metadata["noise_file"]


def test_rir_convolution_safety():
    mixer = AudioMixer(sample_rate=16000)
    sig = np.random.randn(16000).astype(np.float32)
    high_gain_rir = np.ones(3200, dtype=np.float32) * 5.0  # Excessive gain RIR

    proc = mixer.convolve_rir(sig, high_gain_rir)
    assert len(proc) == len(sig)
    assert np.all(np.isfinite(proc))
    # RMS power should remain normalized to input power
    rms_in = mixer.compute_rms(sig)
    rms_out = mixer.compute_rms(proc)
    assert abs(rms_in - rms_out) < 1e-3


def test_impulsive_injection_and_clipping():
    mixer = AudioMixer(sample_rate=16000)
    speech = 0.5 * np.ones(16000, dtype=np.float32)
    noise = 0.1 * np.ones(16000, dtype=np.float32)
    impulse = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)

    # Force impulse
    res_imp = mixer.mix(
        clean_speech=speech,
        noise=noise,
        target_snr_db=5.0,
        impulse_audio=impulse,
        force_impulse=True,
    )
    assert res_imp.has_impulse is True
    assert np.any(res_imp.impulse != 0.0)

    # Force clipping
    res_clip = mixer.mix(
        clean_speech=speech,
        noise=noise,
        target_snr_db=5.0,
        force_clipping=True,
    )
    assert res_clip.is_clipped is True
    assert np.max(np.abs(res_clip.noisy)) <= 1.0
