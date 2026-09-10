"""
Unit tests for NOICELESSX Dataset Pipeline, Manifest Generator, and Audio Mixer.
Verifies format validation, 16kHz mono conversion, speaker-disjoint splits,
dynamic SNR mixing, and STFT feature extraction.
"""

import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from ai.datasets.manifest import (
    AudioRecord,
    ManifestBuilder,
    compute_rms_dbfs,
    load_manifest,
    validate_and_load_audio,
)
from ai.datasets.sources import (
    DEMAND_CLASSES,
    ESC50_CLASSES,
    IMPULSIVE_CLASSES,
    get_all_registered_noise_classes,
    is_impulsive_class,
)
from ai.training.dataset import AudioMixer, SpeechEnhancementDataset, collate_speech_batch


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Creates a temporary directory with test clean speech, noise, and RIR files."""
    speech_dir = tmp_path / "clean_speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rirs"
    speech_dir.mkdir()
    noise_dir.mkdir()
    rir_dir.mkdir()

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)

    # Clean speech for 3 speakers (p225, p226, p227)
    for spk in ["p225", "p226", "p227"]:
        for i in range(1, 4):
            wav = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            sf.write(str(speech_dir / f"{spk}_{i:03d}.wav"), wav, sr)

    # Noise classes (one steady, one impulsive)
    (noise_dir / "office").mkdir()
    (noise_dir / "gun_shot").mkdir()
    sf.write(str(noise_dir / "office" / "office_01.wav"), 0.1 * np.random.randn(sr).astype(np.float32), sr)
    sf.write(str(noise_dir / "gun_shot" / "gun_shot_01.wav"), 0.9 * np.random.randn(sr).astype(np.float32), sr)

    # RIR
    rir = np.zeros(int(0.2 * sr), dtype=np.float32)
    rir[10] = 1.0
    rir[50:] = 0.1 * np.exp(-np.linspace(0, 5, len(rir) - 50))
    sf.write(str(rir_dir / "test_rir.wav"), rir, sr)

    return {"speech": speech_dir, "noise": noise_dir, "rir": rir_dir}


def test_sources_registry_and_ontology():
    classes = get_all_registered_noise_classes()
    assert len(classes) >= 50
    assert "DKITCHEN" in DEMAND_CLASSES
    assert "glass_breaking" in ESC50_CLASSES
    assert is_impulsive_class("gun_shot") is True
    assert is_impulsive_class("glass_breaking") is True
    assert is_impulsive_class("door_wood_knock") is True
    assert is_impulsive_class("office_ambient") is False


def test_validate_and_load_audio(tmp_path):
    sr_orig = 44100
    t = np.linspace(0, 1.0, sr_orig, endpoint=False)
    stereo_data = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1).astype(np.float32)
    fpath = tmp_path / "stereo_44k.wav"
    sf.write(str(fpath), stereo_data, sr_orig)

    mono_16k, duration = validate_and_load_audio(str(fpath), target_sr=16000, normalize_peak=True)
    assert mono_16k.ndim == 1
    assert len(mono_16k) == 16000
    assert abs(duration - 1.0) < 0.05
    assert np.max(np.abs(mono_16k)) <= 0.96


def test_compute_rms_dbfs():
    silence = np.zeros(16000, dtype=np.float32)
    rms_silence = compute_rms_dbfs(silence)
    assert rms_silence < -90.0

    full_sine = np.sin(np.linspace(0, 100 * np.pi, 16000)).astype(np.float32)
    rms_sine = compute_rms_dbfs(full_sine)
    # Sine RMS is 1/sqrt(2) = ~ -3.01 dBFS
    assert abs(rms_sine - (-3.01)) < 0.2


def test_manifest_builder_speaker_disjoint(temp_audio_dir, tmp_path):
    builder = ManifestBuilder(target_sample_rate=16000)
    train_recs, val_recs, test_recs = builder.scan_clean_speech_directory(
        str(temp_audio_dir["speech"]), val_ratio=0.33, test_ratio=0.33, seed=42
    )

    assert len(train_recs) > 0
    assert len(val_recs) > 0
    assert len(test_recs) > 0

    train_spks = {r.speaker_id for r in train_recs}
    val_spks = {r.speaker_id for r in val_recs}
    test_spks = {r.speaker_id for r in test_recs}

    # Strict speaker disjointness
    assert train_spks.isdisjoint(val_spks)
    assert train_spks.isdisjoint(test_spks)
    assert val_spks.isdisjoint(test_spks)

    # Save and reload
    out_json = tmp_path / "test_manifest.json"
    builder.save_manifest(train_recs, str(out_json))
    reloaded = load_manifest(str(out_json))
    assert len(reloaded) == len(train_recs)
    assert reloaded[0].speaker_id == train_recs[0].speaker_id


def test_audio_mixer():
    mixer = AudioMixer(target_sample_rate=16000)
    clean = np.sin(np.linspace(0, 200 * np.pi, 16000)).astype(np.float32)
    noise = np.random.randn(16000).astype(np.float32)

    # Mix at 0 dB SNR -> RMS clean ≈ RMS noise
    mix_0db, clean_0db, noise_0db = mixer.mix(clean, noise, snr_db=0.0)
    rms_c = mixer.compute_rms(clean_0db)
    rms_n = mixer.compute_rms(noise_0db)
    snr_actual = 20.0 * np.log10(rms_c / rms_n)
    assert abs(snr_actual) < 0.5
    assert np.max(np.abs(mix_0db)) <= 0.95

    # Mix at +10 dB SNR -> RMS clean ≈ 3.16 * RMS noise
    mix_10db, clean_10db, noise_10db = mixer.mix(clean, noise, snr_db=10.0)
    rms_c10 = mixer.compute_rms(clean_10db)
    rms_n10 = mixer.compute_rms(noise_10db)
    snr_10_actual = 20.0 * np.log10(rms_c10 / rms_n10)
    assert abs(snr_10_actual - 10.0) < 0.5


def test_speech_enhancement_dataset(temp_audio_dir, tmp_path):
    builder = ManifestBuilder(target_sample_rate=16000)
    train_recs, val_recs, _ = builder.scan_clean_speech_directory(str(temp_audio_dir["speech"]), seed=42)
    noise_recs = builder.scan_noise_directory(str(temp_audio_dir["noise"]))
    rir_recs = builder.scan_rir_directory(str(temp_audio_dir["rir"]))

    train_m = tmp_path / "train.json"
    noise_m = tmp_path / "noise.json"
    rir_m = tmp_path / "rir.json"

    builder.save_manifest(train_recs, str(train_m))
    builder.save_manifest(noise_recs, str(noise_m))
    builder.save_manifest(rir_recs, str(rir_m))

    dataset = SpeechEnhancementDataset(
        clean_manifest_path=str(train_m),
        noise_manifest_path=str(noise_m),
        rir_manifest_path=str(rir_m),
        segment_duration=1.0,  # 16000 samples
        fft_size=512,
        hop_size=80,
    )

    assert len(dataset) == len(train_recs)
    sample = dataset[0]

    assert "noisy_stft" in sample
    assert "clean_stft" in sample
    assert sample["noisy_stft"].shape[0] == 2  # Real, Imag
    assert sample["noisy_stft"].shape[-1] == 257  # Freq bins

    # Collate batch
    batch = collate_speech_batch([dataset[0], dataset[1]])
    assert batch["noisy_stft"].shape[0] == 2
    assert batch["clean_stft"].shape[0] == 2
    assert batch["noisy_wav"].shape[0] == 2
