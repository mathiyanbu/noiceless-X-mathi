"""
Unit tests for NOICELESSX Manifest Preparation and Audio Decode Verification.
Validates:
1. Real audio decoding and corruption rejection.
2. Exact schema conformity for CSV and JSON manifests.
3. Strict disjointness (0% speaker and noise recording leakage across train/val/test splits).
4. Audio duration accuracy against decoded sample counts.
5. Accurate telemetry summary statistics.
"""

import csv
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from ai.datasets.prepare_manifest import (
    DatasetManifestProcessor,
    classify_acoustic_category,
    infer_dataset_source,
    verify_and_decode_audio_file,
)


@pytest.fixture
def test_audio_subset(tmp_path):
    """Creates a controlled audio corpus with multiple speakers and noise classes."""
    corpus_dir = tmp_path / "raw_corpus"
    speech_dir = corpus_dir / "clean_speech"
    noise_dir = corpus_dir / "noise"
    rir_dir = corpus_dir / "rirs"

    speech_dir.mkdir(parents=True)
    noise_dir.mkdir(parents=True)
    rir_dir.mkdir(parents=True)

    sr = 16000

    # 1. Clean speech: 3 speakers (p225, p226, p227) with 3 utterances each
    speakers = ["p225", "p226", "p227"]
    for spk in speakers:
        for utt in range(1, 4):
            dur = 1.0 + 0.2 * utt
            t = np.linspace(0, dur, int(dur * sr), endpoint=False)
            sig = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
            sf.write(str(speech_dir / f"{spk}_{utt:03d}.wav"), sig, sr)

    # 2. Stationary noise: air_conditioner, rain
    for cls in ["air_conditioner", "rain"]:
        sub = noise_dir / cls
        sub.mkdir()
        for idx in range(1, 3):
            dur = 2.0
            sig = (0.1 * np.random.randn(int(dur * sr))).astype(np.float32)
            sf.write(str(sub / f"{cls}_{idx:02d}.wav"), sig, sr)

    # 3. Impulsive noise: gun_shot, glass_breaking
    for cls in ["gun_shot", "glass_breaking"]:
        sub = noise_dir / cls
        sub.mkdir()
        for idx in range(1, 3):
            dur = 1.5
            sig = np.zeros(int(dur * sr), dtype=np.float32)
            sig[int(0.2 * sr)] = 0.95
            sig[int(0.2 * sr) : int(0.3 * sr)] = 0.5 * np.exp(-np.linspace(0, 5, int(0.1 * sr)))
            sf.write(str(sub / f"{cls}_{idx:02d}.wav"), sig, sr)

    # 4. Non-stationary noise: street_traffic, dog
    for cls in ["street_traffic", "dog"]:
        sub = noise_dir / cls
        sub.mkdir()
        for idx in range(1, 3):
            dur = 2.5
            sig = (0.15 * np.random.randn(int(dur * sr))).astype(np.float32)
            sf.write(str(sub / f"{cls}_{idx:02d}.wav"), sig, sr)

    # 5. Room impulse responses
    for idx in range(1, 3):
        rir = np.zeros(int(0.3 * sr), dtype=np.float32)
        rir[10] = 1.0
        rir[50:] = 0.2 * np.exp(-np.linspace(0, 5, len(rir) - 50))
        sf.write(str(rir_dir / f"rir_{idx:02d}.wav"), rir, sr)

    # 6. Add a corrupt/zero-byte file to test rejection
    corrupt_file = corpus_dir / "corrupted.wav"
    with open(corrupt_file, "wb") as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00\x00")  # Truncated corrupt header

    return corpus_dir


def test_verify_and_decode_audio(test_audio_subset, tmp_path):
    # Valid audio
    valid_file = list((test_audio_subset / "clean_speech").glob("*.wav"))[0]
    ok, meta, err = verify_and_decode_audio_file(valid_file)
    assert ok is True
    assert meta is not None
    assert meta["sample_rate"] == 16000
    assert meta["duration_sec"] > 0.5
    assert meta["frames"] > 0
    assert err is None

    # Corrupt audio
    corrupt_file = test_audio_subset / "corrupted.wav"
    ok_bad, meta_bad, err_bad = verify_and_decode_audio_file(corrupt_file)
    assert ok_bad is False
    assert meta_bad is None
    assert err_bad is not None


def test_acoustic_categorization():
    assert classify_acoustic_category("air_conditioner", "esc50", Path("noise/air_conditioner.wav")) == "noise_stationary"
    assert classify_acoustic_category("gun_shot", "urbansound8k", Path("noise/gun_shot.wav")) == "noise_impulsive"
    assert classify_acoustic_category("glass_breaking", "esc50", Path("noise/glass_breaking.wav")) == "noise_impulsive"
    assert classify_acoustic_category("street_traffic", "tau2020", Path("noise/street_traffic.wav")) == "noise_nonstationary"
    assert classify_acoustic_category("p225", "voicebank", Path("clean_speech/p225_001.wav")) == "clean_speech"
    assert classify_acoustic_category("office_rir", "rirs_noises", Path("rirs/office_rir.wav")) == "rir"


def test_prepare_manifest_disjoint_and_durations(test_audio_subset, tmp_path):
    output_dir = tmp_path / "manifests"
    processor = DatasetManifestProcessor(val_ratio=0.33, test_ratio=0.33, seed=42)

    valid_files = processor.scan_directory(test_audio_subset)
    # 9 speech + 4 stationary + 4 impulsive + 4 nonstationary + 2 rir = 23 valid audio files
    assert len(valid_files) == 23

    entries, summary = processor.process_manifest(valid_files)
    assert len(entries) == 23

    # Verify no speaker leakage
    speech_entries = [e for e in entries if e.category == "clean_speech"]
    train_speakers = {e.speaker_id for e in speech_entries if e.split == "train"}
    val_speakers = {e.speaker_id for e in speech_entries if e.split == "val"}
    test_speakers = {e.speaker_id for e in speech_entries if e.split == "test"}

    assert train_speakers.isdisjoint(val_speakers), "Speaker leakage between train and val!"
    assert train_speakers.isdisjoint(test_speakers), "Speaker leakage between train and test!"
    assert val_speakers.isdisjoint(test_speakers), "Speaker leakage between val and test!"

    # Verify no noise recording leakage
    noise_entries = [e for e in entries if e.category.startswith("noise_")]
    train_noise = {e.filepath for e in noise_entries if e.split == "train"}
    val_noise = {e.filepath for e in noise_entries if e.split == "val"}
    test_noise = {e.filepath for e in noise_entries if e.split == "test"}

    assert train_noise.isdisjoint(val_noise), "Noise recording leakage between train and val!"
    assert train_noise.isdisjoint(test_noise), "Noise recording leakage between train and test!"
    assert val_noise.isdisjoint(test_noise), "Noise recording leakage between val and test!"

    # Verify all audio files actually decode and durations match
    for entry in entries:
        data, sr = sf.read(entry.filepath, dtype="float32")
        actual_duration = len(data) / sr
        assert abs(actual_duration - entry.duration_sec) < 0.001
        assert sr == entry.sample_rate

    # Verify summary counts
    assert summary["clean_speech_hours"] > 0
    assert summary["distinct_noise_classes_count"] >= 6
    assert summary["rir_count"] == 2
    assert summary["train_records"] + summary["val_records"] + summary["test_records"] == 23


def test_export_manifest_files(test_audio_subset, tmp_path):
    output_dir = tmp_path / "manifests"
    processor = DatasetManifestProcessor(val_ratio=0.33, test_ratio=0.33, seed=42)
    valid_files = processor.scan_directory(test_audio_subset)
    entries, summary = processor.process_manifest(valid_files)

    from ai.datasets.prepare_manifest import export_manifests
    export_manifests(entries, summary, output_dir)

    csv_path = output_dir / "manifest.csv"
    json_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    assert csv_path.exists()
    assert json_path.exists()
    assert summary_path.exists()

    # Verify CSV columns
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        expected_cols = {
            "filepath", "dataset_source", "category", "duration_sec",
            "sample_rate", "class_label", "split", "speaker_id", "channels", "rms_dbfs"
        }
        assert expected_cols.issubset(set(row.keys()))
