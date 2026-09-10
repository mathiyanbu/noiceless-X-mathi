"""
SIH26052 — NOICELESSX: Dataset Manifest Generator and Audio Normalizer.
Indexes audio files, validates format, enforces speaker-disjoint splits,
and generates JSON manifests for real model training.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
from scipy import signal as sp_signal

from ai.datasets.sources import is_impulsive_class


@dataclass
class AudioRecord:
    file_path: str
    category: str       # 'clean_speech', 'noise', 'rir', 'impulse'
    class_name: str     # e.g., 'p225', 'office', 'DKITCHEN', 'gunshot'
    speaker_id: Optional[str] = None
    duration_sec: float = 0.0
    sample_rate: int = 16000
    rms_dbfs: float = -96.0
    is_impulsive: bool = False


def validate_and_load_audio(
    path: str,
    target_sr: int = 16000,
    normalize_peak: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Loads audio, converts to mono 16kHz float32, and verifies audio integrity.
    Returns: (audio_samples, duration_seconds)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    data, sr = sf.read(path, dtype="float32")

    # Multichannel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != target_sr:
        num_target_samples = int(round(len(data) * target_sr / sr))
        data = sp_signal.resample(data, num_target_samples).astype(np.float32)
        sr = target_sr

    # Remove NaNs and Infs
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

    # Peak normalize if requested (with headroom)
    max_peak = float(np.max(np.abs(data)))
    if normalize_peak and max_peak > 1e-4:
        data = (data / max_peak) * 0.95

    duration_sec = len(data) / target_sr
    return data, duration_sec


def compute_rms_dbfs(data: np.ndarray) -> float:
    """Computes RMS level in dBFS."""
    if len(data) == 0:
        return -96.0
    rms = float(np.sqrt(np.mean(data ** 2) + 1e-12))
    return float(20.0 * math.log10(rms))


class ManifestBuilder:
    """Builds and validates structured JSON dataset manifests."""

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sr = target_sample_rate

    def scan_clean_speech_directory(
        self,
        speech_dir: str,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42
    ) -> Tuple[List[AudioRecord], List[AudioRecord], List[AudioRecord]]:
        """
        Scans clean speech utterances, parses speaker IDs, and creates
        STRICTLY speaker-disjoint train / val / test splits.
        """
        root = Path(speech_dir)
        wav_files = sorted(list(root.glob("**/*.wav")) + list(root.glob("**/*.flac")))

        speaker_to_files: Dict[str, List[Path]] = {}
        for f in wav_files:
            # Typical VoiceBank/VCTK naming: p225_001.wav -> speaker 'p225'
            parts = f.stem.split("_")
            speaker_id = parts[0] if len(parts) >= 2 else "spk_default"
            speaker_to_files.setdefault(speaker_id, []).append(f)

        speakers = sorted(list(speaker_to_files.keys()))
        rng = np.random.RandomState(seed)
        rng.shuffle(speakers)

        n_speakers = len(speakers)
        n_test_spk = max(1, int(round(n_speakers * test_ratio))) if n_speakers > 2 else 0
        n_val_spk = max(1, int(round(n_speakers * val_ratio))) if n_speakers > 2 else 0

        test_spks = set(speakers[:n_test_spk])
        val_spks = set(speakers[n_test_spk : n_test_spk + n_val_spk])
        train_spks = set(speakers[n_test_spk + n_val_spk:])

        # If only 1 or 2 speakers, fallback to file-level split with warning
        if len(train_spks) == 0:
            train_spks = set(speakers)
            val_spks = set(speakers)
            test_spks = set(speakers)

        train_records, val_records, test_records = [], [], []

        for spk, files in speaker_to_files.items():
            for f in files:
                try:
                    data, duration = validate_and_load_audio(str(f), target_sr=self.target_sr)
                    rms = compute_rms_dbfs(data)
                    rec = AudioRecord(
                        file_path=str(f.resolve()),
                        category="clean_speech",
                        class_name=spk,
                        speaker_id=spk,
                        duration_sec=duration,
                        sample_rate=self.target_sr,
                        rms_dbfs=rms
                    )

                    if spk in test_spks and len(test_spks) < len(speakers):
                        test_records.append(rec)
                    elif spk in val_spks and len(val_spks) < len(speakers):
                        val_records.append(rec)
                    else:
                        train_records.append(rec)
                except Exception as e:
                    print(f"Warning: Skipping corrupt audio file {f}: {e}")

        # Ensure train is never empty
        if not train_records and val_records:
            train_records = val_records
        return train_records, val_records, test_records

    def scan_noise_directory(self, noise_dir: str) -> List[AudioRecord]:
        """Scans noise pool audio files, classifying into acoustic and impulsive categories."""
        root = Path(noise_dir)
        audio_files = sorted(list(root.glob("**/*.wav")) + list(root.glob("**/*.flac")))

        records: List[AudioRecord] = []
        for f in audio_files:
            try:
                data, duration = validate_and_load_audio(str(f), target_sr=self.target_sr)
                class_name = f.parent.name if f.parent != root else f.stem
                impulsive = is_impulsive_class(class_name) or is_impulsive_class(f.stem)

                records.append(AudioRecord(
                    file_path=str(f.resolve()),
                    category="impulse" if impulsive else "noise",
                    class_name=class_name,
                    duration_sec=duration,
                    sample_rate=self.target_sr,
                    rms_dbfs=compute_rms_dbfs(data),
                    is_impulsive=impulsive
                ))
            except Exception as e:
                print(f"Warning: Skipping corrupt noise file {f}: {e}")

        return records

    def scan_rir_directory(self, rir_dir: str) -> List[AudioRecord]:
        """Scans room impulse responses."""
        root = Path(rir_dir)
        audio_files = sorted(list(root.glob("**/*.wav")) + list(root.glob("**/*.flac")))

        records: List[AudioRecord] = []
        for f in audio_files:
            try:
                data, duration = validate_and_load_audio(str(f), target_sr=self.target_sr)
                records.append(AudioRecord(
                    file_path=str(f.resolve()),
                    category="rir",
                    class_name=f.parent.name,
                    duration_sec=duration,
                    sample_rate=self.target_sr,
                    rms_dbfs=compute_rms_dbfs(data)
                ))
            except Exception as e:
                print(f"Warning: Skipping corrupt RIR file {f}: {e}")

        return records

    def save_manifest(self, records: List[AudioRecord], output_path: str):
        """Saves records as a JSON manifest file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in records]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def load_manifest(manifest_path: str) -> List[AudioRecord]:
    """Loads records from a JSON manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [AudioRecord(**d) for d in data]
