"""
SIH26052 — NOICELESSX: Dynamic Dual-Source PyTorch Audio Dataset & On-The-Fly Mixer.
Provides on-the-fly mixing of clean speech, 200+ acoustic noise classes, room reverberation (RIR),
impulsive event injection, sensor noise floor, and real/imaginary STFT feature extraction for ComplexCRN.
"""

import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal as sp_signal
from torch.utils.data import Dataset

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.datasets.manifest import AudioRecord, load_manifest, validate_and_load_audio
from ai.preprocessing.mixer import (
    AudioMixer as PhysicalAudioMixer,
    ManifestAudioMixer,
    MixtureResult,
)


class AudioMixer:
    """
    On-the-fly acoustic mixer implementing the physical dual-microphone model:
      x[n] = s[n] * h_s[n] + v[n] * h_v[n]
    Calculates root-mean-square (RMS) energy and scales noise to achieve exact target SNR.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        eps: float = 1e-12,
    ):
        self.sr = target_sample_rate
        self.eps = eps

    def compute_rms(self, audio: np.ndarray) -> float:
        """Computes root-mean-square amplitude."""
        if len(audio) == 0:
            return self.eps
        return float(np.sqrt(np.mean(audio ** 2) + self.eps))

    def convolve_rir(self, signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """Applies room impulse response convolution, preserving original signal duration."""
        reverberant = sp_signal.fftconvolve(signal, rir, mode="full")
        peak_idx = int(np.argmax(np.abs(rir)))
        start_idx = max(0, peak_idx)
        aligned = reverberant[start_idx : start_idx + len(signal)]
        if len(aligned) < len(signal):
            aligned = np.pad(aligned, (0, len(signal) - len(aligned)))
        return aligned.astype(np.float32)

    def mix(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
        snr_db: float,
        clean_rir: Optional[np.ndarray] = None,
        noise_rir: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mixes clean speech and noise at specified target SNR (dB).
        Returns:
            noisy_mix:    x[n]
            clean_target: s[n]
            noise_target: v[n]
        """
        target_len = len(clean)
        if len(noise) < target_len:
            repeats = int(math.ceil(target_len / len(noise)))
            noise = np.tile(noise, repeats)[:target_len]
        elif len(noise) > target_len:
            start = random.randint(0, len(noise) - target_len)
            noise = noise[start : start + target_len]

        clean_processed = self.convolve_rir(clean, clean_rir) if clean_rir is not None else clean
        noise_processed = self.convolve_rir(noise, noise_rir) if noise_rir is not None else noise

        rms_clean = self.compute_rms(clean_processed)
        rms_noise = self.compute_rms(noise_processed)

        target_rms_noise = rms_clean / (10.0 ** (snr_db / 20.0))
        scale = target_rms_noise / (rms_noise + self.eps)
        noise_scaled = noise_processed * scale

        mix = clean_processed + noise_scaled

        max_val = np.max(np.abs(mix))
        if max_val > 0.95:
            norm_factor = 0.949 / max_val
            mix = mix * norm_factor
            clean_processed = clean_processed * norm_factor
            noise_scaled = noise_scaled * norm_factor

        return mix.astype(np.float32), clean_processed.astype(np.float32), noise_scaled.astype(np.float32)


class SpeechEnhancementDataset(Dataset):
    """
    PyTorch Dataset for ComplexCRN Training with dynamic on-the-fly mixing.
    Loads clean speech from disjoint speaker manifests, samples from 200+ noise classes,
    applies room reverberation (RIR), injects impulsive transients, simulates sensor noise floor,
    and extracts STFT features (Real, Imag).
    
    Supports:
    1. Unified manifest mode: manifest_path='data/manifests/manifest.csv', split='train'
    2. Legacy split manifest mode: clean_manifest_path, noise_manifest_path, etc.
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        split: str = "train",
        clean_manifest_path: Optional[str] = None,
        noise_manifest_path: Optional[str] = None,
        rir_manifest_path: Optional[str] = None,
        sample_rate: int = 16000,
        segment_duration: float = 2.0,  # 32,000 samples for 2s chunk
        snr_range: Tuple[float, float] = (-5.0, 15.0),
        rir_prob: float = 0.40,
        impulse_prob: float = 0.15,
        clipping_prob: float = 0.10,
        sensor_noise_floor_dbfs: float = -60.0,
        fft_size: int = 512,
        hop_size: int = 80,
        seed: Optional[int] = None,
        epoch_length: Optional[int] = None,
    ):
        super().__init__()
        self.sr = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.epoch_length = epoch_length
        self.window = torch.hann_window(fft_size)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.manifest_mixer: Optional[ManifestAudioMixer] = None
        self.legacy_mode = False

        if manifest_path is not None and os.path.exists(manifest_path):
            physical_mixer = PhysicalAudioMixer(
                sample_rate=sample_rate,
                snr_range_db=snr_range,
                rir_prob=rir_prob,
                impulse_prob=impulse_prob,
                clipping_prob=clipping_prob,
                sensor_noise_floor_dbfs=sensor_noise_floor_dbfs,
            )
            self.manifest_mixer = ManifestAudioMixer(
                manifest_path=manifest_path,
                split=split,
                sample_rate=sample_rate,
                segment_duration_sec=segment_duration,
                mixer=physical_mixer,
                seed=seed if seed is not None else 42,
            )
        elif clean_manifest_path is not None and noise_manifest_path is not None:
            self.legacy_mode = True
            self.clean_records = load_manifest(clean_manifest_path)
            self.noise_records = load_manifest(noise_manifest_path)
            self.rir_records = (
                load_manifest(rir_manifest_path)
                if rir_manifest_path and os.path.exists(rir_manifest_path)
                else []
            )
            if not self.clean_records:
                raise ValueError(f"Clean speech manifest is empty: {clean_manifest_path}")
            if not self.noise_records:
                raise ValueError(f"Noise manifest is empty: {noise_manifest_path}")

            self.snr_min, self.snr_max = snr_range
            self.rir_prob = rir_prob
            self.mixer = AudioMixer(target_sample_rate=sample_rate)
        else:
            raise ValueError("Must provide either 'manifest_path' or both 'clean_manifest_path' and 'noise_manifest_path'.")

    def __len__(self) -> int:
        if self.epoch_length is not None:
            return self.epoch_length
        if self.manifest_mixer is not None:
            return max(1, len(self.manifest_mixer.clean_records))
        return len(self.clean_records)

    def _extract_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extracts complex STFT with 2 channels: (Real, Imag)
        Input: (T_samples,)
        Output: (2, time_steps, num_bins=257)
        """
        stft_cplx = torch.stft(
            waveform,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size,
            window=self.window,
            center=True,
            return_complex=True,
        )
        # stft_cplx shape: (F=257, T)
        # Permute to (T, F):
        real = stft_cplx.real.T
        imag = stft_cplx.imag.T
        # Stack as channels: (2, T, F)
        return torch.stack([real, imag], dim=0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.manifest_mixer is not None:
            mix_result = self.manifest_mixer.generate_mixture()
            noisy_t = torch.from_numpy(mix_result.noisy).float()
            clean_t = torch.from_numpy(mix_result.clean).float()

            noisy_stft = self._extract_stft(noisy_t)
            clean_stft = self._extract_stft(clean_t)

            return {
                "noisy_stft": noisy_stft,
                "clean_stft": clean_stft,
                "noisy_wav": noisy_t,
                "clean_wav": clean_t,
                "target_snr": float(mix_result.target_snr_db),
                "measured_snr": float(mix_result.measured_snr_db),
                "clean_speaker": mix_result.metadata.get("speaker_id", "unknown"),
                "noise_class": mix_result.metadata.get("noise_class", "unknown"),
                "has_reverb": bool(mix_result.has_reverb),
                "has_impulse": bool(mix_result.has_impulse),
                "is_clipped": bool(mix_result.is_clipped),
            }

        # Legacy branch
        clean_rec = self.clean_records[idx % len(self.clean_records)]
        noise_rec = random.choice(self.noise_records)

        clean_audio, _ = validate_and_load_audio(clean_rec.file_path, target_sr=self.sr)
        noise_audio, _ = validate_and_load_audio(noise_rec.file_path, target_sr=self.sr)

        if len(clean_audio) >= self.segment_samples:
            max_start = len(clean_audio) - self.segment_samples
            start = random.randint(0, max_start)
            clean_audio = clean_audio[start : start + self.segment_samples]
        else:
            pad_len = self.segment_samples - len(clean_audio)
            clean_audio = np.pad(clean_audio, (0, pad_len), mode="constant")

        clean_rir = None
        if self.rir_records and random.random() < self.rir_prob:
            rir_rec = random.choice(self.rir_records)
            clean_rir, _ = validate_and_load_audio(rir_rec.file_path, target_sr=self.sr)

        target_snr = random.uniform(self.snr_min, self.snr_max)
        noisy_audio, target_speech, noise_signal = self.mixer.mix(
            clean=clean_audio,
            noise=noise_audio,
            snr_db=target_snr,
            clean_rir=clean_rir,
        )

        noisy_t = torch.from_numpy(noisy_audio).float()
        clean_t = torch.from_numpy(target_speech).float()

        noisy_stft = self._extract_stft(noisy_t)
        clean_stft = self._extract_stft(clean_t)

        return {
            "noisy_stft": noisy_stft,
            "clean_stft": clean_stft,
            "noisy_wav": noisy_t,
            "clean_wav": clean_t,
            "target_snr": float(target_snr),
            "measured_snr": float(target_snr),
            "clean_speaker": clean_rec.speaker_id or "unknown",
            "noise_class": noise_rec.class_name,
            "has_reverb": bool(clean_rir is not None),
            "has_impulse": False,
            "is_clipped": False,
        }


ManifestMixtureDataset = SpeechEnhancementDataset


def collate_speech_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collates a batch of audio samples into batched PyTorch tensors."""
    noisy_stfts = torch.stack([item["noisy_stft"] for item in batch], dim=0)
    clean_stfts = torch.stack([item["clean_stft"] for item in batch], dim=0)
    noisy_wavs = torch.stack([item["noisy_wav"] for item in batch], dim=0)
    clean_wavs = torch.stack([item["clean_wav"] for item in batch], dim=0)
    snrs = torch.tensor([item.get("target_snr", 0.0) for item in batch], dtype=torch.float32)

    return {
        "noisy_stft": noisy_stfts,  # (B, 2, T, 257)
        "clean_stft": clean_stfts,  # (B, 2, T, 257)
        "noisy_wav": noisy_wavs,    # (B, samples)
        "clean_wav": clean_wavs,    # (B, samples)
        "snr": snrs,
    }
