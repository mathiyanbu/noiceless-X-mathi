"""
SIH26052 — NOICELESSX: Dynamic Dual-Source PyTorch Audio Dataset & On-The-Fly Mixer.
Provides on-the-fly mixing of clean speech, 200+ acoustic noise classes, room reverberation (RIR),
dynamic SNR assignment [-5 dB to +15 dB], and real/imaginary STFT feature extraction for ComplexCRN.
"""

import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
        # Fast FFT convolution
        reverberant = sp_signal.fftconvolve(signal, rir, mode="full")
        # Direct sound alignment: find peak of RIR
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
        Mixes clean speech and noise at the specified SNR (dB).
        Returns:
            noisy_mix:    x[n]
            clean_target: s[n] (target signal for enhancement)
            noise_target: v[n]
        """
        # Ensure identical lengths
        target_len = len(clean)
        if len(noise) < target_len:
            # Repeat noise if too short
            repeats = int(math.ceil(target_len / len(noise)))
            noise = np.tile(noise, repeats)[:target_len]
        elif len(noise) > target_len:
            # Random crop from longer noise
            start = random.randint(0, len(noise) - target_len)
            noise = noise[start : start + target_len]

        # Apply reverberation if RIRs are supplied
        clean_processed = self.convolve_rir(clean, clean_rir) if clean_rir is not None else clean
        noise_processed = self.convolve_rir(noise, noise_rir) if noise_rir is not None else noise

        # Compute RMS
        rms_clean = self.compute_rms(clean_processed)
        rms_noise = self.compute_rms(noise_processed)

        # Desired noise RMS: RMS_n = RMS_s / 10^(SNR/20)
        target_rms_noise = rms_clean / (10.0 ** (snr_db / 20.0))
        scale = target_rms_noise / (rms_noise + self.eps)
        noise_scaled = noise_processed * scale

        # Additive acoustic superposition
        mix = clean_processed + noise_scaled

        # Prevent digital clipping with 0.95 peak headroom
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
    applies room reverberation (RIR), and extracts STFT features (Real, Imag).
    """

    def __init__(
        self,
        clean_manifest_path: str,
        noise_manifest_path: str,
        rir_manifest_path: Optional[str] = None,
        sample_rate: int = 16000,
        segment_duration: float = 2.0,  # 32,000 samples for 2s chunk
        snr_range: Tuple[float, float] = (-5.0, 15.0),
        rir_prob: float = 0.5,
        fft_size: int = 512,
        hop_size: int = 80,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.clean_records = load_manifest(clean_manifest_path)
        self.noise_records = load_manifest(noise_manifest_path)
        self.rir_records = load_manifest(rir_manifest_path) if rir_manifest_path and os.path.exists(rir_manifest_path) else []

        if not self.clean_records:
            raise ValueError(f"Clean speech manifest is empty: {clean_manifest_path}")
        if not self.noise_records:
            raise ValueError(f"Noise manifest is empty: {noise_manifest_path}")

        self.sr = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        self.snr_min, self.snr_max = snr_range
        self.rir_prob = rir_prob
        self.fft_size = fft_size
        self.hop_size = hop_size

        self.mixer = AudioMixer(target_sample_rate=sample_rate)
        self.window = torch.hann_window(fft_size)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __len__(self) -> int:
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

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, str]]:
        clean_rec = self.clean_records[idx]
        noise_rec = random.choice(self.noise_records)

        # Load clean speech
        clean_audio, _ = validate_and_load_audio(clean_rec.file_path, target_sr=self.sr)
        # Load noise
        noise_audio, _ = validate_and_load_audio(noise_rec.file_path, target_sr=self.sr)

        # Fixed-length segmentation (truncate or zero-pad)
        if len(clean_audio) >= self.segment_samples:
            max_start = len(clean_audio) - self.segment_samples
            start = random.randint(0, max_start)
            clean_audio = clean_audio[start : start + self.segment_samples]
        else:
            pad_len = self.segment_samples - len(clean_audio)
            clean_audio = np.pad(clean_audio, (0, pad_len), mode="constant")

        # Select RIR if enabled and available
        clean_rir, noise_rir = None, None
        if self.rir_records and random.random() < self.rir_prob:
            rir_rec = random.choice(self.rir_records)
            clean_rir, _ = validate_and_load_audio(rir_rec.file_path, target_sr=self.sr)

        # Randomized target SNR in [-5 dB, +15 dB]
        target_snr = random.uniform(self.snr_min, self.snr_max)

        # Mix signals
        noisy_audio, target_speech, noise_signal = self.mixer.mix(
            clean=clean_audio,
            noise=noise_audio,
            snr_db=target_snr,
            clean_rir=clean_rir,
            noise_rir=noise_rir,
        )

        # Convert to torch tensors
        noisy_t = torch.from_numpy(noisy_audio).float()
        target_t = torch.from_numpy(target_speech).float()

        # Extract complex STFT
        noisy_stft = self._extract_stft(noisy_t)
        clean_stft = self._extract_stft(target_t)

        return {
            "noisy_stft": noisy_stft,        # (2, T, 257)
            "clean_stft": clean_stft,        # (2, T, 257)
            "noisy_wav": noisy_t,            # (segment_samples,)
            "clean_wav": target_t,            # (segment_samples,)
            "target_snr": float(target_snr),
            "clean_speaker": clean_rec.speaker_id or "unknown",
            "noise_class": noise_rec.class_name,
        }


def collate_speech_batch(batch: List[Dict[str, Union[torch.Tensor, float, str]]]) -> Dict[str, torch.Tensor]:
    """Collates a batch of audio samples into batched PyTorch tensors."""
    noisy_stfts = torch.stack([item["noisy_stft"] for item in batch], dim=0)
    clean_stfts = torch.stack([item["clean_stft"] for item in batch], dim=0)
    noisy_wavs = torch.stack([item["noisy_wav"] for item in batch], dim=0)
    clean_wavs = torch.stack([item["clean_wav"] for item in batch], dim=0)
    snrs = torch.tensor([item["target_snr"] for item in batch], dtype=torch.float32)

    return {
        "noisy_stft": noisy_stfts,  # (B, 2, T, 257)
        "clean_stft": clean_stfts,  # (B, 2, T, 257)
        "noisy_wav": noisy_wavs,    # (B, samples)
        "clean_wav": clean_wavs,    # (B, samples)
        "snr": snrs,
    }
