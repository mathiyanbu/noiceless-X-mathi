import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.signal import fftconvolve

@dataclass
class MixtureResult:
    noisy: np.ndarray        # x[n] = (s * h_s)[n] + alpha * (v * h_v)[n] + eta[n]
    clean: np.ndarray        # (s * h_s)[n]
    noise: np.ndarray        # alpha * (v * h_v)[n]
    target_snr_db: float     # Realized or requested SNR in dB
    noise_floor_db: float    # Sensor noise level in dBFS

class AudioMixer:
    """
    Synthesizes realistic dual-source mixtures for speech enhancement training:
      x[n] = (s * h_s)[n] + alpha(n) * (v * h_v)[n] + eta[n]
    Where:
      - s: clean speech signal
      - v: ambient/acoustic noise signal
      - h_s: optional speech Room Impulse Response (RIR)
      - h_v: optional noise Room Impulse Response (RIR)
      - alpha(n): static or time-varying gain scaling noise for specified SNR
      - eta[n]: sensor electrical noise floor
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        snr_range_db: Tuple[float, float] = (-5.0, 15.0),
        noise_floor_db: float = -60.0,
        enable_time_varying_gain: bool = False
    ):
        self.sample_rate = sample_rate
        self.snr_min, self.snr_max = snr_range_db
        self.noise_floor_db = noise_floor_db
        self.enable_time_varying_gain = enable_time_varying_gain

    def mix(
        self,
        clean_speech: np.ndarray,
        noise: np.ndarray,
        target_snr_db: Optional[float] = None,
        rir_speech: Optional[np.ndarray] = None,
        rir_noise: Optional[np.ndarray] = None
    ) -> MixtureResult:
        """
        Create a mixed noisy-speech example.
        """
        assert len(clean_speech) > 0, "Clean speech array cannot be empty"
        assert len(noise) > 0, "Noise array cannot be empty"

        # 1. RIR convolution (s * h_s) and (v * h_v)
        if rir_speech is not None and len(rir_speech) > 0:
            s_rev = fftconvolve(clean_speech, rir_speech, mode="full")[:len(clean_speech)]
        else:
            s_rev = clean_speech.copy()

        if rir_noise is not None and len(rir_noise) > 0:
            v_rev = fftconvolve(noise, rir_noise, mode="full")[:len(noise)]
        else:
            v_rev = noise.copy()

        # 2. Length matching (loop or crop noise to match speech length)
        target_len = len(s_rev)
        if len(v_rev) < target_len:
            repeat_count = int(np.ceil(target_len / len(v_rev)))
            v_rev = np.tile(v_rev, repeat_count)[:target_len]
        elif len(v_rev) > target_len:
            # Pick a random segment from the noise file
            start = random.randint(0, len(v_rev) - target_len)
            v_rev = v_rev[start : start + target_len]

        # 3. Energy and SNR calculation
        p_speech = np.mean(s_rev**2) + 1e-12
        p_noise = np.mean(v_rev**2) + 1e-12

        if target_snr_db is None:
            target_snr_db = random.uniform(self.snr_min, self.snr_max)

        # alpha = sqrt( P_s / (P_v * 10^(SNR/10)) )
        alpha_base = np.sqrt(p_speech / (p_noise * (10.0 ** (target_snr_db / 10.0))))

        if self.enable_time_varying_gain:
            # Subtle low-frequency modulation +/- 1.5 dB across the clip
            mod_freq = random.uniform(0.2, 0.8) # Hz
            t = np.arange(target_len) / self.sample_rate
            mod = 1.0 + 0.15 * np.sin(2.0 * np.pi * mod_freq * t)
            alpha_t = alpha_base * mod
        else:
            alpha_t = alpha_base

        scaled_noise = alpha_t * v_rev

        # 4. Sensor noise floor eta[n]
        noise_floor_power = 10.0 ** (self.noise_floor_db / 10.0)
        sensor_noise = np.random.normal(0.0, np.sqrt(noise_floor_power), target_len)

        # 5. Composite mixture
        noisy = s_rev + scaled_noise + sensor_noise

        return MixtureResult(
            noisy=noisy.astype(np.float32),
            clean=s_rev.astype(np.float32),
            noise=scaled_noise.astype(np.float32),
            target_snr_db=float(target_snr_db),
            noise_floor_db=float(self.noise_floor_db)
        )


class DatasetSplitter:
    """
    Partitions audio files into strictly Speaker-Disjoint and Noise-Source-Disjoint
    train, validation, and test splits.
    """
    @staticmethod
    def partition_disjoint(
        items: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        rng = random.Random(seed)
        shuffled = list(items)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_split = shuffled[:n_train]
        val_split = shuffled[n_train : n_train + n_val]
        test_split = shuffled[n_train + n_val :]

        # Verify disjointness
        assert set(train_split).isdisjoint(set(val_split))
        assert set(train_split).isdisjoint(set(test_split))
        assert set(val_split).isdisjoint(set(test_split))

        return train_split, val_split, test_split

    @staticmethod
    def create_split_manifest(
        speaker_ids: List[str],
        noise_source_ids: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, Dict[str, List[str]]]:
        train_spk, val_spk, test_spk = DatasetSplitter.partition_disjoint(
            list(set(speaker_ids)), train_ratio, val_ratio, seed
        )
        train_noi, val_noi, test_noi = DatasetSplitter.partition_disjoint(
            list(set(noise_source_ids)), train_ratio, val_ratio, seed + 100
        )

        return {
            "train": {"speakers": train_spk, "noises": train_noi},
            "val": {"speakers": val_spk, "noises": val_noi},
            "test": {"speakers": test_spk, "noises": test_noi},
        }
