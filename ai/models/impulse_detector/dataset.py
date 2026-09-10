import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

from .features import extract_impulse_features

def generate_synthetic_impulse_frame(n_samples: int = 512, sr: int = 16000) -> np.ndarray:
    """Generates an impulsive frame (spike, click, clap transient, or keyboard strike)."""
    t = np.arange(n_samples) / sr
    frame = np.zeros(n_samples, dtype=np.float32)

    impulse_type = np.random.choice(["click", "tap", "burst", "decay_sine"])
    pos = np.random.randint(10, n_samples - 50)

    if impulse_type == "click":
        # Narrow delta-like spike (1-3 samples)
        spike_amp = np.random.uniform(0.6, 1.0)
        frame[pos] = spike_amp
        if pos + 1 < n_samples:
            frame[pos + 1] = -0.5 * spike_amp
    elif impulse_type == "tap":
        # Fast damped burst (keyboard tap, transient clap)
        decay = np.exp(-t[:100] / np.random.uniform(0.001, 0.005))
        freq = np.random.uniform(1000.0, 5000.0)
        wave = np.sin(2 * np.pi * freq * t[:100]) * decay
        end_idx = min(n_samples, pos + len(wave))
        frame[pos:end_idx] = wave[:end_idx - pos] * np.random.uniform(0.5, 1.0)
    elif impulse_type == "burst":
        # Dense high-frequency burst
        dur = np.random.randint(5, 30)
        end_idx = min(n_samples, pos + dur)
        frame[pos:end_idx] = np.random.uniform(-0.8, 0.8, end_idx - pos)
    else:
        # Rapid decaying oscillation
        freq = np.random.uniform(2000.0, 6000.0)
        tau = np.random.uniform(0.002, 0.008)
        burst_len = min(150, n_samples - pos)
        frame[pos:pos + burst_len] = np.sin(2 * np.pi * freq * t[:burst_len]) * np.exp(-t[:burst_len] / tau)

    # Add low level background ambience
    frame += 0.01 * np.random.randn(n_samples).astype(np.float32)
    return np.clip(frame, -1.0, 1.0)


def generate_synthetic_non_impulse_frame(n_samples: int = 512, sr: int = 16000) -> np.ndarray:
    """Generates a steady-state non-impulsive frame (stationary speech vowel, noise, silence)."""
    t = np.arange(n_samples) / sr
    case = np.random.choice(["vowel", "noise", "silence", "multitone"])

    if case == "vowel":
        f0 = np.random.uniform(100.0, 300.0)
        sig = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2 * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    elif case == "multitone":
        f1, f2 = np.random.uniform(300.0, 1000.0), np.random.uniform(1200.0, 3000.0)
        sig = 0.3 * np.sin(2 * np.pi * f1 * t) + 0.25 * np.sin(2 * np.pi * f2 * t)
    elif case == "noise":
        sig = np.random.uniform(-0.15, 0.15, n_samples)
    else: # silence
        sig = 0.005 * np.random.randn(n_samples)

    return np.clip(sig.astype(np.float32), -1.0, 1.0)


def create_impulse_dataset(
    n_samples_per_class: int = 1000,
    fft_size: int = 512,
    seed: int = 42,
    impulse_manifest_path: str = "data/manifests/impulse.json",
    noise_manifest_path: str = "data/manifests/noise.json",
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates balanced dataset of extracted 8-D features and labels (0 = steady, 1 = impulsive)
    split into train and test DataLoaders.
    Supports loading real recorded audio files from manifests if available,
    supplemented by synthetic generators for full acoustic coverage.
    """
    import os
    import json
    import soundfile as sf
    from scipy import signal as sp_signal

    np.random.seed(seed)
    features_list = []
    labels_list = []
    window = np.hanning(fft_size)

    # 1. Attempt loading real audio from manifests if available
    real_impulse_count = 0
    real_steady_count = 0

    if os.path.exists(impulse_manifest_path) and os.path.exists(noise_manifest_path):
        try:
            with open(impulse_manifest_path, "r", encoding="utf-8") as f:
                impulse_records = json.load(f)
            with open(noise_manifest_path, "r", encoding="utf-8") as f:
                noise_records = json.load(f)

            # Extract from real impulsive files
            prev_mag = np.zeros(fft_size // 2 + 1, dtype=np.float32)
            for rec in impulse_records:
                fpath = rec.get("file_path", "")
                if os.path.exists(fpath):
                    audio, sr = sf.read(fpath, dtype="float32")
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)
                    if sr != 16000:
                        audio = sp_signal.resample(audio, int(round(len(audio) * 16000 / sr))).astype(np.float32)

                    # Slice into 512 frames with 50% overlap
                    for st in range(0, len(audio) - fft_size, fft_size // 2):
                        frame = audio[st : st + fft_size]
                        spec = np.fft.rfft(frame * window)
                        mag = np.abs(spec).astype(np.float32)
                        feats, prev_mag = extract_impulse_features(frame, mag, prev_mag)
                        features_list.append(feats)
                        labels_list.append(1.0)
                        real_impulse_count += 1
                        if real_impulse_count >= n_samples_per_class:
                            break
                if real_impulse_count >= n_samples_per_class:
                    break

            # Extract from real steady noise files
            prev_mag = np.zeros(fft_size // 2 + 1, dtype=np.float32)
            for rec in noise_records:
                fpath = rec.get("file_path", "")
                if os.path.exists(fpath):
                    audio, sr = sf.read(fpath, dtype="float32")
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)
                    if sr != 16000:
                        audio = sp_signal.resample(audio, int(round(len(audio) * 16000 / sr))).astype(np.float32)

                    for st in range(0, len(audio) - fft_size, fft_size // 2):
                        frame = audio[st : st + fft_size]
                        spec = np.fft.rfft(frame * window)
                        mag = np.abs(spec).astype(np.float32)
                        feats, prev_mag = extract_impulse_features(frame, mag, prev_mag)
                        features_list.append(feats)
                        labels_list.append(0.0)
                        real_steady_count += 1
                        if real_steady_count >= n_samples_per_class:
                            break
                if real_steady_count >= n_samples_per_class:
                    break
        except Exception as e:
            print(f"Notice: Could not load real impulse audio: {e}. Using synthetic fallback.")

    # 2. Supplement remaining samples with synthetic generators
    remaining_steady = max(0, n_samples_per_class - real_steady_count)
    prev_mag = np.zeros(fft_size // 2 + 1, dtype=np.float32)
    for _ in range(remaining_steady):
        audio = generate_synthetic_non_impulse_frame(fft_size)
        spec = np.fft.rfft(audio * window)
        mag = np.abs(spec).astype(np.float32)
        feats, prev_mag = extract_impulse_features(audio, mag, prev_mag)
        features_list.append(feats)
        labels_list.append(0.0)

    remaining_impulse = max(0, n_samples_per_class - real_impulse_count)
    prev_mag = np.zeros(fft_size // 2 + 1, dtype=np.float32)
    for _ in range(remaining_impulse):
        audio = generate_synthetic_impulse_frame(fft_size)
        spec = np.fft.rfft(audio * window)
        mag = np.abs(spec).astype(np.float32)
        feats, prev_mag = extract_impulse_features(audio, mag, prev_mag)
        features_list.append(feats)
        labels_list.append(1.0)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32).reshape(-1, 1)

    # Shuffle dataset
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # 80/20 train/val split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    return train_loader, val_loader

