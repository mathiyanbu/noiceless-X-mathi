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
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates balanced dataset of extracted 8-D features and labels (0 = steady, 1 = impulsive)
    split into train and test DataLoaders.
    """
    np.random.seed(seed)
    features_list = []
    labels_list = []

    window = np.hanning(fft_size)

    # Generate non-impulsive samples (class 0)
    prev_mag = np.zeros(fft_size // 2 + 1, dtype=np.float32)
    for _ in range(n_samples_per_class):
        audio = generate_synthetic_non_impulse_frame(fft_size)
        spec = np.fft.rfft(audio * window)
        mag = np.abs(spec).astype(np.float32)
        feats, prev_mag = extract_impulse_features(audio, mag, prev_mag)
        features_list.append(feats)
        labels_list.append(0.0)

    # Generate impulsive samples (class 1)
    prev_mag = np.zeros(fft_size // 2 + 1, dtype=np.float32)
    for _ in range(n_samples_per_class):
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
