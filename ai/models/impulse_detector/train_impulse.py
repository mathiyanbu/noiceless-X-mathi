#!/usr/bin/env python3
"""
SIH26052 — NOICELESSX: Impulse Detector Training on Real Acoustic Manifest Features.
Extracts Phase 8's 8-D physical feature vectors (RMS, Spectral Flux, Crest Factor, ZCR,
and 4 Subband Energies) directly from real impulsive events (gunshot, glass breaking, door slam,
clapping, fireworks) vs real stationary noise and speech from the dataset manifest.
Trains TinyImpulseMLP, evaluates F1-score, and exports to ONNX.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.models.impulse_detector.features import extract_impulse_features
from ai.models.impulse_detector.model import TinyImpulseMLP, export_impulse_model_to_onnx


def extract_features_from_audio_file(
    filepath: Path,
    frame_size: int = 512,
    hop_size: int = 80,
    window: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Decodes audio file and computes the 8-D feature vector for each hop.
    Features: [RMS, Spectral Flux, Crest Factor, ZCR, Band 0, Band 1, Band 2, Band 3]
    """
    if window is None:
        window = np.hanning(frame_size).astype(np.float32)

    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]

    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy.signal import resample
        mono = resample(mono, int(round(len(mono) * 16000 / sr))).astype(np.float32)

    features = []
    prev_mag = np.zeros(frame_size // 2 + 1, dtype=np.float32)

    for start in range(0, len(mono) - frame_size, hop_size):
        frame = mono[start : start + frame_size]
        spec = np.fft.rfft(frame * window)
        mag = np.abs(spec).astype(np.float32)

        feat, prev_mag = extract_impulse_features(frame, mag, prev_mag)
        features.append(feat)

    return features


def build_real_feature_dataset(
    manifest_path: Path,
    max_samples_per_class: int = 2000,
    split: str = "train",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads real audio files from manifest, extracting 8-D feature vectors:
    - Positive Class (1.0): real audio from 'noise_impulsive' (gunshot, glass break, door slam, clap, etc.)
    - Negative Class (0.0): real audio from 'clean_speech' and 'noise_stationary' / 'noise_nonstationary'
    """
    rng = np.random.RandomState(seed)

    impulsive_files: List[Path] = []
    non_impulsive_files: List[Path] = []

    # Parse manifest CSV or JSON
    if manifest_path.suffix.lower() == ".csv":
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if split != "all" and r.get("split", "train").lower() != split.lower():
                    continue
                fpath = Path(r["filepath"])
                if fpath.exists():
                    if r.get("category") == "noise_impulsive":
                        impulsive_files.append(fpath)
                    else:
                        non_impulsive_files.append(fpath)
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f)
            for r in records:
                if split != "all" and r.get("split", "train").lower() != split.lower():
                    continue
                fpath = Path(r["filepath"])
                if fpath.exists():
                    if r.get("category") == "noise_impulsive":
                        impulsive_files.append(fpath)
                    else:
                        non_impulsive_files.append(fpath)

    print(f"Loaded manifest: {len(impulsive_files)} impulsive audio files, {len(non_impulsive_files)} non-impulsive audio files.")

    # 1. Extract Impulsive Features (Class 1)
    pos_features: List[np.ndarray] = []
    rng.shuffle(impulsive_files)
    for f in impulsive_files:
        feats = extract_features_from_audio_file(f)
        pos_features.extend(feats)
        if len(pos_features) >= max_samples_per_class:
            break

    pos_features = pos_features[:max_samples_per_class]

    # 2. Extract Non-Impulsive Features (Class 0)
    neg_features: List[np.ndarray] = []
    rng.shuffle(non_impulsive_files)
    for f in non_impulsive_files:
        feats = extract_features_from_audio_file(f)
        neg_features.extend(feats)
        if len(neg_features) >= max_samples_per_class:
            break

    neg_features = neg_features[:max_samples_per_class]

    print(f"Extracted {len(pos_features)} real impulsive feature vectors (Class 1).")
    print(f"Extracted {len(neg_features)} real non-impulsive feature vectors (Class 0).")

    X = np.vstack([pos_features, neg_features]).astype(np.float32)
    y = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))]).astype(np.float32).reshape(-1, 1)

    # Shuffle
    indices = rng.permutation(len(X))
    return X[indices], y[indices]


def train_impulse_detector_model(
    manifest_path: str = "data/manifests/manifest.csv",
    output_onnx: str = "models/onnx/impulse_detector.onnx",
    epochs: int = 20,
    lr: float = 0.005,
    max_samples: int = 1500,
    val_ratio: float = 0.20,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[TinyImpulseMLP, Dict[str, float]]:
    """
    Trains TinyImpulseMLP on real 8-D acoustic features and exports to ONNX.
    """
    if verbose:
        print("==========================================================================")
        print("       SIH26052 NOICELESSX — Real Feature Impulse Detector Training       ")
        print("==========================================================================")

    X, y = build_real_feature_dataset(
        manifest_path=Path(manifest_path),
        max_samples_per_class=max_samples,
        split="all",
        seed=seed,
    )

    # Train / Val Split
    n_total = len(X)
    n_val = int(n_total * val_ratio)
    X_train, y_train = X[n_val:], y[n_val:]
    X_val, y_val = X[:n_val], y[:n_val]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = TinyImpulseMLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if verbose:
        print(f"Training TinyImpulseMLP ({sum(p.numel() for p in model.parameters())} parameters) on {len(X_train)} samples...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vp = model(vx)
                val_loss += criterion(vp, vy).item() * len(vx)
                all_preds.extend(vp.squeeze(1).numpy())
                all_targets.extend(vy.squeeze(1).numpy())

        val_loss /= len(val_loader.dataset)
        preds_bin = (np.array(all_preds) >= 0.5).astype(int)
        targets_bin = np.array(all_targets).astype(int)
        accuracy = float(np.mean(preds_bin == targets_bin))

        if verbose and (epoch % 5 == 0 or epoch == epochs):
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy * 100:.1f}%")

    # Final Metrics
    tp = np.sum((preds_bin == 1) & (targets_bin == 1))
    fp = np.sum((preds_bin == 1) & (targets_bin == 0))
    fn = np.sum((preds_bin == 0) & (targets_bin == 1))
    precision = float(tp / (tp + fp + 1e-7))
    recall = float(tp / (tp + fn + 1e-7))
    f1 = float(2 * (precision * recall) / (precision + recall + 1e-7))

    metrics = {
        "val_loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    if verbose:
        print("\nFinal Real-Audio Validation Report:")
        print(f"  Accuracy:  {accuracy * 100:.2f}%")
        print(f"  Precision: {precision * 100:.2f}%")
        print(f"  Recall:    {recall * 100:.2f}%")
        print(f"  F1-Score:  {f1:.4f}")

    # Export to ONNX
    if output_onnx:
        if verbose:
            print(f"\nExporting trained impulse detector to ONNX: {output_onnx}")
        export_impulse_model_to_onnx(model, output_path=output_onnx)

    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train TinyImpulseMLP on real acoustic features from manifest."
    )
    parser.add_argument("--manifest", type=str, default="data/manifests/manifest.csv", help="Path to manifest CSV or JSON")
    parser.add_argument("--output", type=str, default="models/onnx/impulse_detector.onnx", help="Path for exported ONNX model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=1500, help="Maximum feature frames per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_impulse_detector_model(
        manifest_path=args.manifest,
        output_onnx=args.output,
        epochs=args.epochs,
        lr=args.lr,
        max_samples=args.max_samples,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
