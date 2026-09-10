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
from typing import Any, Dict, List, Optional, Tuple

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


from ai.preprocessing.mixer import classify_noise_into_bucket


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
    split: str = "all",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads real audio files from manifest, extracting 8-D feature vectors:
    - Positive Class (1.0): real audio from 'impulsive' bucket (gunshot, glass break, door slam, bark, clap, etc.)
    - Negative Class (0.0): real audio from 'stationary' and 'non_stationary' noise buckets.
    Audits and prints exact class list and clip counts.
    """
    rng = np.random.RandomState(seed)

    impulsive_files: List[Tuple[Path, str]] = []
    non_impulsive_files: List[Tuple[Path, str]] = []
    impulsive_classes_found = set()

    # Parse manifest CSV or JSON
    records = []
    if manifest_path.suffix.lower() == ".csv":
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f)

    for r in records:
        if split != "all" and r.get("split", "train").lower() != split.lower():
            continue
        fpath = Path(r["filepath"])
        if not fpath.exists():
            continue

        label = str(r.get("class_label", "unknown"))
        bucket = classify_noise_into_bucket(r)
        if bucket == "impulsive":
            impulsive_files.append((fpath, label))
            impulsive_classes_found.add(label)
        elif bucket in {"stationary", "non_stationary"}:
            non_impulsive_files.append((fpath, label))

    # Audit printout
    audited_classes = sorted(list(impulsive_classes_found))
    print("\n==========================================================================")
    print("      SIH26052 NOICELESSX — Impulse Detector Real Audio Dataset Audit      ")
    print("==========================================================================")
    print(f"Manifest source:          {manifest_path} (Split: {split})")
    print(f"Impulsive audio clips:    {len(impulsive_files)} clips across {len(audited_classes)} verified classes")
    print(f"Non-impulsive clips:      {len(non_impulsive_files)} clips (stationary/non-stationary noise)")
    print(f"Audited Impulsive Classes: {audited_classes}")
    print("--------------------------------------------------------------------------")

    # 1. Extract Impulsive Features (Class 1) across all impulsive files
    pos_features: List[np.ndarray] = []
    rng.shuffle(impulsive_files)
    max_per_pos_file = max(5, int(np.ceil(max_samples_per_class / max(1, len(impulsive_files)))))
    for f, _ in impulsive_files:
        file_feats = extract_features_from_audio_file(f)
        if not file_feats:
            continue
        arr_feats = np.array(file_feats)
        # Select peak transient/onset frames (highest crest factor & spectral flux onset)
        scores = arr_feats[:, 2] * np.log1p(np.maximum(0.0, arr_feats[:, 1]))
        n_take = min(len(arr_feats), max_per_pos_file)
        top_idx = np.argsort(scores)[-n_take:]
        pos_features.extend(arr_feats[top_idx])
        if len(pos_features) >= max_samples_per_class:
            break
    pos_features = pos_features[:max_samples_per_class]

    # 2. Extract Non-Impulsive Features (Class 0) across non-impulsive files
    neg_features: List[np.ndarray] = []
    rng.shuffle(non_impulsive_files)
    max_per_neg_file = max(5, int(np.ceil(max_samples_per_class / max(1, len(non_impulsive_files)))))
    for f, _ in non_impulsive_files:
        file_feats = extract_features_from_audio_file(f)
        if not file_feats:
            continue
        arr_feats = np.array(file_feats)
        n_take = min(len(arr_feats), max_per_neg_file)
        sample_idx = rng.choice(len(arr_feats), size=n_take, replace=False)
        neg_features.extend(arr_feats[sample_idx])
        if len(neg_features) >= max_samples_per_class:
            break
    neg_features = neg_features[:max_samples_per_class]

    print(f"Extracted {len(pos_features)} real impulsive feature vectors (Class 1).")
    print(f"Extracted {len(neg_features)} real non-impulsive feature vectors (Class 0).")
    print("==========================================================================\n")

    X = np.vstack([pos_features, neg_features]).astype(np.float32)
    y = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))]).astype(np.float32).reshape(-1, 1)

    # Shuffle
    indices = rng.permutation(len(X))
    meta = {
        "num_pos_files": len(impulsive_files),
        "num_neg_files": len(non_impulsive_files),
        "audited_classes": audited_classes,
        "num_pos_frames": len(pos_features),
        "num_neg_frames": len(neg_features),
    }
    return X[indices], y[indices], meta


def train_impulse_detector_model(
    manifest_path: str = "data/manifests/manifest.csv",
    output_onnx: str = "models/onnx/impulse_detector.onnx",
    epochs: int = 20,
    lr: float = 0.005,
    max_samples: int = 1500,
    val_ratio: float = 0.20,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[TinyImpulseMLP, Dict[str, Any]]:
    """
    Trains TinyImpulseMLP on real 8-D acoustic features and exports to ONNX.
    Reports precision, recall, F1, and 2x2 confusion matrix on held-out split.
    """
    if verbose:
        print("==========================================================================")
        print("       SIH26052 NOICELESSX — Real Feature Impulse Detector Training       ")
        print("==========================================================================")

    X, y, meta = build_real_feature_dataset(
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

    # 2x2 Confusion Matrix Computation on Held-Out Split
    tp = int(np.sum((preds_bin == 1) & (targets_bin == 1)))
    fp = int(np.sum((preds_bin == 1) & (targets_bin == 0)))
    fn = int(np.sum((preds_bin == 0) & (targets_bin == 1)))
    tn = int(np.sum((preds_bin == 0) & (targets_bin == 0)))

    precision = float(tp / (tp + fp + 1e-7))
    recall = float(tp / (tp + fn + 1e-7))
    f1 = float(2 * (precision * recall) / (precision + recall + 1e-7))

    metrics = {
        "val_loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        },
        "dataset_metadata": meta,
    }

    if verbose:
        print("\n==========================================================================")
        print("    SIH26052 NOICELESSX — Held-Out Split Impulse Evaluation & Confusion   ")
        print("==========================================================================")
        print(f"  Accuracy:         {accuracy * 100:.2f}%")
        print(f"  Precision:        {precision * 100:.2f}%")
        print(f"  Recall:           {recall * 100:.2f}%")
        print(f"  F1-Score:         {f1:.4f}")
        print("--------------------------------------------------------------------------")
        print("  Confusion Matrix (2x2):")
        print(f"                    Predicted Negative    Predicted Positive")
        print(f"    Actual Negative   TN = {tn:<8d}       FP = {fp:<8d}")
        print(f"    Actual Positive   FN = {fn:<8d}       TP = {tp:<8d}")
        print("==========================================================================\n")

    # Export to ONNX
    if output_onnx:
        if verbose:
            print(f"Exporting trained impulse detector to ONNX: {output_onnx}")
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
