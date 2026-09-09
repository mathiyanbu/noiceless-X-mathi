#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.models.impulse_detector.model import TinyImpulseMLP, export_impulse_model_to_onnx
from ai.models.impulse_detector.dataset import create_impulse_dataset

def train_impulse_detector(
    epochs: int = 25,
    lr: float = 0.005,
    output_onnx: str = "models/onnx/impulse_detector.onnx",
    samples_per_class: int = 1500,
    verbose: bool = True
) -> TinyImpulseMLP:
    """Trains TinyImpulseMLP and exports to ONNX."""
    if verbose:
        print("==========================================================================")
        print("        SIH26052 NOICELESSX — Impulsive Noise Detector Training          ")
        print("==========================================================================")

    # 1. Dataset
    if verbose:
        print(f"[Step 1] Generating training dataset ({samples_per_class * 2} balanced frames)...")
    train_loader, val_loader = create_impulse_dataset(n_samples_per_class=samples_per_class)

    # 2. Model & Optimizer
    model = TinyImpulseMLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if verbose:
        print("[Step 2] Training TinyImpulseMLP (289 parameters)...")

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                p = model(x_val)
                v_loss = criterion(p, y_val)
                val_loss += v_loss.item() * len(x_val)
                all_preds.extend(p.squeeze(1).numpy())
                all_targets.extend(y_val.squeeze(1).numpy())

        val_loss /= len(val_loader.dataset)
        preds_bin = (np.array(all_preds) >= 0.5).astype(int)
        targets_bin = np.array(all_targets).astype(int)
        accuracy = float(np.mean(preds_bin == targets_bin))

        if verbose and (epoch % 5 == 0 or epoch == epochs):
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy * 100:.1f}%")

    # 4. Evaluation Metrics
    tp = np.sum((preds_bin == 1) & (targets_bin == 1))
    fp = np.sum((preds_bin == 1) & (targets_bin == 0))
    fn = np.sum((preds_bin == 0) & (targets_bin == 1))
    precision = float(tp / (tp + fp + 1e-7))
    recall = float(tp / (tp + fn + 1e-7))
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    if verbose:
        print("\n[Step 3] Final Validation Performance:")
        print(f"  Accuracy:  {accuracy * 100:.2f}%")
        print(f"  Precision: {precision * 100:.2f}%")
        print(f"  Recall:    {recall * 100:.2f}%")
        print(f"  F1-Score:  {f1:.4f}")

    assert accuracy >= 0.85, f"Validation accuracy {accuracy:.2f} below target 0.85!"

    # 5. Export to ONNX
    if verbose:
        print(f"\n[Step 4] Exporting model to ONNX: {output_onnx}")
    export_impulse_model_to_onnx(model, output_path=output_onnx)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyImpulseMLP and export to ONNX.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--output", type=str, default="models/onnx/impulse_detector.onnx", help="Output ONNX path")
    parser.add_argument("--samples", type=int, default=1500, help="Samples per class")
    args = parser.parse_args()

    train_impulse_detector(
        epochs=args.epochs,
        lr=args.lr,
        output_onnx=args.output,
        samples_per_class=args.samples,
        verbose=True
    )
