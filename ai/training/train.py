"""
SIH26052 — NOICELESSX: PyTorch Training Loop & Automated ONNX Export for ComplexCRN.
Trains ComplexCRN on dynamic speech + 200+ noise + RIR mixtures using CompositeEnhancementLoss,
tracks best validation checkpoints, and exports directly to FP32 and INT8 ONNX models for Raspberry Pi.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.datasets.manifest import load_manifest
from ai.export.onnx_export import export_onnx_model
from ai.export.onnx_quantize import quantize_model
from ai.losses.composite_loss import CompositeEnhancementLoss
from ai.models.complex_crn import ComplexCRN
from ai.training.dataset import SpeechEnhancementDataset, collate_speech_batch


class ComplexCRNTrainer:
    """
    Manages training, validation, checkpointing, and automated ONNX export for ComplexCRN.
    """

    def __init__(
        self,
        model: ComplexCRN,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: CompositeEnhancementLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str = "models/checkpoints",
        fft_size: int = 512,
        hop_size: int = 80,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window = torch.hann_window(fft_size, device=device)

        self.best_val_loss = float("inf")

    def _stft_to_waveform(self, complex_stft_2ch: torch.Tensor, length: int) -> torch.Tensor:
        """
        Converts (B, 2, T, F) STFT back to time-domain waveform (B, length) via iSTFT.
        """
        b, c, t, f = complex_stft_2ch.shape
        # Permute (B, 2, T, F) -> (B, F, T) complex tensor
        real = complex_stft_2ch[:, 0].transpose(1, 2).contiguous()
        imag = complex_stft_2ch[:, 1].transpose(1, 2).contiguous()
        cplx = torch.complex(real, imag)
        # iSTFT
        wav = torch.istft(
            cplx,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size,
            window=self.window,
            center=True,
            length=length,
        )
        return wav

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full training epoch."""
        self.model.train()
        total_loss = 0.0
        total_stft = 0.0
        total_cplx = 0.0
        total_sisnr = 0.0
        num_batches = len(self.train_loader)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            noisy_stft = batch["noisy_stft"].to(self.device)  # (B, 2, T, 257)
            clean_stft = batch["clean_stft"].to(self.device)  # (B, 2, T, 257)
            clean_wav = batch["clean_wav"].to(self.device)    # (B, samples)

            self.optimizer.zero_grad()

            # Forward pass through ComplexCRN
            s_hat, mask, _ = self.model(noisy_stft)

            # iSTFT for SI-SNR time-domain loss
            s_hat_wav = self._stft_to_waveform(s_hat, length=clean_wav.shape[-1])

            # Compute composite multi-objective loss
            loss, metrics = self.criterion(
                s_hat_complex=s_hat,
                s_clean_complex=clean_stft,
                s_hat_waveform=s_hat_wav,
                s_clean_waveform=clean_wav,
            )

            # Backpropagation
            loss.backward()
            # Gradient clipping to prevent gradient explosion in deep RNNs
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_stft += metrics.get("loss_stft", 0.0)
            total_cplx += metrics.get("loss_complex", 0.0)
            total_sisnr += metrics.get("loss_sisnr", 0.0)

            if (batch_idx + 1) % max(1, num_batches // 5) == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch:02d} [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {loss.item():.4f} (STFT: {metrics.get('loss_stft', 0.0):.4f}, "
                    f"Cplx: {metrics.get('loss_complex', 0.0):.4f}, "
                    f"SI-SNR: {metrics.get('loss_sisnr', 0.0):.4f}) | "
                    f"{elapsed:.1f}s"
                )

        avg_metrics = {
            "loss": total_loss / num_batches,
            "loss_stft": total_stft / num_batches,
            "loss_complex": total_cplx / num_batches,
            "loss_sisnr": total_sisnr / num_batches,
        }
        return avg_metrics

    def validate(self) -> Dict[str, float]:
        """Evaluates model on validation set without gradients."""
        if not self.val_loader or len(self.val_loader) == 0:
            return {"loss": 0.0}

        self.model.eval()
        total_loss = 0.0
        total_stft = 0.0
        total_cplx = 0.0
        total_sisnr = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                noisy_stft = batch["noisy_stft"].to(self.device)
                clean_stft = batch["clean_stft"].to(self.device)
                clean_wav = batch["clean_wav"].to(self.device)

                s_hat, mask, _ = self.model(noisy_stft)
                s_hat_wav = self._stft_to_waveform(s_hat, length=clean_wav.shape[-1])

                loss, metrics = self.criterion(
                    s_hat_complex=s_hat,
                    s_clean_complex=clean_stft,
                    s_hat_waveform=s_hat_wav,
                    s_clean_waveform=clean_wav,
                )

                total_loss += loss.item()
                total_stft += metrics.get("loss_stft", 0.0)
                total_cplx += metrics.get("loss_complex", 0.0)
                total_sisnr += metrics.get("loss_sisnr", 0.0)

        return {
            "loss": total_loss / num_batches,
            "loss_stft": total_stft / num_batches,
            "loss_complex": total_cplx / num_batches,
            "loss_sisnr": total_sisnr / num_batches,
        }

    def train(self, num_epochs: int = 5) -> Tuple[Path, Path]:
        """
        Executes multi-epoch training and validation loop.
        Returns paths to (best_checkpoint, last_checkpoint).
        """
        best_ckpt = self.checkpoint_dir / "best_model.pth"
        last_ckpt = self.checkpoint_dir / "last_model.pth"

        print("\n==========================================================================")
        print("          SIH26052 NOICELESSX — ComplexCRN Speech Enhancement Training    ")
        print("==========================================================================")
        print(f"Device:           {self.device}")
        print(f"Epochs:           {num_epochs}")
        print(f"Train Batches:    {len(self.train_loader)}")
        print(f"Val Batches:      {len(self.val_loader) if self.val_loader else 0}")
        print(f"Checkpoints:      {self.checkpoint_dir}")
        print("==========================================================================\n")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            print(
                f">> Epoch {epoch:02d} Complete | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )

            # Step learning rate scheduler if present
            if self.scheduler:
                self.scheduler.step(val_metrics["loss"])

            # Save last checkpoint
            torch.save(self.model.state_dict(), str(last_ckpt))

            # Save best checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                torch.save(self.model.state_dict(), str(best_ckpt))
                print(f"  [*] Saved new best model checkpoint (Val Loss: {self.best_val_loss:.4f})")

        print("\nTraining completed successfully.")
        return best_ckpt, last_ckpt


def train_speech_enhancer(
    train_manifest: str = "data/manifests/train.json",
    val_manifest: str = "data/manifests/val.json",
    noise_manifest: str = "data/manifests/noise.json",
    rir_manifest: Optional[str] = "data/manifests/rir.json",
    checkpoint_dir: str = "models/checkpoints",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 5e-4,
    device_name: Optional[str] = None,
    export_onnx: bool = True,
    quantize_int8: bool = True,
) -> Dict[str, str]:
    """
    High-level entry point to run dataset loading, model training,
    best checkpoint selection, and ONNX export/quantization.
    """
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Datasets & Loaders
    train_dataset = SpeechEnhancementDataset(
        clean_manifest_path=train_manifest,
        noise_manifest_path=noise_manifest,
        rir_manifest_path=rir_manifest,
        segment_duration=2.0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_speech_batch,
        num_workers=0,
    )

    val_loader = None
    if val_manifest and os.path.exists(val_manifest):
        try:
            val_dataset = SpeechEnhancementDataset(
                clean_manifest_path=val_manifest,
                noise_manifest_path=noise_manifest,
                rir_manifest_path=rir_manifest,
                segment_duration=2.0,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_speech_batch,
                num_workers=0,
            )
        except ValueError:
            val_loader = None

    # Instantiate ComplexCRN
    model = ComplexCRN(num_bins=257)

    # Loss & Optimizer
    criterion = CompositeEnhancementLoss(lambda_sisnr=1.0, lambda_stft=1.0, lambda_complex=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    trainer = ComplexCRNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    best_ckpt, last_ckpt = trainer.train(num_epochs=epochs)

    results = {
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
    }

    # Post-training automated ONNX export
    if export_onnx:
        fp32_onnx_path = "models/onnx/speech_enhancer_fp32.onnx"
        print(f"\n[Post-Training] Exporting best checkpoint to ONNX: {fp32_onnx_path}")
        export_success = export_onnx_model(
            checkpoint_path=str(best_ckpt),
            output_path=fp32_onnx_path,
            num_bins=257,
            verbose=True,
        )
        if export_success:
            results["onnx_fp32"] = fp32_onnx_path

            if quantize_int8:
                int8_onnx_path = "models/onnx/speech_enhancer_int8.onnx"
                print(f"\n[Post-Training] Quantizing to INT8: {int8_onnx_path}")
                quant_success = quantize_model(
                    input_fp32=fp32_onnx_path,
                    output_int8=int8_onnx_path,
                    verbose=True,
                )
                if quant_success:
                    results["onnx_int8"] = int8_onnx_path

    return results


def main():
    parser = argparse.ArgumentParser(description="Train ComplexCRN Speech Enhancement model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--train-manifest", type=str, default="data/manifests/train.json")
    parser.add_argument("--val-manifest", type=str, default="data/manifests/val.json")
    parser.add_argument("--noise-manifest", type=str, default="data/manifests/noise.json")
    parser.add_argument("--rir-manifest", type=str, default="data/manifests/rir.json")
    parser.add_argument("--checkpoints", type=str, default="models/checkpoints")
    parser.add_argument("--no-onnx", action="store_true", help="Skip post-training ONNX export")
    parser.add_argument("--no-quant", action="store_true", help="Skip INT8 quantization")
    args = parser.parse_args()

    train_speech_enhancer(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        noise_manifest=args.noise_manifest,
        rir_manifest=args.rir_manifest,
        checkpoint_dir=args.checkpoints,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        export_onnx=not args.no_onnx,
        quantize_int8=not args.no_quant,
    )


if __name__ == "__main__":
    main()
