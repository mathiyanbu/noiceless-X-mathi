#!/usr/bin/env python3
"""
SIH26052 — NOICELESSX: PyTorch ComplexCRN Training, Checkpointing, Equivalence Verification & Test Evaluation.
Trains ComplexCRN on real speech + noise + RIR mixtures from dataset manifest using CompositeEnhancementLoss,
enforces streaming equivalence assertions, tracks best checkpoints, logs structured metrics to CSV,
evaluates on held-out test split (ΔSNR, ΔSI-SNR, STOI, PESQ), and exports directly to FP32 & INT8 ONNX models.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.evaluation.metrics import (
    compute_pesq,
    compute_si_sdr,
    compute_stoi,
    compute_true_snr,
)
from ai.losses.composite_loss import CompositeEnhancementLoss
from ai.models.complex_crn import ComplexCRN
from ai.training.dataset import (
    ManifestMixtureDataset,
    SpeechEnhancementDataset,
    collate_speech_batch,
)


def verify_streaming_equivalence(
    model: ComplexCRN,
    device: torch.device,
    fft_size: int = 512,
    hop_size: int = 80,
    tolerance: float = 1e-4,
    time_steps: int = 201,  # ~1.0 second of audio at 16kHz
) -> float:
    """
    Validation check: runs a 1-second test input through the model in batch mode,
    then runs it frame-by-frame in streaming mode, updating GRU hidden state.
    Asserts max absolute difference < tolerance (1e-4).
    If difference exceeds tolerance, raises RuntimeError flagging causality violation.
    """
    model.eval()
    rng = torch.Generator(device="cpu").manual_seed(42)
    # Deterministic complex STFT: (1, 2, time_steps, num_bins=257)
    test_stft = torch.randn(1, 2, time_steps, fft_size // 2 + 1, generator=rng).float().to(device)

    with torch.no_grad():
        # 1. Batch Execution
        s_hat_batch, _, _ = model(test_stft)

        # 2. Frame-by-Frame Streaming Execution
        hidden = model.init_hidden(1, device=device)
        stream_frames = []
        for t in range(time_steps):
            frame = test_stft[:, :, t : t + 1, :]
            s_frame, _, hidden = model.forward(frame, hidden)
            stream_frames.append(s_frame)
        s_hat_stream = torch.cat(stream_frames, dim=2)

        max_diff = float(torch.max(torch.abs(s_hat_batch - s_hat_stream)).item())

    if max_diff >= tolerance:
        raise RuntimeError(
            f"Streaming equivalence assertion FAILED! Max absolute diff {max_diff:.2e} >= {tolerance}. "
            f"Causality or recurrent state preservation violation detected."
        )

    return max_diff


class ComplexCRNTrainer:
    """
    Manages multi-objective training, mixed-precision acceleration, validation,
    streaming equivalence assertions, structured CSV logging, checkpointing, and resume.
    """

    def __init__(
        self,
        model: ComplexCRN,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: CompositeEnhancementLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
        use_amp: bool = False,
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
        self.use_amp = use_amp and device.type == "cuda"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window = torch.hann_window(fft_size, device=device)

        # Mixed precision GradScaler
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.best_val_loss = float("inf")
        self.start_epoch = 1
        self.log_file = self.checkpoint_dir / "training_log.csv"
        self._init_csv_logger()

    def _init_csv_logger(self):
        """Initializes structured CSV logger with column headers if not already created."""
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_loss_stft",
                    "train_loss_complex",
                    "train_loss_sisnr",
                    "val_loss",
                    "val_loss_stft",
                    "val_loss_complex",
                    "val_loss_sisnr",
                    "val_sisnr_db",
                    "streaming_max_diff",
                    "learning_rate",
                    "epoch_time_sec",
                ])

    def _stft_to_waveform(self, complex_stft_2ch: torch.Tensor, length: int) -> torch.Tensor:
        """Converts (B, 2, T, F) complex STFT back to (B, length) waveform via iSTFT."""
        real = complex_stft_2ch[:, 0].transpose(1, 2).contiguous()
        imag = complex_stft_2ch[:, 1].transpose(1, 2).contiguous()
        cplx = torch.complex(real, imag)
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

    def resume_from_checkpoint(self, checkpoint_path: Path):
        """Loads model weights, optimizer, and scheduler states to resume training."""
        if not checkpoint_path.exists():
            print(f"[Warning] Resume checkpoint not found: {checkpoint_path}")
            return

        print(f"[Checkpoint] Resuming from: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"[Checkpoint] Successfully resumed from epoch {self.start_epoch} (Best Val Loss: {self.best_val_loss:.4f})")
        else:
            # Raw state_dict
            self.model.load_state_dict(ckpt)
            print("[Checkpoint] Loaded model weights from state_dict.")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Executes one full training epoch with mixed precision and gradient clipping."""
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
            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=self.use_amp)
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                else torch.cuda.amp.autocast(enabled=self.use_amp)
            )
            with autocast_ctx:
                # Forward pass through ComplexCRN
                s_hat, mask, _ = self.model(noisy_stft)

                # iSTFT to reconstruct enhanced time-domain signal
                s_hat_wav = self._stft_to_waveform(s_hat, length=clean_wav.shape[-1])

                # Multi-objective composite loss
                loss, metrics = self.criterion(
                    s_hat_complex=s_hat,
                    s_clean_complex=clean_stft,
                    s_hat_waveform=s_hat_wav,
                    s_clean_waveform=clean_wav,
                )

            # Backward pass with mixed-precision scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
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

        return {
            "loss": total_loss / max(1, num_batches),
            "loss_stft": total_stft / max(1, num_batches),
            "loss_complex": total_cplx / max(1, num_batches),
            "loss_sisnr": total_sisnr / max(1, num_batches),
        }

    def validate(self) -> Dict[str, float]:
        """Evaluates model on validation set and verifies streaming equivalence."""
        if not self.val_loader or len(self.val_loader) == 0:
            # Still run streaming equivalence check
            diff = verify_streaming_equivalence(self.model, self.device, self.fft_size, self.hop_size)
            return {
                "loss": 0.0,
                "loss_stft": 0.0,
                "loss_complex": 0.0,
                "loss_sisnr": 0.0,
                "val_sisnr_db": 0.0,
                "streaming_max_diff": diff,
            }

        self.model.eval()
        total_loss = 0.0
        total_stft = 0.0
        total_cplx = 0.0
        total_sisnr = 0.0
        val_sisnr_sum = 0.0
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

                # Measure actual SI-SNR in dB: -loss_sisnr
                val_sisnr_sum += -metrics.get("loss_sisnr", 0.0)

        # Streaming equivalence verification (raises RuntimeError if violated)
        streaming_diff = verify_streaming_equivalence(
            self.model, self.device, self.fft_size, self.hop_size, tolerance=1e-4
        )

        return {
            "loss": total_loss / num_batches,
            "loss_stft": total_stft / num_batches,
            "loss_complex": total_cplx / num_batches,
            "loss_sisnr": total_sisnr / num_batches,
            "val_sisnr_db": val_sisnr_sum / num_batches,
            "streaming_max_diff": streaming_diff,
        }

    def train(self, num_epochs: int = 5) -> Tuple[Path, Path]:
        """
        Executes multi-epoch training, validation, checkpointing, and structured logging.
        Returns paths to (best_checkpoint, last_checkpoint).
        """
        best_ckpt = self.checkpoint_dir / "best_model.pth"
        last_ckpt = self.checkpoint_dir / "last_model.pth"

        current_lr = self.optimizer.param_groups[0]["lr"]
        print("\n==========================================================================")
        print("          SIH26052 NOICELESSX — ComplexCRN Speech Enhancement Training    ")
        print("==========================================================================")
        print(f"Device:               {self.device} ({'AMP' if self.use_amp else 'FP32'})")
        print(f"Epochs:               {self.start_epoch} -> {self.start_epoch + num_epochs - 1}")
        print(f"Train Batches:        {len(self.train_loader)}")
        print(f"Val Batches:          {len(self.val_loader) if self.val_loader else 0}")
        print(f"Initial LR:           {current_lr}")
        print(f"Checkpoints Dir:      {self.checkpoint_dir}")
        print(f"Structured Log:       {self.log_file}")
        print("==========================================================================\n")

        end_epoch = self.start_epoch + num_epochs - 1

        for epoch in range(self.start_epoch, end_epoch + 1):
            epoch_t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            epoch_duration = time.time() - epoch_t0

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f">> Epoch {epoch:02d} Complete | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val SI-SNR: {val_metrics['val_sisnr_db']:+.2f} dB | "
                f"Stream Equiv Diff: {val_metrics['streaming_max_diff']:.2e} (< 1e-4) | "
                f"LR: {current_lr:.2e} | {epoch_duration:.1f}s"
            )

            # Step scheduler if configured
            if self.scheduler:
                self.scheduler.step(val_metrics["loss"])

            # Save last checkpoint (full resumable payload)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "best_val_loss": self.best_val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                str(last_ckpt),
            )

            # Save best checkpoint
            val_loss = val_metrics["loss"] if self.val_loader else train_metrics["loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_loss": self.best_val_loss,
                        "val_metrics": val_metrics,
                    },
                    str(best_ckpt),
                )
                print(f"  [*] Saved new best model checkpoint (Val Loss: {self.best_val_loss:.4f})")

            # Structured CSV logging
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['loss_stft']:.6f}",
                    f"{train_metrics['loss_complex']:.6f}",
                    f"{train_metrics['loss_sisnr']:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['loss_stft']:.6f}",
                    f"{val_metrics['loss_complex']:.6f}",
                    f"{val_metrics['loss_sisnr']:.6f}",
                    f"{val_metrics['val_sisnr_db']:.4f}",
                    f"{val_metrics['streaming_max_diff']:.3e}",
                    f"{current_lr:.6e}",
                    f"{epoch_duration:.2f}",
                ])

        print("\nTraining completed successfully.")
        return best_ckpt, last_ckpt


def evaluate_test_split(
    model: ComplexCRN,
    test_loader: DataLoader,
    device: torch.device,
    output_path: Optional[Path] = None,
    fft_size: int = 512,
    hop_size: int = 80,
) -> Dict[str, Any]:
    """
    Offline evaluation on held-out test split using real metrics:
    - Ground-truth SNR improvement: ΔSNR = SNR_out - SNR_in
    - Scale-Invariant SNR improvement: ΔSI-SNR = SI-SNR_out - SI-SNR_in
    - Real STOI via pystoi
    - Real PESQ via pesq (or explicit null/notice if pesq C library not compiled on Windows)
    Zero fabricated numbers.
    """
    model.eval()
    window = torch.hann_window(fft_size, device=device)

    snr_in_list = []
    snr_out_list = []
    delta_snr_list = []
    sisnr_in_list = []
    sisnr_out_list = []
    delta_sisnr_list = []
    stoi_list = []
    pesq_list = []

    print("\n==========================================================================")
    print("      SIH26052 NOICELESSX — Held-Out Test Split Ground-Truth Evaluation   ")
    print("==========================================================================")
    print(f"Test Batches: {len(test_loader)}")
    print("Computing real SNR, SI-SNR, STOI, and PESQ on decoded waveforms...")
    print("--------------------------------------------------------------------------")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            noisy_stft = batch["noisy_stft"].to(device)
            clean_wav_batch = batch["clean_wav"].numpy()
            noisy_wav_batch = batch["noisy_wav"].numpy()

            # Forward pass through model
            s_hat, _, _ = model(noisy_stft)

            # iSTFT
            real = s_hat[:, 0].transpose(1, 2).contiguous()
            imag = s_hat[:, 1].transpose(1, 2).contiguous()
            cplx = torch.complex(real, imag)
            enhanced_wav_batch = torch.istft(
                cplx,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=fft_size,
                window=window,
                center=True,
                length=clean_wav_batch.shape[-1],
            ).cpu().numpy()

            # Evaluate each waveform in the batch
            for i in range(len(clean_wav_batch)):
                c = clean_wav_batch[i]
                n = noisy_wav_batch[i]
                e = enhanced_wav_batch[i]

                # True SNR
                snr_res = compute_true_snr(c, n, e)
                snr_in_list.append(snr_res["snr_in"])
                snr_out_list.append(snr_res["snr_out"])
                delta_snr_list.append(snr_res["delta_snr"])

                # SI-SNR
                sisnr_in = compute_si_sdr(c, n)
                sisnr_out = compute_si_sdr(c, e)
                sisnr_in_list.append(sisnr_in)
                sisnr_out_list.append(sisnr_out)
                delta_sisnr_list.append(sisnr_out - sisnr_in)

                # STOI
                stoi_val = compute_stoi(c, e, sample_rate=16000)
                if stoi_val is not None:
                    stoi_list.append(stoi_val)

                # PESQ
                pesq_val = compute_pesq(c, e, sample_rate=16000)
                if pesq_val is not None:
                    pesq_list.append(pesq_val)

    # Compute aggregates
    num_samples = len(delta_snr_list)
    avg_snr_in = float(np.mean(snr_in_list)) if snr_in_list else 0.0
    avg_snr_out = float(np.mean(snr_out_list)) if snr_out_list else 0.0
    avg_delta_snr = float(np.mean(delta_snr_list)) if delta_snr_list else 0.0

    avg_sisnr_in = float(np.mean(sisnr_in_list)) if sisnr_in_list else 0.0
    avg_sisnr_out = float(np.mean(sisnr_out_list)) if sisnr_out_list else 0.0
    avg_delta_sisnr = float(np.mean(delta_sisnr_list)) if delta_sisnr_list else 0.0

    avg_stoi = float(np.mean(stoi_list)) if stoi_list else None
    avg_pesq = float(np.mean(pesq_list)) if pesq_list else None

    print(f"Evaluated Test Samples:  {num_samples}")
    print(f"Mean Input SNR:          {avg_snr_in:8.2f} dB")
    print(f"Mean Output SNR:         {avg_snr_out:8.2f} dB")
    print(f"Mean SNR Improvement:    {avg_delta_snr:+8.2f} dB (Delta-SNR)")
    print(f"Mean Input SI-SNR:       {avg_sisnr_in:8.2f} dB")
    print(f"Mean Output SI-SNR:      {avg_sisnr_out:8.2f} dB")
    print(f"Mean SI-SNR Improvement: {avg_delta_sisnr:+8.2f} dB (Delta-SI-SNR)")
    print(f"Mean STOI:               {f'{avg_stoi:.4f}' if avg_stoi is not None else 'N/A (pystoi error)'}")
    print(f"Mean PESQ (WB):          {f'{avg_pesq:.3f}' if avg_pesq is not None else 'N/A (C library uncompiled)'}")
    print("==========================================================================\n")

    results = {
        "num_test_samples": num_samples,
        "mean_snr_in_db": avg_snr_in,
        "mean_snr_out_db": avg_snr_out,
        "mean_delta_snr_db": avg_delta_snr,
        "mean_sisnr_in_db": avg_sisnr_in,
        "mean_sisnr_out_db": avg_sisnr_out,
        "mean_delta_sisnr_db": avg_delta_sisnr,
        "mean_stoi": avg_stoi,
        "mean_pesq": avg_pesq,
        "pesq_status": "computed" if avg_pesq is not None else "uncompiled_on_windows_platform",
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved test evaluation report to: {output_path}")

    return results


def train_speech_enhancer(
    manifest_path: str = "data/manifests/manifest.csv",
    train_manifest: Optional[str] = None,
    val_manifest: Optional[str] = None,
    noise_manifest: Optional[str] = None,
    rir_manifest: Optional[str] = None,
    checkpoint_dir: str = "models/checkpoints",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 5e-4,
    device_name: Optional[str] = None,
    resume_path: Optional[str] = None,
    eval_test: bool = True,
    export_onnx: bool = True,
    quantize_int8: bool = True,
) -> Dict[str, Any]:
    """
    Primary training pipeline:
    1. Detects hardware (CUDA AMP / CPU FP32).
    2. Constructs train and val DataLoaders from real manifest.
    3. Initializes ComplexCRN with causal temporal GRU.
    4. Runs training loop with composite loss and streaming equivalence assertions.
    5. Saves best and last checkpoints.
    6. Performs offline evaluation on held-out test split.
    7. Exports to FP32 and INT8 ONNX models.
    """
    # 1. Hardware Detection
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = device.type == "cuda"
    if use_amp:
        print(f"[Hardware] CUDA GPU detected: {torch.cuda.get_device_name(0)} | Mixed Precision (AMP) ENABLED")
    else:
        print("[Hardware] Running on CPU with float32 precision.")

    # 2. Datasets and Loaders
    has_unified = Path(manifest_path).exists()

    if has_unified:
        print(f"[Dataset] Loading unified manifest: {manifest_path}")
        train_dataset = SpeechEnhancementDataset(
            manifest_path=manifest_path,
            split="train",
            segment_duration=2.0,
            rir_prob=0.40,
            impulse_prob=0.15,
            clipping_prob=0.10,
        )
        val_dataset = None
        try:
            val_dataset = SpeechEnhancementDataset(
                manifest_path=manifest_path,
                split="val",
                segment_duration=2.0,
                rir_prob=0.40,
                impulse_prob=0.15,
                clipping_prob=0.10,
            )
        except ValueError:
            val_dataset = None
    else:
        # Legacy fallback
        print(f"[Dataset] Loading legacy manifests: {train_manifest}, {noise_manifest}")
        train_dataset = SpeechEnhancementDataset(
            clean_manifest_path=train_manifest or "data/manifests/train.json",
            noise_manifest_path=noise_manifest or "data/manifests/noise.json",
            rir_manifest_path=rir_manifest,
            segment_duration=2.0,
        )
        val_dataset = None
        if val_manifest and os.path.exists(val_manifest):
            try:
                val_dataset = SpeechEnhancementDataset(
                    clean_manifest_path=val_manifest,
                    noise_manifest_path=noise_manifest or "data/manifests/noise.json",
                    rir_manifest_path=rir_manifest,
                    segment_duration=2.0,
                )
            except ValueError:
                val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_speech_batch,
        num_workers=0,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_speech_batch,
            num_workers=0,
        )

    # 3. Model Architecture
    model = ComplexCRN(num_bins=257)

    # 4. Multi-Objective Loss, Optimizer & Scheduler
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
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir,
    )

    # Resume if requested
    if resume_path:
        trainer.resume_from_checkpoint(Path(resume_path))

    # 5. Execute Training Loop
    best_ckpt, last_ckpt = trainer.train(num_epochs=epochs)

    results: Dict[str, Any] = {
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "best_val_loss": trainer.best_val_loss,
    }

    # Load best checkpoint weights for evaluation and ONNX export
    best_weights = torch.load(str(best_ckpt), map_location=device)
    if isinstance(best_weights, dict) and "model_state_dict" in best_weights:
        model.load_state_dict(best_weights["model_state_dict"])
    else:
        model.load_state_dict(best_weights)

    # 6. Held-Out Test Split Evaluation
    if eval_test and has_unified:
        try:
            test_dataset = SpeechEnhancementDataset(
                manifest_path=manifest_path,
                split="test",
                segment_duration=2.0,
                rir_prob=0.40,
                impulse_prob=0.15,
                clipping_prob=0.10,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_speech_batch,
                num_workers=0,
            )
            test_res_path = Path(checkpoint_dir) / "evaluation_test_results.json"
            test_metrics = evaluate_test_split(
                model=model,
                test_loader=test_loader,
                device=device,
                output_path=test_res_path,
            )
            results["test_evaluation"] = test_metrics
        except ValueError as e:
            print(f"[Notice] Could not run test evaluation: {e}")

    # 7. Post-Training Automated ONNX Export & INT8 Quantization
    if export_onnx:
        from ai.export.onnx_export import export_onnx_model
        from ai.export.onnx_quantize import quantize_model

        fp32_onnx_path = "models/onnx/speech_enhancer_fp32.onnx"
        print(f"\n[Post-Training] Exporting best model to ONNX: {fp32_onnx_path}")
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
    parser.add_argument("--manifest", type=str, default="data/manifests/manifest.csv", help="Path to unified manifest CSV or JSON")
    parser.add_argument("--train-manifest", type=str, default=None, help="Legacy clean speech train manifest")
    parser.add_argument("--val-manifest", type=str, default=None, help="Legacy clean speech val manifest")
    parser.add_argument("--noise-manifest", type=str, default=None, help="Legacy noise manifest")
    parser.add_argument("--rir-manifest", type=str, default=None, help="Legacy RIR manifest")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--checkpoints", type=str, default="models/checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--eval-test", action="store_true", default=True, help="Evaluate on held-out test split after training")
    parser.add_argument("--no-onnx", action="store_true", help="Skip post-training ONNX export")
    parser.add_argument("--no-quant", action="store_true", help="Skip INT8 quantization")
    args = parser.parse_args()

    train_speech_enhancer(
        manifest_path=args.manifest,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        noise_manifest=args.noise_manifest,
        rir_manifest=args.rir_manifest,
        checkpoint_dir=args.checkpoints,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_path=args.resume,
        eval_test=args.eval_test,
        export_onnx=not args.no_onnx,
        quantize_int8=not args.no_quant,
    )


if __name__ == "__main__":
    main()
