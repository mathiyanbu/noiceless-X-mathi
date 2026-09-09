import math
import numpy as np
import pytest
import torch

from ai.models.complex_crn import ComplexCRN
from ai.losses.composite_loss import (
    SISNRLoss,
    STFTMagnitudeLoss,
    ComplexSpectralLoss,
    CompositeEnhancementLoss
)
from ai.preprocessing.mixer import AudioMixer, DatasetSplitter

def test_complex_crn_io_shapes():
    """Verify I/O tensor shapes through ComplexCRN at each stage."""
    model = ComplexCRN(num_bins=257)
    model.eval()

    batch_size = 2
    time_steps = 8
    num_bins = 257

    # Noisy STFT input (B, 2, T, F) where 2 = (Real, Imag)
    x = torch.randn(batch_size, 2, time_steps, num_bins)

    with torch.no_grad():
        s_hat, mask, new_hidden = model(x)

    assert s_hat.shape == (batch_size, 2, time_steps, num_bins)
    assert mask.shape == (batch_size, 2, time_steps, num_bins)
    assert new_hidden.shape == (model.gru_num_layers, batch_size, model.gru_hidden_size)

def test_streaming_step_vs_batch_equivalence():
    """
    CRITICAL TEST: Verify that sequential step-by-step inference matches
    full-sequence batch forward inference down to numerical precision (< 1e-5).
    """
    model = ComplexCRN(num_bins=257)
    model.eval()

    time_steps = 12
    num_bins = 257
    x = torch.randn(1, 2, time_steps, num_bins)

    # 1. Batch execution
    with torch.no_grad():
        s_hat_batch, mask_batch, _ = model(x)

    # 2. Streaming execution (one frame at a time)
    hidden = model.init_hidden(1)
    s_hat_step_list = []
    mask_step_list = []

    with torch.no_grad():
        for t in range(time_steps):
            frame_t = x[:, :, t, :]  # Shape (1, 2, 257)
            enhanced_t, mask_t, hidden = model.step(frame_t, hidden)
            s_hat_step_list.append(enhanced_t)
            mask_step_list.append(mask_t)

    s_hat_stream = torch.stack(s_hat_step_list, dim=2)  # (1, 2, T, 257)
    mask_stream = torch.stack(mask_step_list, dim=2)

    # 3. Assert exact mathematical equivalence
    max_diff_mask = torch.max(torch.abs(mask_stream - mask_batch)).item()
    max_diff_shat = torch.max(torch.abs(s_hat_stream - s_hat_batch)).item()

    assert max_diff_mask < 1e-5, f"Mask streaming discrepancy: {max_diff_mask}"
    assert max_diff_shat < 1e-5, f"S_hat streaming discrepancy: {max_diff_shat}"

def test_si_snr_loss():
    """Verify SI-SNR loss computation and sensitivity."""
    loss_fn = SISNRLoss()

    # Clean signal
    s = torch.randn(2, 16000)
    # Perfect estimate
    loss_perfect = loss_fn(s, s)
    assert loss_perfect.item() < -30.0, "Perfect estimate should yield very high SNR (very negative loss)"

    # Corrupted estimate with noise
    s_noisy = s + 0.5 * torch.randn_like(s)
    loss_noisy = loss_fn(s_noisy, s)
    assert loss_noisy.item() > loss_perfect.item(), "Noisy signal must have higher loss than clean"

def test_composite_loss_backward():
    """Verify composite loss gradient propagation through model."""
    model = ComplexCRN(num_bins=257)
    model.train()

    criterion = CompositeEnhancementLoss(lambda_sisnr=1.0, lambda_stft=1.0, lambda_complex=1.0)

    x = torch.randn(2, 2, 6, 257, requires_grad=True)
    target_spec = torch.randn(2, 2, 6, 257)

    s_hat, mask, _ = model(x)
    loss, metrics = criterion(s_hat, target_spec)

    assert "loss_stft" in metrics
    assert "loss_complex" in metrics
    assert loss.item() > 0.0

    # Backpropagation
    loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

def test_audio_mixer_and_disjoint_splits():
    """Verify synthetic dataset mixing and disjoint splitting."""
    mixer = AudioMixer(sample_rate=16000, snr_range_db=(-5.0, 15.0), noise_floor_db=-60.0)

    speech = np.sin(2.0 * np.pi * 400.0 * np.arange(16000) / 16000.0).astype(np.float32)
    noise = np.random.randn(16000).astype(np.float32)

    result = mixer.mix(speech, noise, target_snr_db=6.0)

    assert len(result.noisy) == 16000
    assert result.target_snr_db == 6.0
    assert np.all(np.isfinite(result.noisy))

    # Test Speaker-Disjoint and Noise-Disjoint Partitioning
    speakers = [f"spk_{i:03d}" for i in range(50)]
    noises = [f"noise_{i:03d}" for i in range(30)]

    manifest = DatasetSplitter.create_split_manifest(speakers, noises, train_ratio=0.8, val_ratio=0.1)

    train_spk = set(manifest["train"]["speakers"])
    val_spk = set(manifest["val"]["speakers"])
    test_spk = set(manifest["test"]["speakers"])

    assert train_spk.isdisjoint(val_spk)
    assert train_spk.isdisjoint(test_spk)
    assert val_spk.isdisjoint(test_spk)

    train_noi = set(manifest["train"]["noises"])
    val_noi = set(manifest["val"]["noises"])
    test_noi = set(manifest["test"]["noises"])

    assert train_noi.isdisjoint(val_noi)
    assert train_noi.isdisjoint(test_noi)
    assert val_noi.isdisjoint(test_noi)
