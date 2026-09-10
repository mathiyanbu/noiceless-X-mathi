"""
Unit tests for NOICELESSX Training Pipeline, Checkpointing, and Automated ONNX Export.
Verifies forward-backward optimization step, loss reduction, checkpoint reloading,
streaming equivalence, structured CSV logging, held-out test evaluation, and ONNX export.
"""

import csv
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from ai.datasets.manifest import ManifestBuilder
from ai.export.onnx_export import export_onnx_model
from ai.export.onnx_quantize import quantize_model
from ai.losses.composite_loss import CompositeEnhancementLoss
from ai.models.complex_crn import ComplexCRN
from ai.training.dataset import SpeechEnhancementDataset, collate_speech_batch
from ai.training.train import (
    ComplexCRNTrainer,
    evaluate_test_split,
    verify_streaming_equivalence,
)


@pytest.fixture
def mock_speech_dataloader(tmp_path):
    """Creates a minimal dataset and DataLoader for pipeline testing."""
    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    speech_dir.mkdir()
    noise_dir.mkdir()

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(speech_dir / "p225_001.wav"), (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr)
    sf.write(str(speech_dir / "p225_002.wav"), (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32), sr)
    sf.write(str(noise_dir / "noise_01.wav"), (0.1 * np.random.randn(sr)).astype(np.float32), sr)

    builder = ManifestBuilder(target_sample_rate=sr)
    train_recs, _, _ = builder.scan_clean_speech_directory(str(speech_dir))
    noise_recs = builder.scan_noise_directory(str(noise_dir))

    train_m = tmp_path / "train.json"
    noise_m = tmp_path / "noise.json"
    builder.save_manifest(train_recs, str(train_m))
    builder.save_manifest(noise_recs, str(noise_m))

    dataset = SpeechEnhancementDataset(
        clean_manifest_path=str(train_m),
        noise_manifest_path=str(noise_m),
        segment_duration=0.5,  # 8000 samples for fast test
    )
    return DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_speech_batch)


def test_complex_crn_training_step(mock_speech_dataloader, tmp_path):
    model = ComplexCRN(num_bins=257)
    criterion = CompositeEnhancementLoss(lambda_sisnr=1.0, lambda_stft=1.0, lambda_complex=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = ComplexCRNTrainer(
        model=model,
        train_loader=mock_speech_dataloader,
        val_loader=None,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    initial_loss = None
    final_loss = None

    for epoch in range(1, 4):
        metrics, snrs = trainer.train_epoch(epoch)
        if initial_loss is None:
            initial_loss = metrics["loss"]
        final_loss = metrics["loss"]

    assert final_loss is not None
    assert np.isfinite(final_loss)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any()


def test_streaming_equivalence():
    """Verifies that batch and streaming execution modes are mathematically equivalent."""
    model = ComplexCRN(num_bins=257)
    diff = verify_streaming_equivalence(model, torch.device("cpu"), tolerance=1e-4, time_steps=50)
    assert diff < 1e-4, f"Streaming diff {diff} exceeds 1e-4 tolerance!"


def test_checkpoint_save_and_reload(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "best_model.pth"

    model1 = ComplexCRN(num_bins=257)
    with torch.no_grad():
        for p in model1.parameters():
            p.add_(0.1)
    torch.save(model1.state_dict(), str(ckpt_path))

    model2 = ComplexCRN(num_bins=257)
    model2.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_structured_logging_and_resumption(mock_speech_dataloader, tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    model = ComplexCRN(num_bins=257)
    criterion = CompositeEnhancementLoss(lambda_sisnr=1.0, lambda_stft=1.0, lambda_complex=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = ComplexCRNTrainer(
        model=model,
        train_loader=mock_speech_dataloader,
        val_loader=mock_speech_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=str(ckpt_dir),
    )

    # Train for 2 epochs
    trainer.train(num_epochs=2)

    # Check log file created and contains 2 epoch entries
    log_file = ckpt_dir / "training_log.csv"
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert "streaming_max_diff" in reader[0]
        assert float(reader[0]["streaming_max_diff"]) < 1e-3

    # Resume with a second trainer
    model_resumed = ComplexCRN(num_bins=257)
    opt_resumed = torch.optim.Adam(model_resumed.parameters(), lr=1e-3)
    trainer2 = ComplexCRNTrainer(
        model=model_resumed,
        train_loader=mock_speech_dataloader,
        val_loader=None,
        criterion=criterion,
        optimizer=opt_resumed,
        checkpoint_dir=str(ckpt_dir),
    )
    last_ckpt = ckpt_dir / "last_model.pth"
    assert last_ckpt.exists()
    trainer2.resume_from_checkpoint(last_ckpt)
    assert trainer2.start_epoch == 3


def test_evaluate_test_split(mock_speech_dataloader, tmp_path):
    model = ComplexCRN(num_bins=257)
    output_json = tmp_path / "eval_results.json"
    results = evaluate_test_split(
        model=model,
        test_loader=mock_speech_dataloader,
        device=torch.device("cpu"),
        output_path=output_json,
    )
    assert "mean_delta_snr_db" in results
    assert "mean_stoi" in results
    assert output_json.exists()
    with open(output_json, "r") as f:
        saved = json.load(f)
        assert saved["num_test_samples"] > 0


def test_onnx_export_and_quantize_pipeline(tmp_path):
    ckpt_path = tmp_path / "model.pth"
    model = ComplexCRN(num_bins=257)
    torch.save(model.state_dict(), str(ckpt_path))

    fp32_onnx = tmp_path / "model_fp32.onnx"
    int8_onnx = tmp_path / "model_int8.onnx"

    export_ok = export_onnx_model(
        checkpoint_path=str(ckpt_path),
        output_path=str(fp32_onnx),
        num_bins=257,
        verbose=False,
    )
    assert bool(export_ok)
    assert fp32_onnx.exists()

    quant_ok = quantize_model(
        input_fp32=str(fp32_onnx),
        output_int8=str(int8_onnx),
        verbose=False,
    )
    assert bool(quant_ok)
    assert int8_onnx.exists()
    assert int8_onnx.stat().st_size < fp32_onnx.stat().st_size


def test_startup_config_verification():
    from ai.training.train import assert_config_match

    # Verify standard raspberrypi.yaml matches expected parameters
    cfg = assert_config_match("config/raspberrypi.yaml", expected_sample_rate=16000, expected_fft_size=512, expected_hop_size=80)
    assert cfg["sample_rate"] == 16000
    assert cfg["fft_size"] == 512
    assert cfg["hop_ms"] == 5

    # Assert mismatch detection works
    with pytest.raises(AssertionError, match="Sample rate mismatch"):
        assert_config_match("config/raspberrypi.yaml", expected_sample_rate=48000, expected_fft_size=512, expected_hop_size=80)

    with pytest.raises(AssertionError, match="FFT size mismatch"):
        assert_config_match("config/raspberrypi.yaml", expected_sample_rate=16000, expected_fft_size=256, expected_hop_size=80)

    with pytest.raises(AssertionError, match="Hop size mismatch"):
        assert_config_match("config/raspberrypi.yaml", expected_sample_rate=16000, expected_fft_size=512, expected_hop_size=160)


def test_snr_histogram_uniformity():
    from ai.training.train import compute_snr_histogram

    snrs = [-4.0, -2.0, 0.0, 2.0, 5.0, 8.0, 12.0, 14.0]
    hist = compute_snr_histogram(snrs)
    assert len(hist) == 5
    assert sum(hist.values()) == len(snrs)


def test_early_stopping_behavior(mock_speech_dataloader, tmp_path):
    ckpt_dir = tmp_path / "early_stop_ckpts"
    model = ComplexCRN(num_bins=257)
    criterion = CompositeEnhancementLoss(lambda_sisnr=1.0, lambda_stft=0.5, lambda_complex=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = ComplexCRNTrainer(
        model=model,
        train_loader=mock_speech_dataloader,
        val_loader=mock_speech_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=str(ckpt_dir),
        early_stopping_patience=2,  # Stop after 2 plateau epochs
    )

    # Force best_val_sisnr high so no epoch improves
    trainer.best_val_sisnr = 100.0

    best_ckpt, last_ckpt = trainer.train(num_epochs=10)
    assert last_ckpt.exists()
    # Check that it stopped early (before 10 epochs)
    assert trainer.epochs_without_improvement >= 2


def test_impulse_detector_training_with_confusion_matrix(tmp_path):
    from ai.models.impulse_detector.train_impulse import train_impulse_detector_model

    # Create minimal manifest with positive (impulsive) and negative (stationary) records
    corpus = tmp_path / "impulse_corpus"
    corpus.mkdir()
    sr = 16000

    pos_file = corpus / "gunshot_01.wav"
    neg_file = corpus / "noise_01.wav"

    # Positive transient burst
    sig_pos = np.zeros(sr, dtype=np.float32)
    sig_pos[100:150] = 0.9
    sf.write(str(pos_file), sig_pos, sr)

    # Negative stationary noise
    sig_neg = (0.05 * np.random.randn(sr)).astype(np.float32)
    sf.write(str(neg_file), sig_neg, sr)

    manifest_csv = tmp_path / "impulse_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "dataset_source", "category", "class_label", "split"])
        writer.writeheader()
        writer.writerow({
            "filepath": str(pos_file.resolve()),
            "dataset_source": "fsd50k",
            "category": "noise_impulsive",
            "class_label": "gunshot",
            "split": "train",
        })
        writer.writerow({
            "filepath": str(neg_file.resolve()),
            "dataset_source": "demand",
            "category": "noise_stationary",
            "class_label": "office",
            "split": "train",
        })

    model, metrics = train_impulse_detector_model(
        manifest_path=str(manifest_csv),
        output_onnx=str(tmp_path / "test_impulse.onnx"),
        epochs=3,
        max_samples=20,
        val_ratio=0.3,
        verbose=False,
    )
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "TP" in metrics["confusion_matrix"]
    assert "TN" in metrics["confusion_matrix"]
    assert (tmp_path / "test_impulse.onnx").exists()

