"""
Unit tests for Comprehensive Verification Suite (ai/evaluation/full_verification.py).
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from ai.evaluation.full_verification import (
    enhance_waveform,
    load_checkpoint_model,
    verify_dataset_disjointness_gate,
    verify_realtime_feasibility_check,
    verify_streaming_equivalence_gate,
)
from ai.models.complex_crn import ComplexCRN

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_load_checkpoint_model():
    """Verifies that trained checkpoint loads correctly into ComplexCRN."""
    ckpt_path = REPO_ROOT / "models" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        pytest.skip("best_model.pth not found; run training first.")

    model, ckpt_meta = load_checkpoint_model(ckpt_path)
    assert isinstance(model, ComplexCRN)
    assert "model_state_dict" in ckpt_meta or "epoch" in ckpt_meta
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1_000_000


def test_enhance_waveform_shape_and_values():
    """Verifies that enhance_waveform outputs valid finite audio matching input length."""
    model = ComplexCRN()
    model.eval()
    sr = 16000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)
    sig = 0.3 * np.sin(2 * np.pi * 440.0 * t)

    enh = enhance_waveform(model, sig)
    assert enh.shape == sig.shape
    assert np.all(np.isfinite(enh))
    assert enh.dtype == np.float32


def test_generalization_disjointness_gate():
    """Verifies that dataset manifest passes the zero-leakage generalization gate."""
    manifest_path = REPO_ROOT / "data" / "manifests" / "manifest.csv"
    if not manifest_path.exists():
        pytest.skip("manifest.csv not found")

    import csv
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    res = verify_dataset_disjointness_gate(records)
    assert res["gate_passed"] is True
    assert res["speaker_leakage_count"] == 0
    assert res["noise_leakage_count"] == 0
    assert res["rir_leakage_count"] == 0
    assert len(res["test_speakers"]) > 0


def test_streaming_equivalence_gate():
    """Verifies that streaming equivalence gate passes with tolerance < 1e-3."""
    ckpt_path = REPO_ROOT / "models" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        pytest.skip("best_model.pth not found")

    model, _ = load_checkpoint_model(ckpt_path)
    res = verify_streaming_equivalence_gate(model, device=torch.device("cpu"), tolerance=1e-3, num_frames=10)
    assert res["gate_passed"] is True
    assert res["max_absolute_diff"] < 1e-3


def test_realtime_feasibility_check():
    """Verifies that FP32 streaming forward execution meets the 5.0ms real-time deadline."""
    ckpt_path = REPO_ROOT / "models" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        pytest.skip("best_model.pth not found")

    model, _ = load_checkpoint_model(ckpt_path)
    res = verify_realtime_feasibility_check(model, device=torch.device("cpu"), warmup_frames=10, benchmark_frames=50, hop_ms=5.0)
    assert res["mean_latency_ms"] < 5.0
    assert res["real_time_factor"] < 1.0
    assert res["realtime_feasible"] is True
    assert res["throughput_fps"] > 200.0


def test_verification_report_files_exist():
    """Verifies that VERIFICATION_REPORT.md and verification_results.json exist and contain all 8 sections."""
    rep_path = REPO_ROOT / "models" / "VERIFICATION_REPORT.md"
    json_path = REPO_ROOT / "models" / "verification_results.json"

    assert rep_path.exists(), "models/VERIFICATION_REPORT.md must exist."
    assert json_path.exists(), "models/verification_results.json must exist."

    with open(rep_path, "r", encoding="utf-8") as f:
        md = f.read()

    assert "## 1. Standard Benchmark" in md
    assert "## 2. Per-Noise-Category Breakdown" in md
    assert "## 3. SNR Sweep" in md
    assert "## 4. Reverb Condition Check" in md
    assert "## 5. Unseen-Speaker & Unseen-Noise Generalization Gate" in md
    assert "## 6. Streaming vs. Batch Equivalence Gate" in md
    assert "## 7. Impulse Detector Verification" in md
    assert "## 8. Real-Time Feasibility Check" in md

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in [
        "standard_benchmark",
        "category_breakdown",
        "snr_sweep",
        "reverb_condition",
        "generalization_gate",
        "streaming_gate",
        "impulse_suite",
        "realtime_feasibility",
    ]:
        assert key in data, f"Missing key '{key}' in verification_results.json"
