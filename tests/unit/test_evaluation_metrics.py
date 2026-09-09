"""
Unit tests for Real Evaluation Metrics, Live Noise-Floor SNR Tracker, and Offline Evaluator.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

from ai.evaluation.metrics import (
    compute_true_snr,
    compute_si_sdr,
    compute_stoi,
    compute_pesq,
    evaluate_metrics,
    LiveNoiseFloorTracker,
    _HAS_PYSTOI,
    _HAS_PESQ
)
from tools.evaluate import evaluate_triple


def test_true_snr_exact_formula():
    """
    Verify exact mathematical formula:
      SNR_in  = 10*log10( Σs^2 / (Σ(x-s)^2 + eps) )
      SNR_out = 10*log10( Σs^2 / (Σ(s_hat-s)^2 + eps) )
      ΔSNR = SNR_out - SNR_in
    """
    n_samples = 16000
    t = np.linspace(0, 1.0, n_samples, endpoint=False)
    # Clean speech tone: power = 0.5 * 1.0^2 = 0.5
    clean = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    # Input noise with variance 0.01 (amplitude std=0.1) -> theoretical SNR ~ 0.5 / 0.01 = 50 (16.99 dB)
    np.random.seed(42)
    noise_in = np.random.normal(0, 0.1, n_samples).astype(np.float32)
    noisy = clean + noise_in

    # Enhanced output has attenuated noise (attenuated by factor of 10 in power = ~10 dB improvement)
    noise_out = noise_in * np.sqrt(0.1)
    enhanced = clean + noise_out

    res = compute_true_snr(clean, noisy, enhanced)

    # Validate mathematical equivalence
    expected_snr_in = 10.0 * np.log10(np.sum(clean**2) / np.sum((noisy - clean)**2))
    expected_snr_out = 10.0 * np.log10(np.sum(clean**2) / np.sum((enhanced - clean)**2))
    expected_delta = expected_snr_out - expected_snr_in

    assert pytest.approx(expected_snr_in, rel=1e-4) == res["snr_in"]
    assert pytest.approx(expected_snr_out, rel=1e-4) == res["snr_out"]
    assert pytest.approx(expected_delta, rel=1e-4) == res["delta_snr"]
    # Net improvement should be ~10 dB
    assert 9.0 < res["delta_snr"] < 11.0


def test_si_sdr_scale_invariance():
    """Verify Scale-Invariant SDR is invariant to arbitrary linear scaling factors."""
    n = 16000
    np.random.seed(123)
    clean = np.random.normal(0, 1.0, n).astype(np.float32)
    # Perfect reconstruction scaled by 5.5
    scaled = clean * 5.5

    sdr = compute_si_sdr(clean, scaled)
    # Scale-invariant projection should yield > 80 dB (near infinite target-to-residual ratio)
    assert sdr > 80.0

    # Add noise
    noisy = clean + np.random.normal(0, 0.2, n).astype(np.float32)
    sdr_noisy = compute_si_sdr(clean, noisy)
    assert 10.0 < sdr_noisy < 20.0


def test_stoi_calculation():
    """Verify STOI behaves accurately and doesn't invent fake scores."""
    if not _HAS_PYSTOI:
        pytest.skip("pystoi not installed")

    n = 16000
    t = np.linspace(0, 1.0, n, endpoint=False)
    clean = np.sin(2 * np.pi * 300.0 * t).astype(np.float32)

    # Identical clean signal should have STOI ~ 1.0
    perfect_stoi = compute_stoi(clean, clean, sample_rate=16000)
    assert perfect_stoi is not None
    assert perfect_stoi >= 0.99

    # Uncorrelated random white noise should have degraded STOI
    noise = np.random.normal(0, 0.5, n).astype(np.float32)
    bad_stoi = compute_stoi(clean, noise, sample_rate=16000)
    assert bad_stoi is not None
    assert bad_stoi < 0.40


def test_pesq_graceful_handling():
    """Verify PESQ returns None when unavailable or valid score when installed."""
    n = 16000
    clean = np.zeros(n, dtype=np.float32)
    val = compute_pesq(clean, clean, sample_rate=16000)

    if _HAS_PESQ:
        assert val is not None
    else:
        # Strict zero-mock: never invent a score
        assert val is None


def test_live_noise_floor_tracker():
    """Verify LiveNoiseFloorTracker estimates noise floor during non-speech and computes estimated SNR."""
    tracker = LiveNoiseFloorTracker(sample_rate=16000, vad_threshold=0.30)
    hop_len = 80

    # Phase 1: 50 hops of stationary noise during non-speech frames (vad_prob = 0.05)
    np.random.seed(42)
    noise_sigma = 0.05
    for _ in range(50):
        noise_frame = np.random.normal(0, noise_sigma, hop_len).astype(np.float32)
        out = tracker.update(
            primary_samples=noise_frame,
            output_samples=noise_frame * 0.5,  # 6 dB attenuated
            vad_prob=0.05,
            reference_samples=noise_frame
        )

    # Noise floor should have adapted close to noise_sigma^2 = 0.0025
    expected_var = noise_sigma**2
    assert 0.0005 < tracker.p_noise_in < 0.01
    assert out["snr_is_estimated"] is True

    # Phase 2: Speech burst frames (vad_prob = 0.90) with high speech energy
    t = np.linspace(0, 0.005, hop_len, endpoint=False)
    speech = np.sin(2 * np.pi * 500.0 * t).astype(np.float32) * 0.5

    for _ in range(20):
        noisy_speech = speech + np.random.normal(0, noise_sigma, hop_len).astype(np.float32)
        enhanced_speech = speech + np.random.normal(0, noise_sigma * 0.3, hop_len).astype(np.float32)
        out = tracker.update(
            primary_samples=noisy_speech,
            output_samples=enhanced_speech,
            vad_prob=0.90,
            reference_samples=np.random.normal(0, noise_sigma, hop_len).astype(np.float32)
        )

    # During speech with high signal energy, estimated SNR should be strongly positive
    assert out["estimated_input_snr_db"] > 5.0
    assert out["estimated_output_snr_db"] > out["estimated_input_snr_db"]
    assert out["estimated_snr_improvement_db"] > 0.0
    assert -96.0 <= out["primary_level_dbfs"] <= 0.0


def test_evaluate_tool_triple():
    """Verify tools/evaluate.py evaluate_triple on real wav files."""
    sr = 16000
    n = 16000
    t = np.linspace(0, 1.0, n, endpoint=False)
    # Multi-harmonic broadband speech formant signal spanning standard STOI octave bands
    clean = sum(0.3 * np.sin(2 * np.pi * f * t) for f in [250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0]).astype(np.float32)
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, n).astype(np.float32)
    noisy = clean + noise
    enhanced = clean + noise * 0.1

    with tempfile.TemporaryDirectory() as tmpdir:
        c_path = os.path.join(tmpdir, "clean.wav")
        n_path = os.path.join(tmpdir, "noisy.wav")
        e_path = os.path.join(tmpdir, "enhanced.wav")

        sf.write(c_path, clean, sr)
        sf.write(n_path, noisy, sr)
        sf.write(e_path, enhanced, sr)

        res = evaluate_triple(c_path, n_path, e_path, target_sr=sr)

        assert res["clean_file"] == "clean.wav"
        assert res["duration_sec"] == 1.0
        assert res["snr_in_db"] > 0.0
        assert res["snr_out_db"] > res["snr_in_db"]
        assert res["delta_snr_db"] > 5.0
        assert res["si_sdr_out_db"] > res["si_sdr_in_db"]
        if _HAS_PYSTOI:
            assert res["stoi_enhanced"] is not None
            assert res["stoi_noisy"] is not None
            assert res["stoi_enhanced"] >= res["stoi_noisy"]
            assert res["delta_stoi"] >= 0.0
