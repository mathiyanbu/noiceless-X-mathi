import numpy as np
import pytest
from pathlib import Path
import yaml
import onnxruntime as ort
import torch

from ai.models.impulse_detector.features import extract_impulse_features
from ai.models.impulse_detector.dataset import (
    generate_synthetic_impulse_frame,
    generate_synthetic_non_impulse_frame
)
from ai.models.impulse_detector.model import TinyImpulseMLP

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def test_feature_extraction_shapes_and_values():
    """Verifies that extract_impulse_features extracts 8 finite, non-negative features."""
    fft_size = 512
    t = np.linspace(0, 1, fft_size, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    spec = np.abs(np.fft.rfft(sig * np.hanning(fft_size))).astype(np.float32)

    feats, next_mag = extract_impulse_features(sig, spec)
    assert feats.shape == (8,)
    assert np.all(np.isfinite(feats))
    # RMS, crest factor, ZCR, band energies must all be non-negative
    assert feats[0] > 0.0 # RMS
    assert feats[2] >= 1.0 # Crest factor >= 1.0 for any signal
    assert feats[3] >= 0.0 # ZCR
    for b in range(4):
        assert feats[4 + b] >= 0.0 # Band energies

def test_synthetic_impulsive_vs_steady_separation():
    """
    Verifies that features and logistic classifier clearly separate
    synthetic impulsive audio (clicks, pops, keyboard clatter) from steady-state audio.
    """
    fft_size = 512
    window = np.hanning(fft_size)

    impulsive_probs = []
    steady_probs = []

    # Logistic weights matching default YAML config
    w = np.array([1.2, 0.001, 3.5, -2.0, -0.5, -0.3, 1.0, 3.0], dtype=np.float32)
    bias = -14.0

    # 1. Impulsive frames
    prev_mag = np.zeros(fft_size // 2 + 1, dtype=np.float32)
    for _ in range(50):
        audio = generate_synthetic_impulse_frame(fft_size)
        spec = np.abs(np.fft.rfft(audio * window)).astype(np.float32)
        feats, prev_mag = extract_impulse_features(audio, spec, prev_mag)
        z = float(np.dot(w, feats) + bias)
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        impulsive_probs.append(prob)

    # 2. Steady-state frames
    prev_mag = np.zeros(fft_size // 2 + 1, dtype=np.float32)
    for _ in range(50):
        audio = generate_synthetic_non_impulse_frame(fft_size)
        spec = np.abs(np.fft.rfft(audio * window)).astype(np.float32)
        feats, prev_mag = extract_impulse_features(audio, spec, prev_mag)
        z = float(np.dot(w, feats) + bias)
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        steady_probs.append(prob)

    mean_impulse = np.mean(impulsive_probs)
    mean_steady = np.mean(steady_probs)

    print(f"\nImpulse Detector Separation Test:")
    print(f"  Mean Impulsive Probability:    {mean_impulse:.3f}")
    print(f"  Mean Steady-State Probability: {mean_steady:.3f}")

    assert mean_impulse > 0.60, f"Impulsive probability {mean_impulse:.3f} too low!"
    assert mean_steady < 0.40, f"Steady-state probability {mean_steady:.3f} too high!"
    assert (mean_impulse - mean_steady) > 0.35, "Insufficient separation margin between classes!"

def test_hysteresis_anti_chattering():
    """
    Verifies that dual thresholds (0.65 on, 0.35 off) and asymmetric smoothing
    prevent rapid on/off flapping on borderline fluctuating signals.
    """
    threshold_on = 0.65
    threshold_off = 0.35
    attack_alpha = 0.80
    release_alpha = 0.15

    # Simulate fluctuating raw probabilities oscillating around 0.50
    np.random.seed(123)
    n_steps = 100
    raw_probs = 0.50 + 0.12 * np.sin(np.linspace(0, 8 * np.pi, n_steps)) + 0.05 * np.random.randn(n_steps)
    raw_probs = np.clip(raw_probs, 0.0, 1.0)

    # 1. Without hysteresis (single threshold 0.50): measure toggle count
    naive_toggles = 0
    naive_detected = False
    for p in raw_probs:
        det = (p >= 0.50)
        if det != naive_detected:
            naive_toggles += 1
            naive_detected = det

    # 2. With hysteresis state machine and asymmetric smoothing
    hysteresis_toggles = 0
    detected_latched = False
    smoothed_prob = 0.0
    for p in raw_probs:
        if p > smoothed_prob:
            smoothed_prob = attack_alpha * p + (1.0 - attack_alpha) * smoothed_prob
        else:
            smoothed_prob = release_alpha * p + (1.0 - release_alpha) * smoothed_prob

        prev_det = detected_latched
        if not detected_latched and smoothed_prob >= threshold_on:
            detected_latched = True
        elif detected_latched and smoothed_prob <= threshold_off:
            detected_latched = False

        if detected_latched != prev_det:
            hysteresis_toggles += 1

    print(f"\nHysteresis Flapping Test (100 borderline frames):")
    print(f"  Naive Toggles (Single Threshold):   {naive_toggles}")
    print(f"  Hysteresis Toggles (Dual Threshold): {hysteresis_toggles}")

    # Hysteresis must drastically reduce chattering on borderline signals
    assert hysteresis_toggles <= naive_toggles // 2, "Hysteresis failed to suppress borderline flapping!"

def test_trained_onnx_model_execution():
    """Verifies that exported ONNX model loads and runs inference on 8-D feature vector."""
    onnx_path = REPO_ROOT / "models" / "onnx" / "impulse_detector.onnx"
    if not onnx_path.exists():
        pytest.skip("models/onnx/impulse_detector.onnx not found; run train.py first.")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session is not None

    test_input = np.array([[0.5, 2.0, 3.5, 0.25, 0.1, 0.3, 0.5, 0.8]], dtype=np.float32)
    output = session.run(None, {"features": test_input})[0]

    assert output.shape == (1, 1)
    prob = float(output[0, 0])
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range [0, 1]!"

def test_yaml_config_impulse_schema():
    """Verifies that raspberrypi.yaml and development.yaml contain valid impulse configurations."""
    for cfg_name in ["raspberrypi.yaml", "development.yaml"]:
        cfg_path = REPO_ROOT / "config" / cfg_name
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "impulse" in cfg, f"Missing 'impulse' section in {cfg_name}"
        imp = cfg["impulse"]
        assert imp.get("enabled") is True
        assert "threshold_on" in imp
        assert "threshold_off" in imp
        assert imp["threshold_on"] > imp["threshold_off"]
        assert "attack_alpha" in imp
        assert "release_alpha" in imp
        assert "weights" in imp
        weights = imp["weights"]
        for key in ["rms", "spectral_flux", "crest_factor", "zcr", "band_low", "band_mid_low", "band_mid_high", "band_high", "bias"]:
            assert key in weights, f"Missing weight '{key}' in {cfg_name}"
