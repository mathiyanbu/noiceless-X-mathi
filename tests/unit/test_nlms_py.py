import math
import numpy as np
import pytest

def test_nlms_python_convergence():
    """Verify standard NLMS equations converge on correlated noise in Python."""
    N = 3000
    L = 32
    mu = 0.1
    eps = 1e-6

    # True impulse response
    h_true = np.array([0.6, -0.4, 0.2, -0.1] + [0.0] * (L - 4))

    # Reference noise
    np.random.seed(42)
    ref = np.random.randn(N)

    # Primary received noise
    primary_noise = np.convolve(ref, h_true, mode='full')[:N]

    # Clean speech tone
    speech = 0.3 * np.sin(2.0 * math.pi * 400.0 * np.arange(N) / 16000.0)
    primary = speech + primary_noise

    # NLMS adaptive filter
    weights = np.zeros(L)
    history = np.zeros(L)
    err = np.zeros(N)

    for n in range(N):
        history = np.roll(history, 1)
        history[0] = ref[n]

        n_hat = np.dot(weights, history)
        e = primary[n] - n_hat
        err[n] = e

        norm_sq = np.dot(history, history)
        step = (mu * e) / (eps + norm_sq)
        weights += step * history

    # Steady state residual error (last 1000 samples) relative to speech
    residual_noise = err[-1000:] - speech[-1000:]
    noise_suppression_db = 10.0 * math.log10(np.mean(primary_noise[-1000:]**2) / np.mean(residual_noise**2))

    assert noise_suppression_db > 15.0, f"Expected >15dB noise suppression, got {noise_suppression_db:.1f}dB"


def test_nlms_speech_preservation_decorrelated_reference():
    """Verify that decorrelated noise at reference does NOT cancel desired speech."""
    N = 3200
    L = 32
    mu = 0.05
    eps = 1e-6

    # Pure speech tone on primary
    np.random.seed(42)
    speech = 0.4 * np.sin(2.0 * math.pi * 500.0 * np.arange(N) / 16000.0)
    primary = speech.copy()

    # Uncorrelated noise at reference
    ref = np.random.normal(0.0, 0.2, N)

    weights = np.zeros(L)
    history = np.zeros(L)
    err = np.zeros(N)

    for n in range(N):
        history = np.roll(history, 1)
        history[0] = ref[n]

        n_hat = np.dot(weights, history)
        e = primary[n] - n_hat
        err[n] = e

        norm_sq = np.dot(history, history)
        step = (mu * e) / (eps + norm_sq)
        weights += step * history

    # Steady state speech preservation (last 1600 samples)
    speech_power = np.mean(speech[-1600:] ** 2)
    output_power = np.mean(err[-1600:] ** 2)
    power_ratio = output_power / speech_power

    # Preserved ratio must exceed 90% (no aggressive speech cancellation)
    assert power_ratio > 0.90, f"Speech was over-canceled: preserved ratio {power_ratio:.3f}"
    # Weights should remain small
    assert np.max(np.abs(weights)) < 0.15


def test_nlms_zero_reference_input():
    """Verify that zero reference input preserves primary signal exactly with zero drift."""
    N = 500
    L = 16
    mu = 0.1
    eps = 1e-6

    primary = np.sin(2.0 * math.pi * 300.0 * np.arange(N) / 16000.0)
    ref = np.zeros(N)

    weights = np.zeros(L)
    history = np.zeros(L)
    err = np.zeros(N)

    for n in range(N):
        history = np.roll(history, 1)
        history[0] = ref[n]
        n_hat = np.dot(weights, history)
        e = primary[n] - n_hat
        err[n] = e

        norm_sq = np.dot(history, history)
        step = (mu * e) / (eps + norm_sq)
        weights += step * history

    assert np.allclose(err, primary, atol=1e-7)
    assert np.all(weights == 0.0)


def test_nlms_step_size_normalization_behavior():
    """Verify that scaling input amplitude by 100x does not cause filter divergence."""
    N = 2000
    L = 16
    mu = 0.1
    eps = 1e-6

    h_true = np.array([0.5, -0.3, 0.1] + [0.0] * (L - 3))
    np.random.seed(123)

    # Large amplitude input
    ref = 50.0 * np.random.randn(N)
    primary = np.convolve(ref, h_true, mode='full')[:N]

    weights = np.zeros(L)
    history = np.zeros(L)
    err = np.zeros(N)

    for n in range(N):
        history = np.roll(history, 1)
        history[0] = ref[n]
        n_hat = np.dot(weights, history)
        e = primary[n] - n_hat
        err[n] = e

        norm_sq = np.dot(history, history)
        step = (mu * e) / (eps + norm_sq)
        weights += step * history

    assert np.all(np.isfinite(weights)), "Weights became NaN/Inf under high input amplitude!"
    assert np.all(np.isfinite(err)), "Errors became NaN/Inf under high input amplitude!"
    # Steady state residual error should be significantly smaller than initial
    initial_p = np.mean(primary[:200] ** 2)
    final_p = np.mean(err[-200:] ** 2)
    assert final_p < initial_p * 0.05

