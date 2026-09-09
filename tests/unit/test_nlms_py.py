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
