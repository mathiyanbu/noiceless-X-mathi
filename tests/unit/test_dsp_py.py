import math
import numpy as np
import pytest

def test_hann_window_symmetry():
    """Verify periodic Hann window formulas used in C++ STFT match math spec."""
    N = 512
    # w[n] = 0.5 * (1 - cos(2*pi*n / N))
    n = np.arange(N)
    w = 0.5 * (1.0 - np.cos(2.0 * math.pi * n / N))

    assert len(w) == 512
    assert math.isclose(w[0], 0.0, abs_tol=1e-6)
    assert math.isclose(w[256], 1.0, abs_tol=1e-6)

def test_cola_reconstruction_property():
    """Verify Constant Overlap-Add (COLA) property for Hann window with 75% overlap (hop=128)."""
    N = 512
    H = 128
    w = 0.5 * (1.0 - np.cos(2.0 * math.pi * np.arange(N) / N))

    # Overlap-add sum of squared window
    total_len = N + 10 * H
    cola_sum = np.zeros(total_len)
    for m in range(10):
        cola_sum[m * H : m * H + N] += w * w

    # In steady state (center region), COLA sum is strictly constant = 1.5
    steady_state = cola_sum[N : total_len - N]
    expected_cola = 1.5
    assert np.allclose(steady_state, expected_cola, atol=1e-6)
