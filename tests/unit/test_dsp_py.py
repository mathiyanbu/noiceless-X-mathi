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


class PythonStftEngine:
    """Python reference implementation of the C++ WOLA StftEngine."""
    def __init__(self, fft_size: int = 512, hop_size: int = 128):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.num_bins = fft_size // 2 + 1
        n = np.arange(fft_size)
        self.window = (0.5 * (1.0 - np.cos(2.0 * np.pi * n / fft_size))).astype(np.float32)

        # Precompute steady-state WOLA normalization sum
        reps = 16
        ola_norm = np.zeros(fft_size + reps * hop_size, dtype=np.float32)
        for m in range(reps):
            ola_norm[m * hop_size : m * hop_size + fft_size] += self.window * self.window
        self.cola_factor = float(np.mean(ola_norm[fft_size : (reps - 2) * hop_size]))

        self.input_fifo = np.zeros(fft_size, dtype=np.float32)
        self.ola_buffer = np.zeros(fft_size + hop_size * 4, dtype=np.float32)

    def analyze(self, hop_in: np.ndarray) -> np.ndarray:
        # Shift in new samples
        self.input_fifo[:-self.hop_size] = self.input_fifo[self.hop_size:]
        self.input_fifo[-self.hop_size:] = hop_in
        windowed = self.input_fifo * self.window
        return np.fft.rfft(windowed, n=self.fft_size)

    def synthesize(self, spec: np.ndarray) -> np.ndarray:
        time_frame = np.fft.irfft(spec, n=self.fft_size).astype(np.float32)
        # Apply synthesis window and normalize by WOLA constant
        windowed_out = (time_frame * self.window) / self.cola_factor

        self.ola_buffer[:self.fft_size] += windowed_out
        out = self.ola_buffer[:self.hop_size].copy()

        # Shift OLA buffer
        self.ola_buffer[:-self.hop_size] = self.ola_buffer[self.hop_size:]
        self.ola_buffer[-self.hop_size:] = 0.0
        return out


@pytest.mark.parametrize("hop_size", [128, 64])
def test_stft_istft_perfect_reconstruction_snr(hop_size):
    """Verify STFT -> iSTFT achieves reconstruction error < -80 dB (SNR > 80 dB)."""
    fft_size = 512
    sample_rate = 16000
    engine = PythonStftEngine(fft_size=fft_size, hop_size=hop_size)

    # Multi-tone test signal: 250Hz, 880Hz, 2400Hz, 6000Hz
    num_hops = 60
    total_samples = num_hops * hop_size
    t = np.arange(total_samples) / sample_rate
    test_sig = (
        0.3 * np.sin(2 * np.pi * 250.0 * t) +
        0.25 * np.cos(2 * np.pi * 880.0 * t) +
        0.15 * np.sin(2 * np.pi * 2400.0 * t) +
        0.1 * np.cos(2 * np.pi * 6000.0 * t)
    ).astype(np.float32)

    reconstructed = np.zeros_like(test_sig)
    for h in range(num_hops):
        chunk = test_sig[h * hop_size : (h + 1) * hop_size]
        spec = engine.analyze(chunk)
        # Identity pass-through
        rec_chunk = engine.synthesize(spec)
        reconstructed[h * hop_size : (h + 1) * hop_size] = rec_chunk

    # Evaluate reconstruction error in steady-state region
    latency = fft_size - hop_size
    warmup = fft_size * 2
    eval_start = warmup
    eval_end = total_samples - hop_size * 2

    ref = test_sig[eval_start - latency : eval_end - latency]
    rec = reconstructed[eval_start : eval_end]

    err = rec - ref
    err_power = np.mean(err ** 2)
    ref_power = np.mean(ref ** 2)
    snr_db = 10.0 * np.log10(ref_power / (err_power + 1e-15))
    err_db = -snr_db

    assert err_db < -80.0, f"STFT/iSTFT reconstruction error {err_db:.2f} dB exceeds -80 dB threshold!"


def test_stft_silence_input():
    """Verify strictly all-zero input produces strictly zero output."""
    engine = PythonStftEngine(fft_size=512, hop_size=128)
    zeros = np.zeros(128, dtype=np.float32)
    for _ in range(10):
        spec = engine.analyze(zeros)
        rec = engine.synthesize(spec)
        assert np.all(rec == 0.0)


def test_stft_dirac_impulse_reconstruction():
    """Verify reconstruction of a sparse Dirac impulse train."""
    fft_size = 512
    hop_size = 128
    engine = PythonStftEngine(fft_size=fft_size, hop_size=hop_size)

    total_samples = 40 * hop_size
    signal = np.zeros(total_samples, dtype=np.float32)
    # Impulses spaced apart
    signal[1000] = 1.0
    signal[2500] = -0.8
    signal[3800] = 0.6

    reconstructed = np.zeros_like(signal)
    for h in range(40):
        chunk = signal[h * hop_size : (h + 1) * hop_size]
        spec = engine.analyze(chunk)
        rec_chunk = engine.synthesize(spec)
        reconstructed[h * hop_size : (h + 1) * hop_size] = rec_chunk

    latency = fft_size - hop_size
    ref = signal[1000]
    rec = reconstructed[1000 + latency]
    assert math.isclose(rec, ref, abs_tol=1e-4)


def test_stft_nyquist_and_dc_bins():
    """Verify DC (bin 0) and Nyquist (bin N/2) are strictly real-valued for real inputs."""
    engine = PythonStftEngine(fft_size=512, hop_size=128)
    np.random.seed(42)
    random_hop = np.random.randn(128).astype(np.float32)
    spec = engine.analyze(random_hop)

    assert len(spec) == 257
    # Bin 0 (DC) and Bin 256 (Nyquist) must have zero imaginary components
    assert math.isclose(spec[0].imag, 0.0, abs_tol=1e-6)
    assert math.isclose(spec[256].imag, 0.0, abs_tol=1e-6)

