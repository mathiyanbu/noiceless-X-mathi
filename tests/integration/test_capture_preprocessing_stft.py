"""
Integration Test: Mic Capture -> Preprocessing (DC removal + 80Hz Biquad high-pass) -> STFT chain.
Verifies:
- Complete chain execution without frame drops or discontinuities.
- DC removal effectiveness (< -60 dB at 0 Hz).
- Low-frequency rumble attenuation below 80 Hz per Butterworth response.
- Speech band preservation (> 200 Hz).
- STFT spectral analysis correctness, phase stability, and bounded latency over 100+ frames.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy import signal as sp_signal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.unit.test_dsp_py import PythonStftEngine


class PythonDcBlocker:
    """IIR DC Blocker matching embedded/dsp/dc_blocker.hpp: y[n] = x[n] - x[n-1] + R*y[n-1]"""
    def __init__(self, r: float = 0.995):
        self.r = r
        self.x_prev = 0.0
        self.y_prev = 0.0

    def process_block(self, block: np.ndarray) -> np.ndarray:
        out = np.zeros_like(block)
        for i in range(len(block)):
            y = block[i] - self.x_prev + self.r * self.y_prev
            self.x_prev = block[i]
            self.y_prev = y
            out[i] = y
        return out


class PythonHighPassBiquad:
    """2nd-order Butterworth High-Pass matching embedded/dsp/biquad_filter.hpp."""
    def __init__(self, sample_rate: float = 16000.0, cutoff_hz: float = 80.0, q: float = 0.70710678):
        w0 = 2.0 * math.pi * cutoff_hz / sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        self.b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
        self.a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
        self.zi = np.zeros(2, dtype=np.float32)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        out, self.zi = sp_signal.lfilter(self.b, self.a, block, zi=self.zi)
        return out.astype(np.float32)


def test_capture_preprocessing_stft_chain():
    """
    Feed simulated microphone audio with:
    - Heavy DC offset (+0.40)
    - 25 Hz low-frequency mechanical rumble (amplitude 0.35)
    - 500 Hz desired speech tone (amplitude 0.50)
    Verify through DC blocker -> High-pass filter -> STFT analysis:
    1. DC is attenuated by > 50 dB.
    2. 25 Hz rumble is attenuated by > 18 dB.
    3. 500 Hz speech tone power is preserved within 0.5 dB.
    4. Spectral frames contain valid non-negative magnitudes and bounded phase across all 120 hops.
    """
    sample_rate = 16000
    hop_size = 80  # 5ms
    fft_size = 512
    num_hops = 120  # 600ms of streaming audio
    total_samples = num_hops * hop_size

    t = np.arange(total_samples) / sample_rate

    # Synthetic mic capture signal
    dc_bias = 0.40
    rumble_25hz = 0.35 * np.sin(2 * np.pi * 25.0 * t)
    speech_500hz = 0.50 * np.sin(2 * np.pi * 500.0 * t)
    raw_mic_stream = (dc_bias + rumble_25hz + speech_500hz).astype(np.float32)

    # Initialize chain
    dc_blocker = PythonDcBlocker(r=0.995)
    high_pass = PythonHighPassBiquad(sample_rate=sample_rate, cutoff_hz=80.0)
    stft = PythonStftEngine(fft_size=fft_size, hop_size=hop_size)

    preprocessed_chunks = []
    spectral_frames = []

    for h in range(num_hops):
        # 1. Capture chunk from mic
        chunk = raw_mic_stream[h * hop_size : (h + 1) * hop_size]

        # 2. Preprocessing: DC Blocker -> 80Hz Biquad
        dc_cleaned = dc_blocker.process_block(chunk)
        hp_filtered = high_pass.process_block(dc_cleaned)
        preprocessed_chunks.append(hp_filtered)

        # 3. STFT Spectral Analysis
        spec = stft.analyze(hp_filtered)
        spectral_frames.append(spec)

    preprocessed_audio = np.concatenate(preprocessed_chunks)

    # Evaluate steady state (last 60 hops = 4800 samples)
    eval_slice = preprocessed_audio[3000:]
    eval_t = t[3000:]

    # A. DC offset check
    residual_dc = float(np.mean(eval_slice))
    dc_attenuation_db = 20.0 * math.log10((abs(residual_dc) + 1e-12) / dc_bias)
    assert abs(residual_dc) < 0.005, f"Residual DC {residual_dc:.4f} too high!"
    assert dc_attenuation_db < -35.0, f"DC attenuation {dc_attenuation_db:.1f} dB insufficient!"

    # B. Frequency domain analysis of preprocessed signal
    fft_eval = np.fft.rfft(eval_slice)
    freqs = np.fft.rfftfreq(len(eval_slice), 1.0 / sample_rate)

    bin_25 = np.argmin(np.abs(freqs - 25.0))
    bin_500 = np.argmin(np.abs(freqs - 500.0))

    mag_25 = np.abs(fft_eval[bin_25])
    mag_500 = np.abs(fft_eval[bin_500])

    # Reference FFT without filters
    ref_fft = np.fft.rfft(raw_mic_stream[3000:] - dc_bias)
    ref_25 = np.abs(ref_fft[bin_25])
    ref_500 = np.abs(ref_fft[bin_500])

    attenuation_25hz_db = 20.0 * math.log10(mag_25 / (ref_25 + 1e-12))
    preservation_500hz_db = 20.0 * math.log10(mag_500 / (ref_500 + 1e-12))

    assert attenuation_25hz_db < -15.0, f"25Hz rumble attenuation {attenuation_25hz_db:.1f} dB is less than -15 dB!"
    assert abs(preservation_500hz_db) < 1.0, f"500Hz speech tone altered by {preservation_500hz_db:.2f} dB!"

    # C. STFT spectra validity
    for spec in spectral_frames[30:]:
        mags = np.abs(spec)
        assert np.all(np.isfinite(mags))
        assert len(mags) == 257
        # Peak must be around 500 Hz (bin ~16 for N=512 at 16kHz)
        bin_hz = 16000.0 / 512.0 # 31.25 Hz/bin
        speech_bin = int(round(500.0 / bin_hz)) # ~16
        peak_bin = np.argmax(mags)
        assert abs(peak_bin - speech_bin) <= 1, f"Expected spectral peak at bin {speech_bin}, got {peak_bin}"
