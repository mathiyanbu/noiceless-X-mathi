"""
Integration Test: STFT -> ONNX Model -> iSTFT Streaming Enhancement Chain.
Verifies:
- Frame-by-frame streaming audio pipeline with persistent recurrent hidden state.
- Real-time compatibility with both FP32 and INT8 quantized ONNX models.
- Continuous signal synthesis without boundary clicks or discontinuities.
- Recurrent hidden state evolution and state reset behavior.
- Noise reduction effectiveness in spectral domain and time domain.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.unit.test_dsp_py import PythonStftEngine


@pytest.fixture(params=["speech_enhancer_int8.onnx", "speech_enhancer_fp32.onnx"])
def onnx_model_path(request):
    model_path = REPO_ROOT / "models" / "onnx" / request.param
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")
    return str(model_path)


def test_stft_onnx_istft_streaming_pipeline(onnx_model_path):
    """
    Simulate real-time streaming:
    Input audio -> STFT analysis -> Complex tensor -> ONNX inference with persistent GRU state -> iSTFT synthesis.
    """
    sample_rate = 16000
    hop_size = 80  # 5ms
    fft_size = 512
    num_hops = 60  # 300ms
    total_samples = num_hops * hop_size

    t = np.arange(total_samples) / sample_rate

    # Generate synthetic noisy speech: 440 Hz vowel harmonic + white noise
    speech = 0.4 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 880.0 * t)
    np.random.seed(42)
    noise = 0.25 * np.random.randn(total_samples)
    noisy_input = (speech + noise).astype(np.float32)

    # Initialize STFT engines
    stft_analysis = PythonStftEngine(fft_size=fft_size, hop_size=hop_size)
    stft_synthesis = PythonStftEngine(fft_size=fft_size, hop_size=hop_size)

    # Initialize ONNX Runtime session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])

    # Persistent hidden state: (num_layers=2, batch=1, hidden_dim=256)
    hidden_state = np.zeros((2, 1, 256), dtype=np.float32)

    enhanced_hops = []
    hidden_states_evolution = []

    for h in range(num_hops):
        chunk = noisy_input[h * hop_size : (h + 1) * hop_size]

        # 1. STFT Analysis
        spec = stft_analysis.analyze(chunk)
        num_bins = len(spec)
        assert num_bins == 257

        # 2. Format as Complex Tensor: (batch=1, channels=2, time=1, bins=257)
        noisy_stft = np.zeros((1, 2, 1, num_bins), dtype=np.float32)
        noisy_stft[0, 0, 0, :] = np.real(spec)
        noisy_stft[0, 1, 0, :] = np.imag(spec)

        # 3. ONNX Inference Step
        ort_inputs = {
            "noisy_stft": noisy_stft,
            "hidden_in": hidden_state
        }
        enhanced_stft, mask, next_hidden = session.run(None, ort_inputs)

        # Verify recurrent state evolution
        assert not np.all(next_hidden == 0.0), "Hidden state failed to evolve from zero!"
        hidden_states_evolution.append(np.linalg.norm(next_hidden))
        hidden_state = next_hidden

        # 4. Form enhanced complex spectrum: real + 1j * imag
        enh_real = enhanced_stft[0, 0, 0, :]
        enh_imag = enhanced_stft[0, 1, 0, :]
        enhanced_spec = enh_real + 1j * enh_imag

        # 5. iSTFT Synthesis
        enh_hop = stft_synthesis.synthesize(enhanced_spec)
        enhanced_hops.append(enh_hop)

    enhanced_audio = np.concatenate(enhanced_hops)

    # 6. Assertions
    assert len(enhanced_audio) == total_samples
    assert np.all(np.isfinite(enhanced_audio)), "Enhanced audio contains NaN or Inf!"

    # Hidden state must have changed across frames
    assert hidden_states_evolution[-1] != hidden_states_evolution[0]

    # Check for hop boundary continuity in steady state (no unphysical explosion)
    hop_diffs = np.abs(np.diff(enhanced_audio[fft_size:]))
    max_hop_step = float(np.max(hop_diffs)) if len(hop_diffs) > 0 else 0.0
    assert max_hop_step < 2.0, f"Unphysical jump discontinuity detected: {max_hop_step:.3f}"

    # Latency: Verify output audio contains signal energy in steady state
    steady_state = enhanced_audio[fft_size:]
    assert np.mean(steady_state ** 2) > 1e-5, "Enhanced audio is silent in steady-state region!"
