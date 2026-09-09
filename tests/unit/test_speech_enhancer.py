import os
import time
import numpy as np
import pytest
from pathlib import Path
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

@pytest.fixture
def onnx_model_path():
    int8_path = REPO_ROOT / "models" / "onnx" / "speech_enhancer_int8.onnx"
    fp32_path = REPO_ROOT / "models" / "onnx" / "speech_enhancer_fp32.onnx"
    if int8_path.exists():
        return str(int8_path)
    if fp32_path.exists():
        return str(fp32_path)
    pytest.skip("No ONNX model available in models/onnx/")

def test_session_instantiation_and_threading(onnx_model_path):
    """Verifies that ONNX Runtime session loads once with intra-op thread tuning."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])
    assert session is not None
    assert len(session.get_inputs()) == 2
    assert len(session.get_outputs()) >= 2

def test_streaming_persistent_hidden_state(onnx_model_path):
    """
    Verifies that recurrent hidden state persists and evolves across frames,
    and reset_state clears it back to zeros.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])

    # Persistent hidden state buffer
    hidden_state = np.zeros((2, 1, 256), dtype=np.float32)

    # Frame 1
    frame_1 = np.random.randn(1, 2, 1, 257).astype(np.float32)
    out1_stft, out1_mask, hidden_state = session.run(
        None, {"noisy_stft": frame_1, "hidden_in": hidden_state}
    )

    # State must now be non-zero after frame 1
    assert np.any(np.abs(hidden_state) > 1e-7), "Hidden state remained all zeros!"
    state_after_frame1 = hidden_state.copy()

    # Frame 2 (fed with previous hidden state)
    frame_2 = np.random.randn(1, 2, 1, 257).astype(np.float32)
    out2_stft, out2_mask, hidden_state = session.run(
        None, {"noisy_stft": frame_2, "hidden_in": hidden_state}
    )
    state_after_frame2 = hidden_state.copy()

    # Frame 2 output state must differ from Frame 1 output state
    assert not np.allclose(state_after_frame1, state_after_frame2), "State did not update across frames!"

    # Reset state
    hidden_state = np.zeros((2, 1, 256), dtype=np.float32)
    # Re-run Frame 1 with reset state: must match state_after_frame1
    _, _, reset_out_state = session.run(
        None, {"noisy_stft": frame_1, "hidden_in": hidden_state}
    )
    assert np.allclose(state_after_frame1, reset_out_state, atol=1e-5), "reset_state did not reproduce state!"

def test_frame_by_frame_streaming_latency(onnx_model_path):
    """
    Runs 50 sequential frames mimicking real-time audio pipeline and evaluates
    average, 95th-percentile latency, and output validity.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])

    num_frames = 50
    latencies_ms = []
    hidden_state = np.zeros((2, 1, 256), dtype=np.float32)

    # Preallocated frame
    input_frame = np.random.randn(1, 2, 1, 257).astype(np.float32)

    for i in range(num_frames):
        t0 = time.perf_counter()
        out_stft, out_mask, hidden_state = session.run(
            None, {"noisy_stft": input_frame, "hidden_in": hidden_state}
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        # Output must be finite
        assert np.all(np.isfinite(out_stft)), f"Non-finite values in enhanced STFT at frame {i}!"

    latencies_ms = np.array(latencies_ms)
    mean_lat = np.mean(latencies_ms)
    p95_lat = np.percentile(latencies_ms, 95)
    max_lat = np.max(latencies_ms)

    print(f"\nPython ORT Inference Latency (n={num_frames}):")
    print(f"  Mean: {mean_lat:.2f} ms | P95: {p95_lat:.2f} ms | Max: {max_lat:.2f} ms")
    assert mean_lat > 0.0
    assert p95_lat > 0.0

def test_error_handling_and_degraded_mode(onnx_model_path):
    """Verifies that invalid input triggers exception without crashing the test runner."""
    opts = ort.SessionOptions()
    session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])

    invalid_frame = np.random.randn(1, 2, 1, 100).astype(np.float32) # wrong freq bins
    hidden_state = np.zeros((2, 1, 256), dtype=np.float32)

    with pytest.raises(Exception):
        session.run(None, {"noisy_stft": invalid_frame, "hidden_in": hidden_state})
