#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.models.complex_crn import ComplexCRN

class StreamingCRNWrapper(nn.Module):
    """
    ONNX Export Wrapper for ComplexCRN ensuring clean signature:
    Inputs:
      - noisy_stft: (batch, 2, time_steps, num_bins)
      - hidden_in:  (num_layers, batch, hidden_dim)
    Outputs:
      - enhanced_stft: (batch, 2, time_steps, num_bins)
      - mask:          (batch, 2, time_steps, num_bins)
      - hidden_out:    (num_layers, batch, hidden_dim)
    """
    def __init__(self, model: ComplexCRN):
        super().__init__()
        self.model = model

    def forward(self, noisy_stft: torch.Tensor, hidden_in: torch.Tensor):
        s_hat, mask, hidden_out = self.model(noisy_stft, hidden_in)
        return s_hat, mask, hidden_out


def export_onnx_model(
    checkpoint_path: Optional[str] = None,
    output_path: str = "models/onnx/speech_enhancer_fp32.onnx",
    num_bins: int = 257,
    tolerance: float = 1e-4,
    opset_version: int = 17,
    verbose: bool = True
) -> bool:
    """
    Exports ComplexCRN to ONNX format with dynamic time axes and validates
    exact numerical equivalence between PyTorch and ONNX Runtime CPU EP.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("==========================================================================")
        print("          SIH26052 NOICELESSX — PyTorch to ONNX Export Pipeline          ")
        print("==========================================================================")

    # 1. Instantiate Model
    model = ComplexCRN(num_bins=num_bins)
    model.eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        if verbose:
            print(f"[Step 1] Loading weights from checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        if verbose:
            print("[Step 1] Initializing fresh baseline model weights for export.")

    # 2. Validate PyTorch model on held-out test input
    batch_size = 1
    test_time_steps = 10
    dummy_x = torch.randn(batch_size, 2, test_time_steps, num_bins, dtype=torch.float32)
    dummy_hidden = model.init_hidden(batch_size, device=torch.device("cpu"))

    with torch.no_grad():
        pyt_s_hat, pyt_mask, pyt_hidden_out = model(dummy_x, dummy_hidden)

    assert torch.all(torch.isfinite(pyt_s_hat)), "PyTorch output contains NaN/Inf!"
    if verbose:
        print(f"[Step 2] Validated PyTorch FP32 forward output shape: {tuple(pyt_s_hat.shape)}")

    # 3. Export to ONNX with dynamic time axes for streaming inference
    wrapper = StreamingCRNWrapper(model)
    wrapper.eval()

    dynamic_axes = {
        "noisy_stft": {2: "time_steps"},
        "enhanced_stft": {2: "time_steps"},
        "mask": {2: "time_steps"}
    }

    if verbose:
        print(f"[Step 3] Exporting ONNX graph to: {out_file}")

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_hidden),
        str(out_file),
        input_names=["noisy_stft", "hidden_in"],
        output_names=["enhanced_stft", "mask", "hidden_out"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True
    )

    # 4. Check ONNX Model Integrity
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    if verbose:
        print("  ONNX model integrity verified successfully.")

    # 5. Run ONNX Runtime CPU EP and validate numerical match
    if verbose:
        print("[Step 4] Running ONNX Runtime (CPUExecutionProvider) validation...")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(out_file), sess_options, providers=["CPUExecutionProvider"])

    ort_inputs = {
        "noisy_stft": dummy_x.numpy(),
        "hidden_in": dummy_hidden.numpy()
    }
    ort_s_hat, ort_mask, ort_hidden_out = session.run(None, ort_inputs)

    # Compute exact error metrics
    max_abs_diff_shat = float(np.max(np.abs(ort_s_hat - pyt_s_hat.numpy())))
    max_abs_diff_mask = float(np.max(np.abs(ort_mask - pyt_mask.numpy())))
    max_abs_diff_hidden = float(np.max(np.abs(ort_hidden_out - pyt_hidden_out.numpy())))

    rel_error = float(np.linalg.norm(ort_s_hat - pyt_s_hat.numpy()) / (np.linalg.norm(pyt_s_hat.numpy()) + 1e-10))

    if verbose:
        print("\nNumerical Equivalence Report (PyTorch vs ONNX Runtime):")
        print(f"  Max Absolute Diff (S_hat):  {max_abs_diff_shat:.6e} (Tolerance: {tolerance:.1e})")
        print(f"  Max Absolute Diff (Mask):   {max_abs_diff_mask:.6e}")
        print(f"  Max Absolute Diff (Hidden): {max_abs_diff_hidden:.6e}")
        print(f"  Relative Frobenius Error:   {rel_error:.6e}")

    assert max_abs_diff_shat < tolerance, f"Max difference {max_abs_diff_shat} exceeds tolerance {tolerance}!"
    assert max_abs_diff_mask < tolerance, f"Mask difference {max_abs_diff_mask} exceeds tolerance {tolerance}!"

    # 6. Test single-frame streaming input on exported ONNX model (time_steps=1)
    single_frame = torch.randn(1, 2, 1, num_bins, dtype=torch.float32)
    single_hidden = dummy_hidden.clone()
    stream_out = session.run(None, {"noisy_stft": single_frame.numpy(), "hidden_in": single_hidden.numpy()})
    assert stream_out[0].shape == (1, 2, 1, num_bins), f"Streaming frame shape mismatch: {stream_out[0].shape}"

    if verbose:
        print("\n[Step 5] Single-frame streaming verification: PASSED (Shape (1, 2, 1, 257))")
        print(f"SUCCESS: Exported model ready at {out_file}\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ComplexCRN PyTorch model to ONNX for Raspberry Pi.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to input PyTorch checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="models/onnx/speech_enhancer_fp32.onnx", help="Output path for .onnx file")
    parser.add_argument("--num-bins", type=int, default=257, help="Number of frequency bins (default: 257)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max absolute difference tolerance")
    args = parser.parse_args()

    success = export_onnx_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_bins=args.num_bins,
        tolerance=args.tolerance,
        verbose=True
    )
    sys.exit(0 if success else 1)
