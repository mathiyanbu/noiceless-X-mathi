import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import onnx
import onnxruntime as ort
import numpy as np

class TinyImpulseMLP(nn.Module):
    """
    Ultra-lightweight Multi-Layer Perceptron (MLP) for single-channel impulsive noise detection.
    
    Architecture:
      Input (8 features):
        [RMS, SpectralFlux, CrestFactor, ZCR, BandLow, BandMidLow, BandMidHigh, BandHigh]
      Hidden Layer 1: 8 -> 16 (ReLU)
      Hidden Layer 2: 16 -> 8 (ReLU)
      Output Layer:   8 -> 1 (Sigmoid -> Probability in [0, 1])
    
    Total parameters: (8*16 + 16) + (16*8 + 8) + (8*1 + 1) = 144 + 136 + 9 = 289 parameters (~1.2 KB).
    Ultra-low latency execution (< 5 microseconds per frame on Raspberry Pi 4/5 CPU).
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch, 8)
        Returns:
            probability: Tensor of shape (batch, 1) in range [0, 1]
        """
        return self.net(features)


def export_impulse_model_to_onnx(
    model: TinyImpulseMLP,
    output_path: str = "models/onnx/impulse_detector.onnx",
    opset_version: int = 17,
    tolerance: float = 1e-5
) -> bool:
    """Exports TinyImpulseMLP to ONNX format and verifies numerical match."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, 8, dtype=torch.float32)
    with torch.no_grad():
        pyt_out = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        input_names=["features"],
        output_names=["probability"],
        dynamic_axes={"features": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=opset_version,
        do_constant_folding=True
    )

    # Validate ONNX graph
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)

    # Validate numerical equivalence with ONNX Runtime
    session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"features": dummy_input.numpy()})[0]

    max_diff = float(np.max(np.abs(pyt_out.numpy() - ort_out)))
    assert max_diff < tolerance, f"ONNX validation max diff {max_diff} exceeds tolerance {tolerance}!"

    print(f"Successfully exported and validated ONNX impulse detector at: {out_file} (Max diff: {max_diff:.2e})")
    return True
