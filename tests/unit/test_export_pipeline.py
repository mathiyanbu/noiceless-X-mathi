import os
import tempfile
from pathlib import Path
import pytest
import torch

from ai.export.onnx_export import export_onnx_model
from ai.export.onnx_quantize import (
    quantize_model,
    evaluate_fp32_vs_int8,
    select_deployment_model
)

@pytest.fixture(scope="module")
def exported_models():
    """Fixture exporting FP32 model and INT8 quantized model into temporary dir."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fp32_path = str(Path(tmp_dir) / "test_model_fp32.onnx")
        int8_path = str(Path(tmp_dir) / "test_model_int8.onnx")

        # 1. Export FP32
        success_export = export_onnx_model(
            checkpoint_path=None,
            output_path=fp32_path,
            num_bins=257,
            tolerance=1e-4,
            verbose=False
        )
        assert success_export and Path(fp32_path).exists()

        # 2. Quantize to INT8
        success_quant = quantize_model(fp32_path, int8_path, verbose=False)
        assert success_quant and Path(int8_path).exists()

        yield {"fp32": fp32_path, "int8": int8_path}

def test_onnx_export_file_and_compression(exported_models):
    """Verify that export produces valid files and INT8 reduces footprint."""
    fp32_path = exported_models["fp32"]
    int8_path = exported_models["int8"]

    size_fp32 = os.path.getsize(fp32_path)
    size_int8 = os.path.getsize(int8_path)

    assert size_fp32 > 0
    assert size_int8 > 0
    # INT8 should achieve meaningful model size reduction
    assert size_int8 < size_fp32

def test_quantization_evaluation_metrics(exported_models):
    """Verify that evaluation metrics compute realistic deltas between FP32 and INT8."""
    metrics = evaluate_fp32_vs_int8(
        exported_models["fp32"],
        exported_models["int8"],
        num_test_samples=2,
        sample_rate=16000,
        verbose=False
    )

    assert "delta_snr_db" in metrics
    assert "delta_stoi" in metrics
    assert "delta_pesq" in metrics
    assert "max_spectral_diff" in metrics

    # The difference between FP32 and INT8 should be finite and bounded
    assert abs(metrics["delta_stoi"]) < 0.20
    assert abs(metrics["delta_snr_db"]) < 5.0

def test_model_selection_policy():
    """Verify that deployment selector rejects INT8 when degradation exceeds threshold."""
    # Case A: Acceptable degradation -> Choose INT8
    good_metrics = {
        "delta_snr_db": -0.4,
        "delta_stoi": -0.01,
        "delta_pesq": -0.05,
    }
    decision_good = select_deployment_model(
        good_metrics,
        max_snr_drop_db=1.0,
        max_stoi_drop=0.03,
        max_pesq_drop=0.15,
        verbose=False
    )
    assert decision_good == "INT8"

    # Case B: Severe degradation in STOI -> Reject INT8 and keep FP32
    bad_metrics = {
        "delta_snr_db": -0.4,
        "delta_stoi": -0.08, # Greater than 0.03 max drop
        "delta_pesq": -0.05,
    }
    decision_bad = select_deployment_model(
        bad_metrics,
        max_snr_drop_db=1.0,
        max_stoi_drop=0.03,
        max_pesq_drop=0.15,
        verbose=False
    )
    assert decision_bad == "FP32"
