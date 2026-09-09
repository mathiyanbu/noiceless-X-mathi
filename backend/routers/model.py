"""
AI model and quantization metadata endpoint.
"""

import os
import yaml
from fastapi import APIRouter
from backend.schemas import ModelResponse
from backend.config import settings
from backend.ipc_client import ipc_client, RuntimeUnavailableError

router = APIRouter(prefix="/api/model", tags=["model"])


def _load_model_metadata() -> ModelResponse:
    model_path = "models/onnx/speech_enhancer_int8.onnx"
    threads = 2
    ep = "CPUExecutionProvider"

    if os.path.exists(settings.config_yaml_path):
        try:
            with open(settings.config_yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            ai_cfg = data.get("ai", {})
            model_path = ai_cfg.get("model_path", model_path)
            threads = ai_cfg.get("intra_op_num_threads", threads)
            ep = ai_cfg.get("execution_provider", ep)
        except Exception:
            pass

    quant = "INT8 (Dynamic Quantization for ARM NEON)" if "int8" in model_path.lower() else "FP32"

    return ModelResponse(
        model_path=model_path,
        model_name="ComplexCRN",
        quantization=quant,
        execution_provider=ep,
        intra_op_threads=threads,
        input_shape=["noisy_stft: (B, 2, T, 257)", "hidden_in: (2, B, 128)"],
        output_shape=["enhanced_stft: (B, 2, T, 257)", "mask: (B, 2, T, 257)", "hidden_out: (2, B, 128)"]
    )


@router.get("", response_model=ModelResponse)
async def get_model_info():
    """
    Returns the currently active ONNX neural speech enhancement model details,
    quantization precision, execution provider, and tensor I/O shapes.
    """
    try:
        res = ipc_client.send_command("get_model")
        if res.get("status") == "ok":
            return ModelResponse(**res.get("model", {}))
    except RuntimeUnavailableError:
        pass

    return _load_model_metadata()
