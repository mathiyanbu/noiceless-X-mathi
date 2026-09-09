"""
Telemetry and metrics endpoint providing live latency, RTF, XRUN, and fusion state.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from backend.schemas import MetricsResponse, StageLatencies
from backend.ipc_client import ipc_client, RuntimeUnavailableError

router = APIRouter(prefix="/api", tags=["metrics"])


def parse_telemetry_dict(data: Dict[str, Any]) -> MetricsResponse:
    """
    Parses either nested or flat telemetry dictionary into a typed MetricsResponse.
    """
    if "latencies" in data and isinstance(data["latencies"], dict):
        latencies = StageLatencies(**data["latencies"])
    else:
        latencies = StageLatencies(
            capture_us=float(data.get("capture_us", 0.0)),
            preprocessing_us=float(data.get("preprocessing_us", 0.0)),
            stft_us=float(data.get("stft_us", 0.0)),
            ai_inference_us=float(data.get("ai_inference_us", 0.0)),
            istft_us=float(data.get("istft_us", 0.0)),
            nlms_us=float(data.get("nlms_us", 0.0)),
            fusion_us=float(data.get("fusion_us", 0.0)),
            playback_queue_us=float(data.get("playback_queue_us", 0.0)),
            total_processing_us=float(data.get("total_processing_us", 0.0)),
            end_to_end_latency_ms=float(data.get("end_to_end_latency_ms", 0.0)),
        )

    if "alsa_xruns" in data and isinstance(data["alsa_xruns"], dict):
        alsa_xruns = data["alsa_xruns"]
    else:
        alsa_xruns = {
            "primary": int(data.get("alsa_xruns_primary", 0)),
            "reference": int(data.get("alsa_xruns_reference", 0)),
            "playback": int(data.get("alsa_xruns_playback", 0)),
        }

    mode_val = data.get("fusion_mode", "NORMAL")
    mode_str = mode_val.name if hasattr(mode_val, "name") else str(mode_val)

    return MetricsResponse(
        latencies=latencies,
        rtf=float(data.get("rtf", 0.0)),
        processed_frames=int(data.get("processed_frames", 0)),
        dropped_frames=int(data.get("dropped_frames", 0)),
        alsa_xruns=alsa_xruns,
        drift_ms=float(data.get("drift_ms", 0.0)),
        drift_samples=int(data.get("drift_samples", 0)),
        drift_warning=bool(data.get("drift_warning", False)),
        fusion_mode=mode_str,
        current_lambda=float(data.get("current_lambda", 0.5)),
        impulse_envelope_gain=float(data.get("impulse_envelope_gain", 1.0)),
        ai_confidence=float(data.get("ai_confidence", 1.0)),
        impulse_probability=float(data.get("impulse_probability", 0.0)),
        vad_probability=float(data.get("vad_probability", 0.0)),
        estimated_input_snr_db=float(data["estimated_input_snr_db"]) if "estimated_input_snr_db" in data and data["estimated_input_snr_db"] is not None else None,
        estimated_output_snr_db=float(data["estimated_output_snr_db"]) if "estimated_output_snr_db" in data and data["estimated_output_snr_db"] is not None else None,
        estimated_snr_improvement_db=float(data["estimated_snr_improvement_db"]) if "estimated_snr_improvement_db" in data and data["estimated_snr_improvement_db"] is not None else None,
        snr_is_estimated=bool(data.get("snr_is_estimated", True)),
        primary_level_dbfs=float(data["primary_level_dbfs"]) if "primary_level_dbfs" in data and data["primary_level_dbfs"] is not None else None,
        reference_level_dbfs=float(data["reference_level_dbfs"]) if "reference_level_dbfs" in data and data["reference_level_dbfs"] is not None else None,
        output_level_dbfs=float(data["output_level_dbfs"]) if "output_level_dbfs" in data and data["output_level_dbfs"] is not None else None,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Returns latest real telemetry snapshot (latencies, RTF, XRUN counts, clock drift, fusion mode).
    Honestly returns 503 if the C++ runtime is offline.
    """
    try:
        res = ipc_client.send_command("get_metrics")
        if res.get("status") == "ok":
            metrics_data = res.get("metrics", {})
            return parse_telemetry_dict(metrics_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=res.get("message", "Runtime returned non-ok status for get_metrics")
            )
    except RuntimeUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"C++ runtime is offline or unreachable: {e}"
        )
