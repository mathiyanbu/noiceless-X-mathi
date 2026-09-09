"""
Runtime lifecycle and state control endpoints (start, stop, bypass, reset).
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status, Body
from backend.schemas import RuntimeCommandResponse, BypassRequest
from backend.ipc_client import ipc_client, RuntimeUnavailableError

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


@router.post("/start", response_model=RuntimeCommandResponse)
async def start_runtime():
    """
    Start the embedded C++ audio capture, processing, and playback threads.
    """
    try:
        res = ipc_client.send_command("start")
        return RuntimeCommandResponse(
            status=res.get("status", "ok"),
            command="start",
            message=res.get("message", "Real-time processing pipeline started successfully"),
            runtime_state=res.get("runtime_state", "running")
        )
    except RuntimeUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"C++ runtime IPC unreachable: {e}"
        )


@router.post("/stop", response_model=RuntimeCommandResponse)
async def stop_runtime():
    """
    Stop all real-time processing threads and cleanly drain audio buffers.
    """
    try:
        res = ipc_client.send_command("stop")
        return RuntimeCommandResponse(
            status=res.get("status", "ok"),
            command="stop",
            message=res.get("message", "Real-time processing pipeline stopped cleanly"),
            runtime_state=res.get("runtime_state", "stopped")
        )
    except RuntimeUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"C++ runtime IPC unreachable: {e}"
        )


@router.post("/bypass", response_model=RuntimeCommandResponse)
async def set_bypass(req: Optional[BypassRequest] = None):
    """
    Toggle or set operator BYPASS mode.
    When bypass is enabled, raw primary audio is routed directly to playback,
    bypassing AI/NLMS processing while preserving telemetry tracking.
    """
    bypass_val = True if req is None or req.bypass is None else req.bypass
    try:
        res = ipc_client.send_command("bypass", bypass=bypass_val)
        return RuntimeCommandResponse(
            status=res.get("status", "ok"),
            command="bypass",
            message=res.get("message", f"Bypass mode set to {bypass_val}"),
            runtime_state=res.get("runtime_state")
        )
    except RuntimeUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"C++ runtime IPC unreachable: {e}"
        )


@router.post("/reset", response_model=RuntimeCommandResponse)
async def reset_runtime():
    """
    Reset internal DSP filter weights, AI recurrent hidden states, and telemetry counters.
    """
    try:
        res = ipc_client.send_command("reset")
        return RuntimeCommandResponse(
            status=res.get("status", "ok"),
            command="reset",
            message=res.get("message", "Runtime state and telemetry counters reset successfully"),
            runtime_state=res.get("runtime_state", "ready")
        )
    except RuntimeUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"C++ runtime IPC unreachable: {e}"
        )
