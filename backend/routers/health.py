"""
Health probe endpoint reporting honest backend and runtime connectivity status.
"""

from fastapi import APIRouter
from backend.schemas import HealthResponse
from backend.ipc_client import ipc_client, RuntimeUnavailableError

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """
    Probe system health.
    If the embedded C++ runtime is unreachable, honestly reports status='degraded'
    and runtime_connected=False (never fabricates healthy status).
    """
    try:
        res = ipc_client.send_command("ping")
        is_conn = (res.get("status") == "ok")
        state = res.get("runtime", "ready")
        error_msg = None
    except RuntimeUnavailableError as e:
        is_conn = False
        state = "offline"
        error_msg = str(e)

    if is_conn:
        return HealthResponse(
            status="healthy",
            backend="healthy",
            runtime_connected=True,
            runtime_state=state
        )
    else:
        return HealthResponse(
            status="degraded",
            backend="healthy",
            runtime_connected=False,
            runtime_error=error_msg or "C++ runtime IPC socket unreachable. Is sih26052 running?",
            runtime_state="offline"
        )
