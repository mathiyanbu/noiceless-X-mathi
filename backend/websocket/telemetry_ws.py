"""
WebSocket endpoint streaming live telemetry snapshots at 20 Hz (50ms).
Strictly transmits telemetry metadata and metrics; NEVER transmits real-time audio samples.
"""

import asyncio
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.config import settings
from backend.ipc_client import ipc_client, RuntimeUnavailableError
from backend.routers.metrics import parse_telemetry_dict

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    """
    Continuous telemetry stream for frontend dashboard visualization.
    Pushes latency breakdown, RTF, XRUN counters, drift, and fusion parameters every 50ms (20 Hz).
    """
    await websocket.accept()
    push_interval = settings.ws_push_interval_ms / 1000.0

    try:
        while True:
            t_now = time.time()
            try:
                # Query real telemetry from C++ runtime over IPC
                res = ipc_client.send_command("get_metrics")
                if res.get("status") == "ok":
                    metrics = parse_telemetry_dict(res.get("metrics", {}))
                    payload = {
                        "type": "telemetry",
                        "status": "connected",
                        "data": metrics.model_dump(),
                        "timestamp": t_now
                    }
                else:
                    payload = {
                        "type": "telemetry_warning",
                        "status": "runtime_error",
                        "message": res.get("message", "Non-ok response from runtime"),
                        "timestamp": t_now
                    }
            except RuntimeUnavailableError as e:
                # Honestly report runtime disconnection without crashing the WebSocket
                payload = {
                    "type": "telemetry_offline",
                    "status": "disconnected",
                    "error": str(e),
                    "timestamp": t_now
                }

            await websocket.send_text(json.dumps(payload))

            # Non-blocking sleep with check for incoming client messages (e.g. ping/close)
            try:
                client_msg = await asyncio.wait_for(websocket.receive_text(), timeout=push_interval)
                # Client sent a message, handle commands like ping or bye
                try:
                    data = json.loads(client_msg)
                    if data.get("command") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
                except Exception:
                    pass
            except asyncio.TimeoutError:
                # Normal interval timeout, loop and push next frame
                pass

    except WebSocketDisconnect:
        # Normal client disconnection
        pass
    except Exception:
        # Socket or protocol abort
        pass
