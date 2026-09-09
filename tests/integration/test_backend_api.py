"""
Integration tests for the NOICELESSX FastAPI Backend.
Strict Zero-Mock Enforcement:
- NO mock classes, NO MagicMock, NO monkeypatching.
- All offline tests hit closed OS TCP sockets to verify honest connection failure reporting.
- All online tests execute against a real running RealtimePipeline instance exposed over
  a live OS network TCP socket via RuntimeIpcServer.
"""

import json
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.ipc_client import ipc_client
from backend.config import settings
from ai.runtime.realtime_pipeline import RealtimePipeline
from ai.runtime.ipc_server import RuntimeIpcServer


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def running_runtime_server():
    """
    Spawns a REAL RealtimePipeline instance and starts a REAL RuntimeIpcServer
    listening on an OS-assigned ephemeral TCP port.
    Feeds actual audio samples so the pipeline generates real DSP, AI, NLMS, and fusion metrics.
    """
    sample_rate = 16000
    hop_size = 80
    pipeline = RealtimePipeline(sample_rate=sample_rate, hop_size=hop_size)

    # Process several real audio hops with synthetic acoustic signals
    # to populate live DSP filters, VAD, and fusion state
    t_axis = np.linspace(0, 0.005, hop_size, dtype=np.float32)
    sig_primary = 0.5 * np.sin(2 * np.pi * 440.0 * t_axis)
    sig_reference = 0.1 * np.random.normal(0, 0.05, hop_size).astype(np.float32)

    for _ in range(10):
        pipeline.process_hop(
            primary_samples=sig_primary,
            reference_samples=sig_reference,
            ai_confidence=0.94,
            impulse_prob=0.01,
            vad_prob=0.89,
            drift_ms=0.15
        )

    # Start live IPC server on ephemeral port (port 0 lets OS assign free port)
    server = RuntimeIpcServer(pipeline=pipeline, host="127.0.0.1", port=0)
    server.start()

    # Reconfigure client to talk to this live server's real OS socket
    old_host = ipc_client.tcp_host
    old_port = ipc_client.tcp_port
    ipc_client.tcp_host = "127.0.0.1"
    ipc_client.tcp_port = server.actual_port
    ipc_client.timeout = 2.0

    yield server, pipeline

    # Clean teardown of real sockets
    server.stop()
    pipeline.stop()
    ipc_client.tcp_host = old_host
    ipc_client.tcp_port = old_port


# ==============================================================================
# SECTION 1: DISCONNECTED RUNTIME TESTS (HONEST ZERO-MOCK FAILURE REPORTING)
# ==============================================================================

def test_health_reports_degraded_when_runtime_unreachable(client):
    """
    Verifies that when C++ runtime is unreachable, /api/health honestly reports
    status='degraded', runtime_connected=False, and NEVER returns fabricated healthy status.
    Uses a guaranteed closed TCP port to test real OS socket connection rejection.
    """
    # Guarantee offline socket: point to closed ephemeral port
    ipc_client.tcp_port = 59999

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "degraded"
    assert data["backend"] == "healthy"
    assert data["runtime_connected"] is False
    assert data["runtime_state"] == "offline"
    assert data["runtime_error"] is not None
    assert "unreachable" in data["runtime_error"].lower()


def test_metrics_returns_503_when_runtime_unreachable(client):
    """
    Verifies that /api/metrics honestly returns HTTP 503 instead of fabricating telemetry
    when the C++ runtime is offline.
    """
    ipc_client.tcp_port = 59999

    response = client.get("/api/metrics")
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"].lower()


def test_runtime_controls_return_503_when_runtime_unreachable(client):
    """
    Verifies that POST /api/runtime/* endpoints fail honestly with HTTP 503
    if the runtime cannot be reached over the OS socket.
    """
    ipc_client.tcp_port = 59999

    for endpoint in ["/api/runtime/start", "/api/runtime/stop", "/api/runtime/bypass", "/api/runtime/reset"]:
        response = client.post(endpoint)
        assert response.status_code == 503, f"Endpoint {endpoint} should return 503 when disconnected"


def test_system_and_audio_fallback_gracefully_when_disconnected(client):
    """
    GET /api/system, /api/audio/devices, and /api/model should continue to serve
    valid hardware configuration and kernel telemetry even if runtime is offline.
    """
    ipc_client.tcp_port = 59999

    # System metrics fallback to host kernel / proc
    res_sys = client.get("/api/system")
    assert res_sys.status_code == 200
    sys_data = res_sys.json()
    assert "cpu" in sys_data
    assert "memory" in sys_data
    assert sys_data["source"] in ("kernel", "host_kernel")

    # Audio devices fallback to configured YAML
    res_audio = client.get("/api/audio/devices")
    assert res_audio.status_code == 200
    audio_data = res_audio.json()
    assert audio_data["config"]["sample_rate"] == 16000
    assert len(audio_data["devices"]) >= 1

    # Model info fallback to configured YAML
    res_model = client.get("/api/model")
    assert res_model.status_code == 200
    model_data = res_model.json()
    assert "ComplexCRN" in model_data["model_name"]


# ==============================================================================
# SECTION 2: CONNECTED RUNTIME TESTS (AGAINST REAL RUNNING PIPELINE & IPC SERVER)
# ==============================================================================

def test_health_reports_healthy_when_connected(client, running_runtime_server):
    """
    Verifies /api/health against a real running runtime communicating over real TCP sockets.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["backend"] == "healthy"
    assert data["runtime_connected"] is True
    assert data["runtime_state"] in ("ready", "running")


def test_get_metrics_connected(client, running_runtime_server):
    """
    Verifies /api/metrics against real telemetry calculated by the active pipeline.
    """
    server, pipeline = running_runtime_server

    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "latencies" in data
    assert data["latencies"]["total_processing_us"] > 0.0
    assert data["processed_frames"] == 10
    assert data["fusion_mode"] == "NORMAL"
    assert 0.0 <= data["current_lambda"] <= 1.0
    assert data["drift_ms"] == 0.15
    assert data["alsa_xruns"]["primary"] == 0


def test_get_system_connected(client, running_runtime_server):
    """
    Verifies /api/system queries real CPU and thermal data from the running server.
    """
    response = client.get("/api/system")
    assert response.status_code == 200
    data = response.json()

    assert "cpu" in data
    assert "temperature_c" in data
    assert data["source"] == "c++_runtime"


def test_get_audio_devices_connected(client, running_runtime_server):
    """
    Verifies /api/audio/devices queries real hardware device configuration over IPC.
    """
    response = client.get("/api/audio/devices")
    assert response.status_code == 200
    data = response.json()

    assert data["config"]["sample_rate"] == 16000
    assert len(data["devices"]) >= 1


def test_get_model_connected(client, running_runtime_server):
    """
    Verifies /api/model queries model configuration from the live runtime.
    """
    response = client.get("/api/model")
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "ComplexCRN"
    assert "INT8" in data["quantization"]
    assert data["intra_op_threads"] == 2


def test_runtime_start_stop_bypass_reset_lifecycle(client, running_runtime_server):
    """
    Verifies real state transitions on the running RealtimePipeline instance
    driven entirely by HTTP POST requests through the real socket IPC layer.
    """
    server, pipeline = running_runtime_server

    # 1. Start runtime
    res_start = client.post("/api/runtime/start")
    assert res_start.status_code == 200
    assert res_start.json()["runtime_state"] == "running"
    assert pipeline.is_running is True

    # 2. Set Bypass
    res_bp = client.post("/api/runtime/bypass", json={"bypass": True})
    assert res_bp.status_code == 200
    assert pipeline.fusion.bypass is True

    # 3. Reset runtime
    res_reset = client.post("/api/runtime/reset")
    assert res_reset.status_code == 200
    assert pipeline.telemetry.processed_frames == 0

    # 4. Stop runtime
    res_stop = client.post("/api/runtime/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["runtime_state"] == "stopped"
    assert pipeline.is_running is False


# ==============================================================================
# SECTION 3: WEBSOCKET STREAMING TESTS (REAL TELEMETRY OVER REAL SOCKET)
# ==============================================================================

def test_websocket_telemetry_stream(client, running_runtime_server):
    """
    Connects to /ws/telemetry, receives real telemetry frames generated by the live pipeline,
    and performs bidirectional ping-pong over the WebSocket.
    """
    with client.websocket_connect("/ws/telemetry") as ws:
        msg1 = ws.receive_text()
        data1 = json.loads(msg1)

        assert data1["type"] == "telemetry"
        assert data1["status"] == "connected"
        assert "data" in data1
        assert data1["data"]["fusion_mode"] == "NORMAL"

        # Bidirectional ping
        ws.send_text(json.dumps({"command": "ping"}))
        pong_msg = ws.receive_text()
        pong_data = json.loads(pong_msg)
        assert pong_data["type"] == "pong"


def test_websocket_reports_offline_when_runtime_unreachable(client):
    """
    Verifies that /ws/telemetry sends telemetry_offline notifications
    when the real runtime socket cannot be reached, without closing the WebSocket.
    """
    ipc_client.tcp_port = 59999

    with client.websocket_connect("/ws/telemetry") as ws:
        msg = ws.receive_text()
        data = json.loads(msg)

        assert data["type"] == "telemetry_offline"
        assert data["status"] == "disconnected"
        assert "error" in data
