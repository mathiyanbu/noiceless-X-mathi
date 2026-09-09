"""
Integration tests for the NOICELESSX FastAPI Backend.
Verifies all REST and WebSocket endpoints against:
1. Disconnected C++ runtime (validating honest degradation, zero fabrication)
2. Connected runtime using an explicitly labeled test stub (ExplicitTestRuntimeStub)
3. Live socket IPC connection over loopback TCP
"""

import json
import socket
import threading
import time
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.ipc_client import ipc_client, RuntimeUnavailableError
from backend.schemas import HealthResponse, SystemResponse, AudioDevicesResponse, ModelResponse, MetricsResponse


class ExplicitTestRuntimeStub:
    """
    EXPLICIT TEST STUB ONLY: Simulates the embedded C++ runtime IPC protocol
    strictly during automated test runs when the physical runtime binary is offline.
    Explicitly labeled as stubbed per system requirements.
    """

    def __init__(self, is_alive: bool = True):
        self._alive = is_alive
        self.state = "ready"
        self.bypass_mode = False
        self.processed_frames = 120
        self.current_lambda = 0.65

    def is_alive(self) -> bool:
        return self._alive

    def set_alive(self, alive: bool):
        self._alive = alive

    def handle_command(self, command: str, **kwargs) -> dict:
        if not self._alive:
            raise RuntimeUnavailableError(
                "ExplicitTestRuntimeStub: Simulated runtime IPC unreachable (stub is offline)"
            )

        if command == "ping":
            return {"status": "ok", "runtime": self.state}

        elif command in ("get_metrics", "get_telemetry"):
            return {
                "status": "ok",
                "metrics": {
                    "latencies": {
                        "capture_us": 120.0,
                        "preprocessing_us": 45.0,
                        "stft_us": 150.0,
                        "ai_inference_us": 1850.0,
                        "istft_us": 130.0,
                        "nlms_us": 210.0,
                        "fusion_us": 80.0,
                        "playback_queue_us": 95.0,
                        "total_processing_us": 2680.0,
                        "end_to_end_latency_ms": 7.68
                    },
                    "rtf": 0.268,
                    "processed_frames": self.processed_frames,
                    "dropped_frames": 0,
                    "alsa_xruns": {"primary": 0, "reference": 0, "playback": 0},
                    "drift_ms": 0.12,
                    "drift_samples": 2,
                    "drift_warning": False,
                    "fusion_mode": "NORMAL",
                    "current_lambda": self.current_lambda,
                    "impulse_envelope_gain": 1.0,
                    "ai_confidence": 0.95,
                    "impulse_probability": 0.01,
                    "vad_probability": 0.88
                }
            }

        elif command == "get_system":
            return {
                "status": "ok",
                "cpu": {
                    "overall_pct": 24.5,
                    "cores_pct": [22.0, 26.0, 24.0, 26.0]
                },
                "temperature_c": 48.2,
                "temperature_available": True
            }

        elif command == "start":
            self.state = "running"
            return {
                "status": "ok",
                "command": "start",
                "message": "Real-time pipeline started successfully",
                "runtime_state": self.state
            }

        elif command == "stop":
            self.state = "stopped"
            return {
                "status": "ok",
                "command": "stop",
                "message": "Real-time pipeline stopped cleanly",
                "runtime_state": self.state
            }

        elif command == "bypass":
            self.bypass_mode = kwargs.get("bypass", True)
            return {
                "status": "ok",
                "command": "bypass",
                "message": f"Bypass mode set to {self.bypass_mode}",
                "runtime_state": self.state
            }

        elif command == "reset":
            self.processed_frames = 0
            self.state = "ready"
            return {
                "status": "ok",
                "command": "reset",
                "message": "Runtime state and telemetry counters reset successfully",
                "runtime_state": self.state
            }

        elif command == "get_devices":
            return {
                "status": "ok",
                "config": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "frame_ms": 10,
                    "hop_ms": 5,
                    "primary_device": "hw:CARD=Headset,DEV=0",
                    "reference_device": "hw:CARD=ErrorMic,DEV=0",
                    "output_device": "hw:CARD=Headset,DEV=0",
                    "period_size": 160,
                    "buffer_size": 640
                },
                "devices": [
                    {
                        "card_index": 0,
                        "device_index": 0,
                        "device_type": "Duplex",
                        "identifier": "hw:0,0",
                        "name": "USB Headset Audio"
                    },
                    {
                        "card_index": 1,
                        "device_index": 0,
                        "device_type": "Capture",
                        "identifier": "hw:1,0",
                        "name": "Reference Error Mic"
                    }
                ]
            }

        elif command == "get_model":
            return {
                "status": "ok",
                "model": {
                    "model_path": "models/onnx/speech_enhancer_int8.onnx",
                    "model_name": "ComplexCRN",
                    "quantization": "INT8 (ARM NEON Optimized)",
                    "execution_provider": "CPUExecutionProvider",
                    "intra_op_threads": 2,
                    "input_shape": ["noisy_stft: (B, 2, T, 257)", "hidden_in: (2, B, 128)"],
                    "output_shape": ["enhanced_stft: (B, 2, T, 257)", "mask: (B, 2, T, 257)", "hidden_out: (2, B, 128)"]
                }
            }

        raise RuntimeUnavailableError(f"ExplicitTestRuntimeStub: Unrecognized command '{command}'")


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_ipc_stub():
    # Ensure test stub is cleared after each test
    yield
    ipc_client.clear_test_stub()


# ==============================================================================
# SECTION 1: DISCONNECTED RUNTIME TESTS (HONEST STATUS REPORTING)
# ==============================================================================

def test_health_reports_degraded_when_runtime_unreachable(client):
    """
    Verifies that when C++ runtime is unreachable, /api/health honestly reports
    status='degraded', runtime_connected=False, and NEVER returns fabricated healthy status.
    """
    # Use a non-listening port to guarantee disconnection
    ipc_client.clear_test_stub()
    ipc_client.tcp_port = 59999  # Guaranteed closed port

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
    Verifies that /api/metrics returns HTTP 503 instead of fabricating telemetry
    when the C++ runtime is not running.
    """
    ipc_client.clear_test_stub()
    ipc_client.tcp_port = 59999

    response = client.get("/api/metrics")
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"].lower()


def test_runtime_controls_return_503_when_runtime_unreachable(client):
    """
    Verifies that POST /api/runtime/* endpoints fail honestly with HTTP 503
    if the C++ runtime cannot be reached.
    """
    ipc_client.clear_test_stub()
    ipc_client.tcp_port = 59999

    for endpoint in ["/api/runtime/start", "/api/runtime/stop", "/api/runtime/bypass", "/api/runtime/reset"]:
        response = client.post(endpoint)
        assert response.status_code == 503, f"Endpoint {endpoint} should return 503 when disconnected"


def test_system_and_audio_fallback_gracefully_when_disconnected(client):
    """
    GET /api/system, /api/audio/devices, and /api/model should continue to serve
    valid hardware configuration and kernel telemetry even if runtime is offline.
    """
    ipc_client.clear_test_stub()
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
# SECTION 2: CONNECTED RUNTIME TESTS (VIA EXPLICIT TEST STUB)
# ==============================================================================

def test_health_reports_healthy_when_connected(client):
    stub = ExplicitTestRuntimeStub(is_alive=True)
    ipc_client.set_test_stub(stub)

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["backend"] == "healthy"
    assert data["runtime_connected"] is True
    assert data["runtime_state"] == "ready"


def test_get_metrics_connected(client):
    stub = ExplicitTestRuntimeStub(is_alive=True)
    ipc_client.set_test_stub(stub)

    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "latencies" in data
    assert data["latencies"]["capture_us"] == 120.0
    assert data["latencies"]["ai_inference_us"] == 1850.0
    assert data["latencies"]["total_processing_us"] == 2680.0
    assert data["rtf"] == 0.268
    assert data["processed_frames"] == 120
    assert data["fusion_mode"] == "NORMAL"
    assert data["current_lambda"] == 0.65
    assert data["alsa_xruns"]["primary"] == 0


def test_get_system_connected(client):
    stub = ExplicitTestRuntimeStub(is_alive=True)
    ipc_client.set_test_stub(stub)

    response = client.get("/api/system")
    assert response.status_code == 200
    data = response.json()

    assert data["cpu"]["overall_pct"] == 24.5
    assert len(data["cpu"]["cores_pct"]) == 4
    assert data["temperature_c"] == 48.2
    assert data["temperature_available"] is True
    assert data["source"] == "c++_runtime"


def test_get_audio_devices_connected(client):
    stub = ExplicitTestRuntimeStub(is_alive=True)
    ipc_client.set_test_stub(stub)

    response = client.get("/api/audio/devices")
    assert response.status_code == 200
    data = response.json()

    assert data["config"]["sample_rate"] == 16000
    assert len(data["devices"]) == 2
    assert data["devices"][0]["name"] == "USB Headset Audio"
    assert data["devices"][1]["name"] == "Reference Error Mic"


def test_get_model_connected(client):
    stub = ExplicitTestRuntimeStub(is_alive=True)
    ipc_client.set_test_stub(stub)

    response = client.get("/api/model")
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "ComplexCRN"
    assert "INT8" in data["quantization"]
    assert data["intra_op_threads"] == 2


def test_runtime_start_stop_bypass_reset_lifecycle(client):
    stub = ExplicitTestRuntimeStub(is_alive=True)
    ipc_client.set_test_stub(stub)

    # 1. Start runtime
    res_start = client.post("/api/runtime/start")
    assert res_start.status_code == 200
    assert res_start.json()["runtime_state"] == "running"
    assert stub.state == "running"

    # 2. Set Bypass
    res_bp = client.post("/api/runtime/bypass", json={"bypass": True})
    assert res_bp.status_code == 200
    assert stub.bypass_mode is True

    # 3. Reset runtime
    res_reset = client.post("/api/runtime/reset")
    assert res_reset.status_code == 200
    assert stub.processed_frames == 0
    assert stub.state == "ready"

    # 4. Stop runtime
    res_stop = client.post("/api/runtime/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["runtime_state"] == "stopped"
    assert stub.state == "stopped"


# ==============================================================================
# SECTION 3: WEBSOCKET STREAMING TESTS
# ==============================================================================

def test_websocket_telemetry_stream(client):
    """
    Connects to /ws/telemetry via Starlette WebSocket TestClient,
    receives real-time telemetry frames, and sends bidirectional ping message.
    """
    stub = ExplicitTestRuntimeStub(is_alive=True)
    ipc_client.set_test_stub(stub)

    with client.websocket_connect("/ws/telemetry") as ws:
        # Receive first telemetry payload
        msg1 = ws.receive_text()
        data1 = json.loads(msg1)

        assert data1["type"] == "telemetry"
        assert data1["status"] == "connected"
        assert "data" in data1
        assert data1["data"]["rtf"] == 0.268
        assert data1["data"]["fusion_mode"] == "NORMAL"

        # Send bidirectional ping message to WebSocket
        ws.send_text(json.dumps({"command": "ping"}))
        pong_msg = ws.receive_text()
        pong_data = json.loads(pong_msg)
        assert pong_data["type"] == "pong"


def test_websocket_reports_offline_when_runtime_unreachable(client):
    """
    Verifies that /ws/telemetry gracefully sends telemetry_offline messages
    without crashing the WebSocket connection when runtime is disconnected.
    """
    ipc_client.clear_test_stub()
    ipc_client.tcp_port = 59999

    with client.websocket_connect("/ws/telemetry") as ws:
        msg = ws.receive_text()
        data = json.loads(msg)

        assert data["type"] == "telemetry_offline"
        assert data["status"] == "disconnected"
        assert "error" in data


# ==============================================================================
# SECTION 4: REAL TCP SOCKET IPC COMMUNICATION TEST
# ==============================================================================

def test_real_tcp_socket_communication():
    """
    Spawns a real TCP server socket on 127.0.0.1, receives commands from RuntimeIpcClient,
    and returns responses, proving end-to-end socket wire protocol without mocks.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))  # Bind to ephemeral available port
    server_sock.listen(5)
    assigned_port = server_sock.getsockname()[1]

    server_running = True

    def socket_worker():
        while server_running:
            try:
                server_sock.settimeout(0.5)
                conn, _ = server_sock.accept()
                with conn:
                    req_bytes = conn.recv(4096)
                    if not req_bytes:
                        continue
                    req_data = json.loads(req_bytes.decode("utf-8").strip())
                    if req_data.get("command") == "ping":
                        resp = json.dumps({"status": "ok", "runtime": "ready"}) + "\n"
                        conn.sendall(resp.encode("utf-8"))
            except socket.timeout:
                continue
            except Exception:
                break
        try:
            server_sock.close()
        except Exception:
            pass

    th = threading.Thread(target=socket_worker, daemon=True)
    th.start()

    # Point ipc_client to this real listening socket
    ipc_client.clear_test_stub()
    ipc_client.tcp_host = "127.0.0.1"
    ipc_client.tcp_port = assigned_port
    ipc_client.timeout = 2.0

    try:
        res = ipc_client.send_command("ping")
        assert res.get("status") == "ok"
        assert res.get("runtime") == "ready"
        assert ipc_client.is_connected() is True
    finally:
        server_running = False
        th.join(timeout=2.0)
