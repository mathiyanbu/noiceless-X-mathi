"""
Backend configuration and settings loader for NOICELESSX.
"""

import os
from pathlib import Path
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent


class BackendSettings(BaseModel):
    app_name: str = "SIH26052 NOICELESSX API"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    config_yaml_path: str = str(REPO_ROOT / "config" / "raspberrypi.yaml")
    
    # IPC parameters to C++ runtime
    ipc_unix_socket: str = "/tmp/noiselessx.sock"
    ipc_tcp_host: str = "127.0.0.1"
    ipc_tcp_port: int = 9099
    ipc_timeout_sec: float = 1.0

    # Telemetry streaming
    telemetry_ws_endpoint: str = "/ws/telemetry"
    ws_push_interval_ms: int = 50  # 20 Hz


settings = BackendSettings()
