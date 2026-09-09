"""
Local IPC Client connecting FastAPI to the C++ runtime via Unix Domain Socket / Loopback TCP.
"""

import json
import os
import socket
from typing import Optional, Dict, Any

from backend.config import settings


class RuntimeUnavailableError(Exception):
    """Raised when the embedded C++ runtime is not running or unreachable."""
    pass


class RuntimeIpcClient:
    """
    Communicates with the C++ runtime over local Unix Domain Socket or TCP loopback.
    Never transmits real-time audio samples: only telemetry queries and control commands.
    """

    def __init__(
        self,
        unix_socket_path: str = settings.ipc_unix_socket,
        tcp_host: str = settings.ipc_tcp_host,
        tcp_port: int = settings.ipc_tcp_port,
        timeout: float = settings.ipc_timeout_sec
    ):
        self.unix_socket_path = unix_socket_path
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.timeout = timeout
        self._test_stub = None

    def set_test_stub(self, stub):
        """Used strictly in tests to simulate runtime responses with explicit labeling."""
        self._test_stub = stub

    def clear_test_stub(self):
        self._test_stub = None

    def _create_connection(self) -> socket.socket:
        # 1. Prefer Unix Domain Socket on Linux if socket file exists
        if os.name != "nt" and os.path.exists(self.unix_socket_path):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect(self.unix_socket_path)
                return sock
            except OSError:
                pass

        # 2. Connect to localhost TCP loopback
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.tcp_host, self.tcp_port))
            return sock
        except OSError as e:
            raise RuntimeUnavailableError(
                f"C++ runtime IPC unreachable at {self.unix_socket_path} or {self.tcp_host}:{self.tcp_port}: {e}"
            ) from e

    def send_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Send a command payload to the C++ runtime and parse the JSON response.
        """
        if self._test_stub is not None:
            return self._test_stub.handle_command(command, **kwargs)

        sock = self._create_connection()
        try:
            payload = {"command": command, **kwargs}
            msg = json.dumps(payload) + "\n"
            sock.sendall(msg.encode("utf-8"))

            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in response_data:
                    break

            if not response_data:
                raise RuntimeUnavailableError("C++ runtime closed IPC socket without response.")

            return json.loads(response_data.decode("utf-8").strip())
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeUnavailableError(f"IPC communication error: {e}") from e
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def is_connected(self) -> bool:
        """Probe runtime connectivity with a lightweight ping command."""
        if self._test_stub is not None:
            return self._test_stub.is_alive()

        try:
            res = self.send_command("ping")
            return res.get("status") == "ok"
        except RuntimeUnavailableError:
            return False


# Global singleton client instance
ipc_client = RuntimeIpcClient()
