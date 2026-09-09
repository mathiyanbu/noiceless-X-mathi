"""
Structured Session Logger for NOICELESSX.

Every run records an audit trail to logs/<timestamp>/ containing:
  - runtime.jsonl : Per-frame metrics and latency measurements (JSON Lines)
  - metrics.csv   : Tabular time-series for evaluation and plotting
  - errors.log    : Timestamped operational log for errors, warnings, XRUNs, drift
  - system.json   : Complete hardware and system configuration snapshot

Uses a background writer thread with a bounded non-blocking queue so disk I/O
never interrupts real-time audio processing.
"""

import os
import sys
import time
import json
import csv
import queue
import threading
import platform
from typing import Dict, Any, Optional


class SessionLogger:
    """
    Real-time safe structured session logger.
    """

    def __init__(
        self,
        logs_root: str = "logs",
        session_id: Optional[str] = None,
        system_metadata: Optional[Dict[str, Any]] = None,
        max_queue_size: int = 10000
    ):
        self.logs_root = logs_root
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.logs_root, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = True
        self._dropped_log_records = 0

        # File paths
        self.system_path = os.path.join(self.session_dir, "system.json")
        self.runtime_jsonl_path = os.path.join(self.session_dir, "runtime.jsonl")
        self.metrics_csv_path = os.path.join(self.session_dir, "metrics.csv")
        self.errors_log_path = os.path.join(self.session_dir, "errors.log")

        # Open files
        self._jsonl_file = open(self.runtime_jsonl_path, "a", encoding="utf-8", buffering=1)
        self._errors_file = open(self.errors_log_path, "a", encoding="utf-8", buffering=1)
        self._csv_file = open(self.metrics_csv_path, "w", newline="", encoding="utf-8")
        
        self._csv_headers = [
            "timestamp",
            "frame_index",
            "total_processing_us",
            "rtf",
            "estimated_input_snr_db",
            "estimated_output_snr_db",
            "estimated_delta_snr_db",
            "primary_level_dbfs",
            "output_level_dbfs",
            "vad_probability",
            "ai_confidence",
            "impulse_probability",
            "current_lambda",
            "fusion_mode",
            "overall_cpu_pct",
            "cpu_temperature_c",
            "dropped_frames",
            "alsa_xruns_primary",
            "alsa_xruns_playback"
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_headers)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        # Write system snapshot
        self._write_system_json(system_metadata)

        # Start background consumer thread
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        self.log_event("INFO", f"Session logging initialized in {self.session_dir}")

    def _write_system_json(self, metadata: Optional[Dict[str, Any]]):
        """Record complete host and engine configuration snapshot."""
        sys_info = {
            "session_id": self.session_id,
            "session_dir": os.path.abspath(self.session_dir),
            "start_time": time.time(),
            "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count() or 1
            },
            "metadata": metadata or {}
        }

        with open(self.system_path, "w", encoding="utf-8") as f:
            json.dump(sys_info, f, indent=2)

    def log_frame(self, frame_data: Dict[str, Any]):
        """
        Record per-frame telemetry non-blockingly.
        Realtime safe: drops record if queue is completely full to avoid blocking audio threads.
        """
        if not self._running:
            return

        entry = {
            "type": "frame",
            "timestamp": time.time(),
            "data": frame_data
        }
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            self._dropped_log_records += 1

    def log_event(self, level: str, message: str):
        """Record operational log entry (INFO, WARN, ERROR)."""
        if not self._running:
            return

        entry = {
            "type": "event",
            "timestamp": time.time(),
            "level": level.upper(),
            "message": message
        }
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            self._dropped_log_records += 1

    def _writer_loop(self):
        """Background thread handling disk I/O."""
        while self._running or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                item_type = item.get("type")
                t_val = item.get("timestamp", time.time())

                if item_type == "frame":
                    d = item.get("data", {})
                    # Write to JSONL
                    jsonl_record = {"timestamp": t_val, **d}
                    self._jsonl_file.write(json.dumps(jsonl_record) + "\n")

                    # Write row to CSV
                    csv_row = {
                        "timestamp": f"{t_val:.4f}",
                        "frame_index": d.get("processed_frames", 0),
                        "total_processing_us": f"{d.get('total_processing_us', 0.0):.1f}",
                        "rtf": f"{d.get('rtf', 0.0):.4f}",
                        "estimated_input_snr_db": f"{d.get('estimated_input_snr_db', 0.0):.2f}",
                        "estimated_output_snr_db": f"{d.get('estimated_output_snr_db', 0.0):.2f}",
                        "estimated_delta_snr_db": f"{d.get('estimated_snr_improvement_db', 0.0):.2f}",
                        "primary_level_dbfs": f"{d.get('primary_level_dbfs', -96.0):.1f}",
                        "output_level_dbfs": f"{d.get('output_level_dbfs', -96.0):.1f}",
                        "vad_probability": f"{d.get('vad_probability', 0.0):.3f}",
                        "ai_confidence": f"{d.get('ai_confidence', 1.0):.3f}",
                        "impulse_probability": f"{d.get('impulse_probability', 0.0):.3f}",
                        "current_lambda": f"{d.get('current_lambda', 0.5):.3f}",
                        "fusion_mode": str(d.get("fusion_mode", "NORMAL")),
                        "overall_cpu_pct": f"{d.get('overall_cpu_pct', 0.0):.1f}",
                        "cpu_temperature_c": f"{d.get('cpu_temperature_c', 0.0):.1f}",
                        "dropped_frames": d.get("dropped_frames", 0),
                        "alsa_xruns_primary": d.get("alsa_xruns_primary", 0),
                        "alsa_xruns_playback": d.get("alsa_xruns_playback", 0)
                    }
                    self._csv_writer.writerow(csv_row)

                elif item_type == "event":
                    lvl = item.get("level", "INFO")
                    msg = item.get("message", "")
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_val))
                    ms = int((t_val % 1) * 1000)
                    self._errors_file.write(f"[{time_str}.{ms:03d}] [{lvl}] {msg}\n")

                self._queue.task_done()
            except Exception as e:
                try:
                    self._errors_file.write(f"[{time.time()}] [ERROR] Failed to write log item: {e}\n")
                except Exception:
                    pass

    def flush(self):
        """Flush internal buffers to disk."""
        try:
            self._jsonl_file.flush()
            self._csv_file.flush()
            self._errors_file.flush()
        except Exception:
            pass

    def close(self):
        """Stop background worker, flush all remaining items, and close files."""
        self._running = False
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2.0)

        # Final drain
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item.get("type") == "frame":
                    d = item.get("data", {})
                    t_val = item.get("timestamp", time.time())
                    self._jsonl_file.write(json.dumps({"timestamp": t_val, **d}) + "\n")
                elif item.get("type") == "event":
                    t_val = item.get("timestamp", time.time())
                    self._errors_file.write(f"[{t_val}] [{item.get('level')}] {item.get('message')}\n")
            except Exception:
                break

        if self._dropped_log_records > 0:
            self._errors_file.write(f"[{time.time()}] [WARN] Dropped {self._dropped_log_records} log records due to queue overflow\n")

        self.flush()

        try:
            self._jsonl_file.close()
            self._csv_file.close()
            self._errors_file.close()
        except Exception:
            pass
