"""
Unit tests for Structured Session Logger (Phase 13).
"""

import os
import json
import csv
import tempfile
import time
import pytest

from ai.runtime.session_logger import SessionLogger
from ai.runtime.realtime_pipeline import RealtimePipeline
import numpy as np


def test_session_logger_creates_all_artifacts():
    """Verify SessionLogger generates system.json, runtime.jsonl, metrics.csv, errors.log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(
            logs_root=tmpdir,
            session_id="test_session_001",
            system_metadata={"sample_rate": 16000, "model": "ComplexCRN_INT8"}
        )

        assert os.path.exists(logger.session_dir)
        assert os.path.exists(logger.system_path)
        assert os.path.exists(logger.runtime_jsonl_path)
        assert os.path.exists(logger.metrics_csv_path)
        assert os.path.exists(logger.errors_log_path)

        # Verify system.json contents
        with open(logger.system_path, "r", encoding="utf-8") as f:
            sys_data = json.load(f)

        assert sys_data["session_id"] == "test_session_001"
        assert "platform" in sys_data
        assert sys_data["metadata"]["model"] == "ComplexCRN_INT8"

        # Log 10 simulated frames
        for i in range(10):
            frame = {
                "processed_frames": i + 1,
                "total_processing_us": 1250.0 + i * 10,
                "rtf": 0.25,
                "estimated_input_snr_db": 12.5,
                "estimated_output_snr_db": 22.0,
                "estimated_snr_improvement_db": 9.5,
                "primary_level_dbfs": -24.0,
                "output_level_dbfs": -20.0,
                "vad_probability": 0.85,
                "ai_confidence": 0.95,
                "impulse_probability": 0.01,
                "current_lambda": 0.65,
                "fusion_mode": "NORMAL",
                "overall_cpu_pct": 15.2,
                "cpu_temperature_c": 48.5,
                "dropped_frames": 0,
                "alsa_xruns_primary": 0,
                "alsa_xruns_playback": 0
            }
            logger.log_frame(frame)

        # Log events
        logger.log_event("INFO", "Real-time stream started")
        logger.log_event("WARN", "Clock drift 0.2ms observed")

        # Flush and close
        logger.close()

        # Check runtime.jsonl lines
        with open(logger.runtime_jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 10
        assert lines[0]["processed_frames"] == 1
        assert lines[9]["processed_frames"] == 10
        assert lines[0]["estimated_input_snr_db"] == 12.5

        # Check metrics.csv
        with open(logger.metrics_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 10
        assert rows[0]["frame_index"] == "1"
        assert rows[0]["fusion_mode"] == "NORMAL"
        assert rows[0]["estimated_delta_snr_db"] == "9.50"

        # Check errors.log
        with open(logger.errors_log_path, "r", encoding="utf-8") as f:
            error_content = f.read()

        assert "Real-time stream started" in error_content
        assert "Clock drift 0.2ms observed" in error_content


def test_pipeline_integration_with_session_logger():
    """Verify RealtimePipeline seamlessly logs frames when session logging is enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = RealtimePipeline(sample_rate=16000, hop_size=80)
        s_logger = pipeline.enable_session_logging(logs_root=tmpdir, session_id="pipeline_test")
        pipeline.start()

        hop_len = 80
        sig_primary = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.005, hop_len)).astype(np.float32)
        sig_ref = np.random.normal(0, 0.05, hop_len).astype(np.float32)

        for _ in range(5):
            pipeline.process_hop(
                primary_samples=sig_primary,
                reference_samples=sig_ref,
                ai_confidence=0.88,
                impulse_prob=0.02,
                vad_prob=0.80
            )

        pipeline.stop()

        # Confirm files were written
        assert os.path.exists(os.path.join(s_logger.session_dir, "runtime.jsonl"))
        with open(os.path.join(s_logger.session_dir, "runtime.jsonl"), "r", encoding="utf-8") as f:
            logged_lines = [l for l in f if l.strip()]

        assert len(logged_lines) == 5
