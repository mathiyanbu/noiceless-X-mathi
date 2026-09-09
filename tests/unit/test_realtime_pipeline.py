"""
Unit tests for Real-Time Pipeline and System Metrics Engine.
"""

import numpy as np
import pytest
from ai.runtime.realtime_pipeline import (
    PipelineTelemetry,
    SystemMetrics,
    RealtimePipeline
)
from ai.fusion.fusion_controller import FusionMode


def test_system_metrics_sampling():
    """Verify system metrics sampling returns bounded percentages."""
    metrics = SystemMetrics()
    metrics.sample()

    assert 0.0 <= metrics.overall_cpu <= 100.0
    for core in metrics.per_core_cpu:
        assert 0.0 <= core <= 100.0

    if metrics.temp_available:
        assert 0.0 < metrics.temperature_c < 110.0


def test_realtime_pipeline_process_hop():
    """Verify hop processing, latency recording, and RTF computation."""
    pipeline = RealtimePipeline(sample_rate=16000, hop_size=80)
    hop_len = 80

    primary = np.sin(2 * np.pi * 440.0 * np.arange(hop_len) / 16000.0).astype(np.float32)
    reference = np.random.normal(0, 0.1, hop_len).astype(np.float32)

    out = pipeline.process_hop(
        primary_samples=primary,
        reference_samples=reference,
        ai_enhanced_samples=primary * 0.9,
        ai_confidence=0.88,
        impulse_prob=0.05,
        vad_prob=0.80,
        drift_ms=0.5
    )

    assert len(out) == hop_len
    telemetry = pipeline.get_telemetry()

    assert telemetry.processed_frames == 1
    assert telemetry.total_processing_us > 0.0
    assert telemetry.rtf >= 0.0
    assert telemetry.preprocessing_us >= 0.0
    assert telemetry.stft_us >= 0.0
    assert telemetry.fusion_us >= 0.0
    assert telemetry.fusion_mode == FusionMode.NORMAL


def test_realtime_pipeline_drift_fallback():
    """Verify clock drift > 10ms triggers NLMS_FAULT fallback."""
    pipeline = RealtimePipeline(sample_rate=16000, hop_size=80)
    hop_len = 80

    primary = np.ones(hop_len, dtype=np.float32)
    reference = np.ones(hop_len, dtype=np.float32)

    # Exceed drift threshold (15.0ms > 10.0ms)
    out = pipeline.process_hop(
        primary_samples=primary,
        reference_samples=reference,
        drift_ms=15.0
    )

    assert len(out) == hop_len
    telemetry = pipeline.get_telemetry()
    assert telemetry.fusion_mode == FusionMode.NLMS_FAULT
