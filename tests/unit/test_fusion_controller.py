"""
Unit tests for Dual-Mic Fusion Controller and State Machine.
"""

import time
import numpy as np
import pytest
from ai.fusion.fusion_controller import (
    AudioFrame,
    FusionMode,
    FusionConfig,
    FusionInput,
    FusionController
)


def test_normal_mode_dynamic_lambda():
    """Verify NORMAL mode blends AI + NLMS using dynamic lambda."""
    controller = FusionController()
    frame_len = 80
    ai_frame = AudioFrame(samples=np.ones(frame_len, dtype=np.float32))
    nlms_frame = AudioFrame(samples=np.zeros(frame_len, dtype=np.float32))

    # High confidence (0.9) and active speech (0.8) -> lambda = 0.72
    inp = FusionInput(
        ai_output=ai_frame,
        nlms_output=nlms_frame,
        ai_confidence=0.90,
        impulse_probability=0.0,
        vad_probability=0.80,
        nlms_available=True
    )

    out = controller.fuse(inp)

    assert controller.current_mode == FusionMode.NORMAL
    expected_lambda = 0.90 * 0.80
    assert pytest.approx(controller.current_lambda, rel=1e-2) == expected_lambda
    # Since ai is 1.0 and nlms is 0.0, out should be approximately lambda
    np.testing.assert_allclose(out.samples, expected_lambda, atol=0.03)


def test_low_confidence_leans_on_nlms():
    """Verify LOW_CONFIDENCE mode reduces lambda to lean on NLMS."""
    controller = FusionController()
    frame_len = 80
    ai_frame = AudioFrame(samples=np.ones(frame_len, dtype=np.float32))
    nlms_frame = AudioFrame(samples=np.zeros(frame_len, dtype=np.float32))

    # Low AI confidence (0.20 < 0.45)
    inp = FusionInput(
        ai_output=ai_frame,
        nlms_output=nlms_frame,
        ai_confidence=0.20,
        impulse_probability=0.0,
        vad_probability=0.90,
        nlms_available=True
    )

    out = controller.fuse(inp)

    assert controller.current_mode == FusionMode.LOW_CONFIDENCE
    assert controller.current_lambda < 0.25
    # Output leans much closer to NLMS (0.0) than AI (1.0)
    assert np.all(out.samples < 0.25)


def test_impulse_protection_envelope_and_floor():
    """Verify IMPULSE mode fast attack attenuation, configurable floor > 0, and slow recovery."""
    controller = FusionController()
    frame_len = 80
    ai_frame = AudioFrame(samples=np.ones(frame_len, dtype=np.float32))
    nlms_frame = AudioFrame(samples=np.ones(frame_len, dtype=np.float32))

    # Severe impulsive spike (p = 1.0)
    impulse_inp = FusionInput(
        ai_output=ai_frame,
        nlms_output=nlms_frame,
        ai_confidence=0.8,
        impulse_probability=1.0,
        vad_probability=0.5,
        nlms_available=True
    )

    out_impulse = controller.fuse(impulse_inp)

    assert controller.current_mode == FusionMode.IMPULSE
    gain_impulse = controller.current_envelope_gain

    # Fast attack: gain dropped substantially
    assert gain_impulse < 0.50
    # Configurable floor > 0: audio never hard mutes
    assert gain_impulse >= controller.config.impulse_gain_floor
    assert np.all(out_impulse.samples >= controller.config.impulse_gain_floor * 0.99)

    # Impulse ends (p = 0.0)
    quiet_inp = FusionInput(
        ai_output=ai_frame,
        nlms_output=nlms_frame,
        ai_confidence=0.8,
        impulse_probability=0.0,
        vad_probability=0.5,
        nlms_available=True
    )

    # Slower recovery over multiple frames
    prev_gain = gain_impulse
    for _ in range(15):
        controller.fuse(quiet_inp)
        curr_gain = controller.current_envelope_gain
        assert curr_gain > prev_gain
        prev_gain = curr_gain

    assert controller.current_envelope_gain > 0.85


def test_degraded_mode_nlms_only():
    """Verify DEGRADED mode outputs NLMS-only (never silence) when AI fails."""
    controller = FusionController()
    frame_len = 80
    ai_frame = AudioFrame(samples=np.zeros(frame_len, dtype=np.float32))
    nlms_frame = AudioFrame(samples=np.full(frame_len, 0.55, dtype=np.float32))

    inp = FusionInput(
        ai_output=ai_frame,
        nlms_output=nlms_frame,
        ai_confidence=0.0,
        impulse_probability=0.0,
        vad_probability=0.0,
        nlms_available=True,
        ai_available=False  # AI inference failure signal
    )

    out = controller.fuse(inp)

    assert controller.current_mode == FusionMode.DEGRADED
    assert controller.fault_count >= 1
    # Output must equal NLMS residual, never silence
    np.testing.assert_allclose(out.samples, 0.55, atol=1e-3)


def test_nlms_fault_mode_fallback_ai_only():
    """Verify NLMS_FAULT mode falls back to AI-only, logs fault, and keeps impulse protection."""
    controller = FusionController()
    frame_len = 80
    ai_frame = AudioFrame(samples=np.full(frame_len, 0.85, dtype=np.float32))
    nlms_frame = AudioFrame(samples=np.zeros(frame_len, dtype=np.float32))

    inp = FusionInput(
        ai_output=ai_frame,
        nlms_output=nlms_frame,
        ai_confidence=0.9,
        impulse_probability=0.0,
        vad_probability=0.9,
        nlms_available=False  # Reference mic dropped out / drift error
    )

    out = controller.fuse(inp)

    assert controller.current_mode == FusionMode.NLMS_FAULT
    assert controller.fault_count == 1
    assert "Reference" in controller.last_fault_message

    # AI-only fallback output
    np.testing.assert_allclose(out.samples, 0.85, atol=1e-3)


def test_bypass_mode_latency_and_passthrough():
    """Verify BYPASS mode routes input directly with trivial pass-through delay."""
    controller = FusionController()
    frame_len = 80
    raw_samples = np.linspace(-0.5, 0.5, frame_len, dtype=np.float32)
    raw_frame = AudioFrame(samples=raw_samples)
    dummy_ai = AudioFrame(samples=np.zeros(frame_len, dtype=np.float32))
    dummy_nlms = AudioFrame(samples=np.zeros(frame_len, dtype=np.float32))

    inp = FusionInput(
        ai_output=dummy_ai,
        nlms_output=dummy_nlms,
        raw_input=raw_frame,
        bypass_requested=True
    )

    # Measure latency
    t0 = time.perf_counter_ns()
    out = controller.fuse(inp)
    t1 = time.perf_counter_ns()
    latency_us = (t1 - t0) / 1000.0

    assert controller.current_mode == FusionMode.BYPASS
    assert latency_us < 200.0  # Trivial pass-through latency (< 200us)
    np.testing.assert_array_equal(out.samples, raw_samples)


def test_error_mode_hardware_unavailable():
    """Verify ERROR mode reports error and produces zero processing."""
    controller = FusionController()
    frame_len = 80

    inp = FusionInput(
        ai_output=AudioFrame(samples=np.ones(frame_len, dtype=np.float32)),
        nlms_output=AudioFrame(samples=np.ones(frame_len, dtype=np.float32)),
        hardware_available=False
    )

    out = controller.fuse(inp)

    assert controller.current_mode == FusionMode.ERROR
    assert controller.fault_count >= 1
    np.testing.assert_array_equal(out.samples, np.zeros(frame_len, dtype=np.float32))


def test_state_machine_sequential_transitions():
    """Verify sequential transitions across all state machine modes."""
    controller = FusionController()
    frame = AudioFrame(samples=np.full(80, 0.5, dtype=np.float32))

    # 1. NORMAL
    c1 = FusionInput(ai_output=frame, nlms_output=frame, ai_confidence=0.9, impulse_probability=0.05, vad_probability=0.8)
    controller.fuse(c1)
    assert controller.current_mode == FusionMode.NORMAL

    # 2. LOW_CONFIDENCE
    c2 = FusionInput(ai_output=frame, nlms_output=frame, ai_confidence=0.2, impulse_probability=0.05, vad_probability=0.8)
    controller.fuse(c2)
    assert controller.current_mode == FusionMode.LOW_CONFIDENCE

    # 3. IMPULSE
    c3 = FusionInput(ai_output=frame, nlms_output=frame, ai_confidence=0.9, impulse_probability=0.95, vad_probability=0.8)
    controller.fuse(c3)
    assert controller.current_mode == FusionMode.IMPULSE

    # 4. Return to NORMAL
    controller.fuse(c1)
    assert controller.current_mode == FusionMode.NORMAL

    # 5. NLMS_FAULT
    c5 = FusionInput(ai_output=frame, nlms_output=frame, ai_confidence=0.9, impulse_probability=0.05, vad_probability=0.8, nlms_available=False)
    controller.fuse(c5)
    assert controller.current_mode == FusionMode.NLMS_FAULT

    # 6. Reconnect -> NORMAL
    controller.fuse(c1)
    assert controller.current_mode == FusionMode.NORMAL

    # 7. AI fails -> DEGRADED
    c7 = FusionInput(ai_output=frame, nlms_output=frame, ai_confidence=0.0, impulse_probability=0.0, vad_probability=0.0, ai_available=False)
    controller.fuse(c7)
    assert controller.current_mode == FusionMode.DEGRADED

    # 8. Operator BYPASS
    controller.set_bypass(True)
    controller.fuse(c1)
    assert controller.current_mode == FusionMode.BYPASS

    # 9. Release bypass -> NORMAL
    controller.set_bypass(False)
    controller.fuse(c1)
    assert controller.current_mode == FusionMode.NORMAL
