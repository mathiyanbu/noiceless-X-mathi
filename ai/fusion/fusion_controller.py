#!/usr/bin/env python3
"""
Dual-Microphone Fusion Controller with Explicit State Machine.
Fuses AI Speech Enhancement output with NLMS Adaptive Cancellation output.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Optional, Callable, List
import numpy as np


class FusionMode(Enum):
    NORMAL = "NORMAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    IMPULSE = "IMPULSE"
    DEGRADED = "DEGRADED"
    NLMS_FAULT = "NLMS_FAULT"
    BYPASS = "BYPASS"
    ERROR = "ERROR"


@dataclass
class AudioFrame:
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    frame_index: int = 0
    timestamp_ns: int = 0

    def __post_init__(self):
        if not isinstance(self.samples, np.ndarray):
            self.samples = np.asarray(self.samples, dtype=np.float32)
        elif self.samples.dtype != np.float32:
            self.samples = self.samples.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __setitem__(self, idx, val):
        self.samples[idx] = val


@dataclass
class FusionConfig:
    sample_rate: float = 16000.0
    default_frame_size: int = 80
    lambda_min: float = 0.10
    lambda_max: float = 0.95
    low_confidence_threshold: float = 0.45
    low_confidence_scale: float = 0.25
    impulse_threshold: float = 0.60
    impulse_beta: float = 0.90
    impulse_gain_floor: float = 0.10
    attack_alpha: float = 0.85
    release_alpha: float = 0.12
    log_faults: bool = True


@dataclass
class FusionInput:
    ai_output: AudioFrame
    nlms_output: AudioFrame
    ai_confidence: float = 1.0
    impulse_probability: float = 0.0
    vad_probability: float = 0.0
    nlms_available: bool = True
    raw_input: Optional[AudioFrame] = None
    ai_available: bool = True
    hardware_available: bool = True
    bypass_requested: bool = False


class FusionController:
    """
    Dual-Mic Real-time Speech Enhancement & Noise Cancellation Fusion Engine.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.current_mode = FusionMode.NORMAL
        self.previous_mode = FusionMode.NORMAL
        self.current_lambda = 0.5
        self.current_envelope_gain = 1.0
        self.previous_envelope_gain = 1.0
        self.manual_bypass = False
        self.manual_error = False
        self.frame_count = 0
        self.fault_count = 0
        self.last_fault_message = ""
        self.last_processing_time_us = 0.0
        self.fault_logger: Optional[Callable[[FusionMode, str, int], None]] = None

    def reset(self):
        self.current_mode = FusionMode.NORMAL
        self.previous_mode = FusionMode.NORMAL
        self.current_lambda = 0.5
        self.current_envelope_gain = 1.0
        self.previous_envelope_gain = 1.0
        self.manual_bypass = False
        self.manual_error = False
        self.frame_count = 0
        self.fault_count = 0
        self.last_fault_message = ""
        self.last_processing_time_us = 0.0

    def set_bypass(self, bypass: bool):
        self.manual_bypass = bypass

    @property
    def bypass(self) -> bool:
        return self.manual_bypass

    @bypass.setter
    def bypass(self, value: bool):
        self.manual_bypass = value

    def set_manual_error(self, error: bool):
        self.manual_error = error

    def evaluate_mode(self, inp: FusionInput) -> FusionMode:
        # 1. Unrecoverable hardware failure / both paths down
        if self.manual_error or not inp.hardware_available or (not inp.ai_available and not inp.nlms_available):
            return FusionMode.ERROR

        # 2. Operator-triggered or unrecoverable bypass
        if self.manual_bypass or inp.bypass_requested:
            return FusionMode.BYPASS

        # 3. Component faults
        if not inp.ai_available:
            return FusionMode.DEGRADED

        if not inp.nlms_available:
            return FusionMode.NLMS_FAULT

        # 4. Signal condition heuristics
        if inp.impulse_probability >= self.config.impulse_threshold:
            return FusionMode.IMPULSE

        if inp.ai_confidence < self.config.low_confidence_threshold:
            return FusionMode.LOW_CONFIDENCE

        return FusionMode.NORMAL

    def fuse(self, inp: FusionInput) -> AudioFrame:
        t0 = time.perf_counter_ns()
        self.frame_count += 1

        next_mode = self.evaluate_mode(inp)
        if next_mode != self.current_mode:
            self.previous_mode = self.current_mode
            self.current_mode = next_mode

            if self.current_mode == FusionMode.NLMS_FAULT:
                self._log_fault(FusionMode.NLMS_FAULT, "Reference (error) mic dropped out or drift exceeded tolerance.")
            elif self.current_mode == FusionMode.DEGRADED:
                self._log_fault(FusionMode.DEGRADED, "AI speech enhancement inference failing.")
            elif self.current_mode == FusionMode.ERROR:
                self._log_fault(FusionMode.ERROR, "Audio hardware or dual-mic pipeline completely unavailable.")

        # Compute target impulse attenuation gain: g_I = 1 - beta * impulse_probability
        p_impulse = float(np.clip(inp.impulse_probability, 0.0, 1.0))
        target_gain = max(1.0 - (self.config.impulse_beta * p_impulse), self.config.impulse_gain_floor)

        # Dispatch through state machine
        if self.current_mode == FusionMode.NORMAL:
            out_samples = self._process_normal(inp, target_gain)
        elif self.current_mode == FusionMode.LOW_CONFIDENCE:
            out_samples = self._process_low_confidence(inp, target_gain)
        elif self.current_mode == FusionMode.IMPULSE:
            out_samples = self._process_impulse(inp, target_gain)
        elif self.current_mode == FusionMode.DEGRADED:
            out_samples = self._process_degraded(inp, target_gain)
        elif self.current_mode == FusionMode.NLMS_FAULT:
            out_samples = self._process_nlms_fault(inp, target_gain)
        elif self.current_mode == FusionMode.BYPASS:
            out_samples = self._process_bypass(inp)
        elif self.current_mode == FusionMode.ERROR:
            out_samples = self._process_error(inp)
        else:
            out_samples = self._process_normal(inp, target_gain)

        t1 = time.perf_counter_ns()
        self.last_processing_time_us = (t1 - t0) / 1000.0

        ts = inp.ai_output.timestamp_ns if (inp.ai_output and inp.ai_output.timestamp_ns != 0) else t1
        return AudioFrame(samples=out_samples, frame_index=self.frame_count, timestamp_ns=ts)

    def _compute_dynamic_lambda(self, ai_conf: float, vad_prob: float, low_conf: bool = False) -> float:
        conf = float(np.clip(ai_conf, 0.0, 1.0))
        vad = float(np.clip(vad_prob, 0.0, 1.0))
        l_min, l_max = self.config.lambda_min, self.config.lambda_max
        lam = float(np.clip(conf * vad, l_min, l_max))

        if low_conf:
            lam = float(np.clip(lam * self.config.low_confidence_scale, l_min, l_min + (l_max - l_min) * 0.25))

        return lam

    def _update_gain_envelope(self, target_gain: float):
        self.previous_envelope_gain = self.current_envelope_gain
        if target_gain < self.current_envelope_gain:
            # Fast attack
            self.current_envelope_gain = (self.config.attack_alpha * target_gain) + (
                (1.0 - self.config.attack_alpha) * self.current_envelope_gain
            )
        else:
            # Slower release
            self.current_envelope_gain = (self.config.release_alpha * target_gain) + (
                (1.0 - self.config.release_alpha) * self.current_envelope_gain
            )
        self.current_envelope_gain = float(np.clip(self.current_envelope_gain, self.config.impulse_gain_floor, 1.0))

    def _apply_gain_envelope(self, samples: np.ndarray) -> np.ndarray:
        n = len(samples)
        if n == 0:
            return samples
        if n == 1:
            return samples * self.current_envelope_gain
        envelope = np.linspace(self.previous_envelope_gain, self.current_envelope_gain, n, dtype=np.float32)
        return samples * envelope

    def _get_frame_size(self, inp: FusionInput) -> int:
        s_ai = len(inp.ai_output.samples) if inp.ai_output is not None else 0
        s_nlms = len(inp.nlms_output.samples) if inp.nlms_output is not None else 0
        m = max(s_ai, s_nlms)
        return m if m > 0 else self.config.default_frame_size

    def _process_normal(self, inp: FusionInput, target_gain: float) -> np.ndarray:
        n = self._get_frame_size(inp)
        ai = inp.ai_output.samples if len(inp.ai_output.samples) == n else np.resize(inp.ai_output.samples, n)
        nlms = inp.nlms_output.samples if len(inp.nlms_output.samples) == n else np.resize(inp.nlms_output.samples, n)

        self.current_lambda = self._compute_dynamic_lambda(inp.ai_confidence, inp.vad_probability, False)
        fused = (self.current_lambda * ai) + ((1.0 - self.current_lambda) * nlms)

        self._update_gain_envelope(target_gain)
        return self._apply_gain_envelope(fused)

    def _process_low_confidence(self, inp: FusionInput, target_gain: float) -> np.ndarray:
        n = self._get_frame_size(inp)
        ai = inp.ai_output.samples if len(inp.ai_output.samples) == n else np.resize(inp.ai_output.samples, n)
        nlms = inp.nlms_output.samples if len(inp.nlms_output.samples) == n else np.resize(inp.nlms_output.samples, n)

        self.current_lambda = self._compute_dynamic_lambda(inp.ai_confidence, inp.vad_probability, True)
        fused = (self.current_lambda * ai) + ((1.0 - self.current_lambda) * nlms)

        self._update_gain_envelope(target_gain)
        return self._apply_gain_envelope(fused)

    def _process_impulse(self, inp: FusionInput, target_gain: float) -> np.ndarray:
        n = self._get_frame_size(inp)
        ai = inp.ai_output.samples if len(inp.ai_output.samples) == n else np.resize(inp.ai_output.samples, n)
        nlms = inp.nlms_output.samples if len(inp.nlms_output.samples) == n else np.resize(inp.nlms_output.samples, n)

        self.current_lambda = self._compute_dynamic_lambda(inp.ai_confidence, inp.vad_probability, False)
        if inp.nlms_available:
            fused = (self.current_lambda * ai) + ((1.0 - self.current_lambda) * nlms)
        else:
            fused = ai.copy()

        self._update_gain_envelope(target_gain)
        return self._apply_gain_envelope(fused)

    def _process_degraded(self, inp: FusionInput, target_gain: float) -> np.ndarray:
        # AI inference failing -> output NLMS-only (never silence)
        self.current_lambda = 0.0
        n = len(inp.nlms_output.samples) if len(inp.nlms_output.samples) > 0 else self.config.default_frame_size
        nlms = inp.nlms_output.samples.copy() if len(inp.nlms_output.samples) == n else np.zeros(n, dtype=np.float32)

        self._update_gain_envelope(target_gain)
        return self._apply_gain_envelope(nlms)

    def _process_nlms_fault(self, inp: FusionInput, target_gain: float) -> np.ndarray:
        # Reference mic dropped out / drift error -> fall back to AI-only
        self.current_lambda = 1.0
        n = len(inp.ai_output.samples) if len(inp.ai_output.samples) > 0 else self.config.default_frame_size
        ai = inp.ai_output.samples.copy() if len(inp.ai_output.samples) == n else np.zeros(n, dtype=np.float32)

        self._update_gain_envelope(target_gain)
        return self._apply_gain_envelope(ai)

    def _process_bypass(self, inp: FusionInput) -> np.ndarray:
        # Input routed directly to output with zero modification
        if inp.raw_input is not None and len(inp.raw_input.samples) > 0:
            return inp.raw_input.samples.copy()
        if inp.nlms_output is not None and len(inp.nlms_output.samples) > 0:
            return inp.nlms_output.samples.copy()
        if inp.ai_output is not None and len(inp.ai_output.samples) > 0:
            return inp.ai_output.samples.copy()
        return np.zeros(self.config.default_frame_size, dtype=np.float32)

    def _process_error(self, inp: FusionInput) -> np.ndarray:
        n = self._get_frame_size(inp)
        return np.zeros(n, dtype=np.float32)

    def _log_fault(self, mode: FusionMode, reason: str):
        self.fault_count += 1
        self.last_fault_message = reason
        if self.fault_logger:
            self.fault_logger(mode, reason, self.frame_count)
