#!/usr/bin/env python3
"""
Python Real-Time Dual-Microphone Pipeline & System Metrics Engine.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from ai.fusion.fusion_controller import (
    AudioFrame,
    FusionMode,
    FusionConfig,
    FusionInput,
    FusionController
)


@dataclass
class PipelineTelemetry:
    capture_us: float = 0.0
    preprocessing_us: float = 0.0
    stft_us: float = 0.0
    ai_inference_us: float = 0.0
    nlms_us: float = 0.0
    fusion_us: float = 0.0
    istft_us: float = 0.0
    playback_queue_us: float = 0.0
    total_processing_us: float = 0.0
    end_to_end_latency_ms: float = 0.0
    rtf: float = 0.0

    processed_frames: int = 0
    dropped_frames: int = 0
    alsa_xruns_primary: int = 0
    alsa_xruns_reference: int = 0
    alsa_xruns_playback: int = 0

    drift_ms: float = 0.0
    drift_samples: int = 0
    drift_warning: bool = False

    fusion_mode: FusionMode = FusionMode.NORMAL
    current_lambda: float = 0.5
    impulse_envelope_gain: float = 1.0
    ai_confidence: float = 1.0
    impulse_probability: float = 0.0
    vad_probability: float = 0.0

    cpu_per_core: List[float] = field(default_factory=list)
    overall_cpu_pct: float = 0.0
    cpu_temperature_c: float = 0.0
    temp_available: bool = False


class SystemMetrics:
    """Reads real Linux CPU utilization from /proc/stat and thermal files."""

    def __init__(self):
        self.per_core_cpu: List[float] = []
        self.overall_cpu: float = 0.0
        self.temperature_c: float = 0.0
        self.temp_available: bool = False
        self._prev_total = None
        self._prev_cores = []
        self.sample()

    def sample(self):
        self._read_proc_stat()
        self._read_temperature()

    def _read_proc_stat(self):
        if not os.path.exists("/proc/stat"):
            if not self.per_core_cpu:
                self.per_core_cpu = [0.0, 0.0, 0.0, 0.0]
            return

        try:
            with open("/proc/stat", "r") as f:
                lines = f.readlines()

            current_cores = []
            current_total = None

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                label = parts[0]
                if label == "cpu":
                    values = [int(x) for x in parts[1:9]]
                    current_total = values
                elif label.startswith("cpu") and len(label) > 3:
                    values = [int(x) for x in parts[1:9]]
                    current_cores.append(values)

            if self._prev_total is None:
                self._prev_total = current_total
                self._prev_cores = current_cores
                self.per_core_cpu = [0.0] * len(current_cores)
                return

            if current_total and self._prev_total:
                total_delta = sum(current_total) - sum(self._prev_total)
                idle_delta = (current_total[3] + current_total[4]) - (self._prev_total[3] + self._prev_total[4])
                if total_delta > 0:
                    self.overall_cpu = float(np.clip(100.0 * (1.0 - (idle_delta / total_delta)), 0.0, 100.0))
                self._prev_total = current_total

            if len(current_cores) == len(self._prev_cores):
                self.per_core_cpu = []
                for curr, prev in zip(current_cores, self._prev_cores):
                    td = sum(curr) - sum(prev)
                    id_delta = (curr[3] + curr[4]) - (prev[3] + prev[4])
                    if td > 0:
                        pct = float(np.clip(100.0 * (1.0 - (id_delta / td)), 0.0, 100.0))
                    else:
                        pct = 0.0
                    self.per_core_cpu.append(pct)
                self._prev_cores = current_cores
        except Exception:
            pass

    def _read_temperature(self):
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_path):
            try:
                with open(thermal_path, "r") as f:
                    raw = int(f.read().strip())
                    self.temperature_c = raw / 1000.0
                    self.temp_available = True
                    return
            except Exception:
                pass

        self.temperature_c = 0.0
        self.temp_available = False


class RealtimePipeline:
    """Real-time audio processing pipeline orchestrator."""

    def __init__(self, sample_rate: int = 16000, hop_size: int = 80, fft_size: int = 512):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size

        self.fusion = FusionController(FusionConfig(sample_rate=sample_rate, default_frame_size=hop_size))
        self.metrics = SystemMetrics()
        self.telemetry = PipelineTelemetry()
        self.is_running = False

    def process_hop(
        self,
        primary_samples: np.ndarray,
        reference_samples: np.ndarray,
        ai_enhanced_samples: Optional[np.ndarray] = None,
        ai_confidence: float = 0.90,
        impulse_prob: float = 0.0,
        vad_prob: float = 0.85,
        drift_ms: float = 0.0
    ) -> np.ndarray:
        t0 = time.perf_counter_ns()

        n = min(len(primary_samples), self.hop_size)

        # 1. Preprocessing: DC removal and simple high-pass simulation
        t_prep_0 = time.perf_counter_ns()
        preproc = primary_samples[:n] - np.mean(primary_samples[:n])
        t_prep_1 = time.perf_counter_ns()
        self.telemetry.preprocessing_us = (t_prep_1 - t_prep_0) / 1000.0

        # 2. STFT Spectral Analysis
        t_stft_0 = time.perf_counter_ns()
        # Simulated or actual STFT analysis
        spec = np.fft.rfft(preproc, n=self.fft_size)
        t_stft_1 = time.perf_counter_ns()
        self.telemetry.stft_us = (t_stft_1 - t_stft_0) / 1000.0

        # 3. AI branch
        t_ai_0 = time.perf_counter_ns()
        if ai_enhanced_samples is not None and len(ai_enhanced_samples) >= n:
            ai_out = ai_enhanced_samples[:n]
        else:
            ai_out = preproc.copy()
        t_ai_1 = time.perf_counter_ns()
        self.telemetry.ai_inference_us = (t_ai_1 - t_ai_0) / 1000.0

        # 4. iSTFT
        t_istft_0 = time.perf_counter_ns()
        # Synthesis timing
        t_istft_1 = time.perf_counter_ns()
        self.telemetry.istft_us = (t_istft_1 - t_istft_0) / 1000.0

        # 5. NLMS Branch
        t_nlms_0 = time.perf_counter_ns()
        nlms_available = (reference_samples is not None) and (len(reference_samples) >= n) and (abs(drift_ms) <= 10.0)
        if nlms_available:
            ref = reference_samples[:n]
            # Simple normalized cancellation residual for test scaffolding
            nlms_out = preproc - 0.5 * ref
        else:
            nlms_out = preproc.copy()
        t_nlms_1 = time.perf_counter_ns()
        self.telemetry.nlms_us = (t_nlms_1 - t_nlms_0) / 1000.0

        # 6. Fusion Controller
        t_fuse_0 = time.perf_counter_ns()
        inp = FusionInput(
            ai_output=AudioFrame(samples=ai_out),
            nlms_output=AudioFrame(samples=nlms_out),
            ai_confidence=ai_confidence,
            impulse_probability=impulse_prob,
            vad_probability=vad_prob,
            nlms_available=nlms_available,
            raw_input=AudioFrame(samples=primary_samples[:n])
        )
        fused_frame = self.fusion.fuse(inp)
        t_fuse_1 = time.perf_counter_ns()
        self.telemetry.fusion_us = (t_fuse_1 - t_fuse_0) / 1000.0

        t_end = time.perf_counter_ns()
        total_us = (t_end - t0) / 1000.0
        hop_us = (n / self.sample_rate) * 1e6
        rtf = total_us / hop_us if hop_us > 0 else 0.0

        self.telemetry.total_processing_us = total_us
        self.telemetry.rtf = rtf
        self.telemetry.processed_frames += 1
        self.telemetry.fusion_mode = self.fusion.current_mode
        self.telemetry.current_lambda = self.fusion.current_lambda
        self.telemetry.impulse_envelope_gain = self.fusion.current_envelope_gain
        self.telemetry.drift_ms = drift_ms

        return fused_frame.samples

    def get_telemetry(self) -> PipelineTelemetry:
        self.metrics.sample()
        self.telemetry.cpu_per_core = self.metrics.per_core_cpu
        self.telemetry.overall_cpu_pct = self.metrics.overall_cpu
        self.telemetry.cpu_temperature_c = self.metrics.temperature_c
        self.telemetry.temp_available = self.metrics.temp_available
        return self.telemetry
