#!/usr/bin/env python3
"""
SIH26052 — NOICELESSX: Physical Raspberry Pi Realtime Stress Test Runner.
Target: Raspberry Pi 4/5 (ARM64 Linux, Debian 12).

Usage:
  # Quick 15-second sanity check with CPU stress & simulated faults:
  python tools/stress_runner.py --duration 15 --stress-cpu --simulate-faults

  # Full 30-minute physical certification run on Raspberry Pi:
  python tools/stress_runner.py --duration 1800 --stress-cpu
"""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.runtime.realtime_pipeline import RealtimePipeline, PipelineTelemetry
from ai.fusion.fusion_controller import FusionMode, FusionInput, AudioFrame


def _cpu_worker(stop_event):
    """Spins CPU to simulate background load when stress-ng is not available."""
    while not stop_event.is_set():
        _ = [i * i for i in range(2000)]


def run_stress_test(
    duration_sec: float = 1800.0,
    stress_cpu: bool = False,
    simulate_faults: bool = False,
    log_dir: str = "logs/stress"
) -> dict:
    print("================================================================================")
    print("      SIH26052 NOICELESSX — PHYSICAL REALTIME STRESS TEST RUNNER               ")
    print(f"      Target: Raspberry Pi 4/5 (ARM64) | Duration: {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
    print("================================================================================")

    sample_rate = 16000
    hop_size = 80  # 5ms @ 16kHz
    hop_duration_sec = hop_size / sample_rate
    target_frames = int(duration_sec / hop_duration_sec)

    out_dir = Path(log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spawn CPU stress if requested
    cpu_stress_proc = None
    stop_workers_event = None
    workers = []
    if stress_cpu:
        if shutil.which("stress-ng"):
            print("[Stress] Found system stress-ng: launching 2 CPU workers in parallel...")
            cpu_stress_proc = subprocess.Popen(
                ["stress-ng", "--cpu", "2", "--timeout", f"{int(duration_sec + 5)}s"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            print("[Stress] stress-ng not found: spawning multi-process Python CPU load workers...")
            stop_workers_event = multiprocessing.Event()
            n_cores = max(1, multiprocessing.cpu_count() - 1)
            for _ in range(n_cores):
                p = multiprocessing.Process(target=_cpu_worker, args=(stop_workers_event,))
                p.start()
                workers.append(p)

    # 2. Initialize Realtime Pipeline
    pipeline = RealtimePipeline(sample_rate=sample_rate, hop_size=hop_size, fft_size=512)
    session_logger = pipeline.enable_session_logging(
        logs_root=str(out_dir),
        session_id="pi_stress_session",
        metadata={
            "duration_sec": duration_sec,
            "target_frames": target_frames,
            "stress_cpu": stress_cpu,
            "simulate_faults": simulate_faults
        }
    )
    pipeline.start()

    print(f"\n[Running Pipeline] Target: {target_frames} frames ({duration_sec}s)...\n")

    latencies_us = []
    fault_events = []
    start_time = time.perf_counter()
    next_print_time = start_time + 0.25

    try:
        for f in range(target_frames):
            t_frame_0 = time.perf_counter()

            # Generate synthetic dual mic audio
            t_audio = f * hop_duration_sec
            primary = 0.3 * np.sin(2 * np.pi * 440.0 * np.arange(hop_size) / sample_rate) + 0.15 * np.random.randn(hop_size)
            ref = 0.2 * np.random.randn(hop_size)

            ai_confidence = 0.90
            drift_ms = 0.2
            impulse_prob = 0.0
            vad_prob = 0.85
            ai_avail = True

            # Mid-run fault simulation
            if simulate_faults:
                progress = f / target_frames
                if 0.25 <= progress < 0.35:
                    # Error mic dropout
                    ref = np.zeros(0, dtype=np.float32)
                    drift_ms = 25.0
                    if not any(e["type"] == "error_mic_unplug" for e in fault_events):
                        fault_events.append({"type": "error_mic_unplug", "frame": f, "time_s": t_audio})
                elif 0.50 <= progress < 0.60:
                    # Forced ONNX Exception
                    ai_avail = False
                    if not any(e["type"] == "forced_onnx_exception" for e in fault_events):
                        fault_events.append({"type": "forced_onnx_exception", "frame": f, "time_s": t_audio})

            fused_hop = pipeline.process_hop(
                primary_samples=primary.astype(np.float32),
                reference_samples=ref.astype(np.float32),
                ai_enhanced_samples=primary.astype(np.float32) * 0.85,
                ai_confidence=ai_confidence,
                impulse_prob=impulse_prob,
                vad_prob=vad_prob,
                drift_ms=drift_ms,
                ai_available=ai_avail
            )

            t_frame_1 = time.perf_counter()
            latencies_us.append((t_frame_1 - t_frame_0) * 1e6)

            # Live CLI Telemetry Update (every 250ms)
            if t_frame_1 >= next_print_time or f == target_frames - 1:
                t = pipeline.get_telemetry()
                pct = (f + 1) / target_frames * 100.0
                elapsed = t_frame_1 - start_time
                mode_name = t.fusion_mode.name if hasattr(t.fusion_mode, "name") else str(t.fusion_mode)
                temp_str = f"{t.cpu_temperature_c:.1f}C" if t.temp_available else "N/A"
                print(
                    f"\r[{pct:5.1f}% | {elapsed:6.1f}s] "
                    f"Frames: {t.processed_frames:<6} | Drops: {t.dropped_frames:<2} | "
                    f"XRUNs: P={t.alsa_xruns_primary} R={t.alsa_xruns_reference} Out={t.alsa_xruns_playback} | "
                    f"RTF: {t.rtf:5.3f} | Lat: {t.total_processing_us/1000.0:4.2f}ms | "
                    f"CPU: {t.overall_cpu_pct:4.1f}% | Temp: {temp_str:<5} | Mode: {mode_name:<11}",
                    end="",
                    flush=True
                )
                next_print_time = t_frame_1 + 0.25

    finally:
        pipeline.stop()
        if cpu_stress_proc:
            cpu_stress_proc.terminate()
        if stop_workers_event:
            stop_workers_event.set()
        for p in workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()

    total_time = time.perf_counter() - start_time
    telemetry = pipeline.get_telemetry()

    mean_lat_us = float(np.mean(latencies_us))
    p95_lat_us = float(np.percentile(latencies_us, 95))
    p99_lat_us = float(np.percentile(latencies_us, 99))
    max_lat_us = float(np.max(latencies_us))
    rtf_overall = mean_lat_us / (hop_duration_sec * 1e6)

    print("\n\n================================================================================")
    print("                    STRESS TEST EXECUTION REPORT                                ")
    print("================================================================================")
    print(f"  Target Duration:       {duration_sec:.1f} s")
    print(f"  Actual Elapsed Time:   {total_time:.2f} s")
    print(f"  Total Processed:       {telemetry.processed_frames} frames")
    print(f"  Total Dropped Frames:  {telemetry.dropped_frames}")
    print(f"  ALSA Primary XRUNs:    {telemetry.alsa_xruns_primary}")
    print(f"  ALSA Reference XRUNs:  {telemetry.alsa_xruns_reference}")
    print(f"  ALSA Playback XRUNs:   {telemetry.alsa_xruns_playback}")
    print(f"  Mean Processing Time:  {mean_lat_us:.1f} us ({mean_lat_us/1000.0:.2f} ms)")
    print(f"  P95 Processing Time:   {p95_lat_us:.1f} us ({p95_lat_us/1000.0:.2f} ms)")
    print(f"  P99 Processing Time:   {p99_lat_us:.1f} us ({p99_lat_us/1000.0:.2f} ms)")
    print(f"  Max Processing Time:   {max_lat_us:.1f} us ({max_lat_us/1000.0:.2f} ms)")
    print(f"  Real-Time Factor:      {rtf_overall:.3f} (Budget: < 1.00, Target: < 0.50)")
    print(f"  Simulated Faults:      {len(fault_events)} events triggered")
    print(f"  Audit Directory:       {out_dir}")
    print("================================================================================\n")

    report = {
        "status": "PASSED" if telemetry.processed_frames == target_frames else "FAILED",
        "duration_sec": duration_sec,
        "elapsed_sec": total_time,
        "processed_frames": telemetry.processed_frames,
        "dropped_frames": telemetry.dropped_frames,
        "alsa_xruns_primary": telemetry.alsa_xruns_primary,
        "alsa_xruns_reference": telemetry.alsa_xruns_reference,
        "alsa_xruns_playback": telemetry.alsa_xruns_playback,
        "rtf": rtf_overall,
        "latency_mean_us": mean_lat_us,
        "latency_p95_us": p95_lat_us,
        "latency_p99_us": p99_lat_us,
        "fault_events": fault_events,
        "audit_dir": str(out_dir)
    }

    report_file = out_dir / "stress_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime stress test runner for Raspberry Pi 4/5.")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds (default: 10.0s; use 1800.0 for 30-min run)")
    parser.add_argument("--stress-cpu", action="store_true", help="Run parallel background CPU workers (stress-ng)")
    parser.add_argument("--simulate-faults", action="store_true", help="Inject mid-run error mic and ONNX faults")
    parser.add_argument("--log-dir", type=str, default="logs/stress", help="Output directory for logs")
    args = parser.parse_args()

    res = run_stress_test(
        duration_sec=args.duration,
        stress_cpu=args.stress_cpu,
        simulate_faults=args.simulate_faults,
        log_dir=args.log_dir
    )
    sys.exit(0 if res["status"] == "PASSED" else 1)
