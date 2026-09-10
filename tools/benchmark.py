#!/usr/bin/env python3
"""
SIH26052 — NOICELESSX: Streaming ONNX Inference Latency & RTF Benchmarking Tool.
Measures real-time streaming inference latency (per 5.0 ms hop) of ComplexCRN FP32 and INT8 models,
computes p50/p95/p99 percentiles and Real-Time Factor (RTF), and logs performance telemetry.
Designed for execution on both workstation and Raspberry Pi 4/5 ARM NEON targets.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_system_hardware_info() -> Dict[str, Any]:
    """Collects honest host system hardware and OS telemetry."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
    }
    # Check if running on Raspberry Pi
    is_rpi = False
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r") as f:
                model_str = f.read().strip()
                info["rpi_model"] = model_str
                is_rpi = True
    except Exception:
        pass
    info["is_raspberry_pi"] = is_rpi
    return info


def benchmark_single_model(
    model_path: str,
    num_frames: int = 500,
    warmup_frames: int = 50,
    num_threads: int = 2,
    num_bins: int = 257,
    hop_time_ms: float = 5.0,  # 80 samples at 16kHz = 5.0 ms
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmarks single-frame streaming inference:
    Input:  noisy_stft (1, 2, 1, 257), hidden_in (2, 1, 256)
    Output: enhanced_stft (1, 2, 1, 257), mask (1, 2, 1, 257), hidden_out (2, 1, 256)
    Persistent recurrent state passed from frame to frame.
    """
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found: {m_path}")

    # Configure session
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(m_path), sess_opts, providers=providers)
    active_provider = session.get_providers()[0]

    # Preallocate deterministic streaming tensors
    rng = np.random.RandomState(42)
    frame_in = rng.randn(1, 2, 1, num_bins).astype(np.float32)
    hidden_state = np.zeros((2, 1, 256), dtype=np.float32)

    # 1. Warm-up Phase
    for _ in range(warmup_frames):
        outputs = session.run(None, {"noisy_stft": frame_in, "hidden_in": hidden_state})
        hidden_state = outputs[2]

    # 2. Measured Benchmark Phase
    latencies_ms: List[float] = []
    t_start_total = time.perf_counter()

    for _ in range(num_frames):
        t0 = time.perf_counter()
        outputs = session.run(None, {"noisy_stft": frame_in, "hidden_in": hidden_state})
        t1 = time.perf_counter()
        hidden_state = outputs[2]
        latencies_ms.append((t1 - t0) * 1000.0)

    total_bench_time_sec = time.perf_counter() - t_start_total

    # 3. Statistical Analysis
    lat_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_arr))
    median_lat = float(np.median(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    std_lat = float(np.std(lat_arr))

    # Real-Time Factor: latency / audio_duration (5.0 ms)
    rtf = mean_lat / hop_time_ms
    rtf_p95 = p95_lat / hop_time_ms
    throughput_fps = num_frames / total_bench_time_sec

    file_size_mb = os.path.getsize(m_path) / (1024 * 1024)

    results = {
        "model_name": m_path.name,
        "model_path": str(m_path),
        "file_size_mb": file_size_mb,
        "provider": active_provider,
        "threads": num_threads,
        "measured_frames": num_frames,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "std_latency_ms": std_lat,
        "rtf": rtf,
        "rtf_p95": rtf_p95,
        "throughput_fps": throughput_fps,
        "audio_hop_budget_ms": hop_time_ms,
        "realtime_capable": rtf < 1.0,
    }

    if verbose:
        status_str = "REAL-TIME" if rtf < 1.0 else "EXCEEDS BUDGET"
        print(f"  Model:            {m_path.name} ({file_size_mb:.2f} MB)")
        print(f"  Provider/Threads: {active_provider} | {num_threads} threads")
        print(f"  Mean Latency:     {mean_lat:.3f} ms (Hop Budget: {hop_time_ms:.1f} ms | RTF: {rtf:.3f}x)")
        print(f"  Median (p50):     {median_lat:.3f} ms")
        print(f"  p95 Latency:      {p95_lat:.3f} ms (p95 RTF: {rtf_p95:.3f}x)")
        print(f"  p99 Latency:      {p99_lat:.3f} ms")
        print(f"  Min / Max:        {min_lat:.3f} ms / {max_lat:.3f} ms")
        print(f"  Throughput:       {throughput_fps:.1f} frames/sec")
        print(f"  Status:           {status_str} (Headroom: {max(0.0, (1.0 - rtf) * 100):.1f}%)\n")

    return results


def run_benchmark_comparison(
    fp32_model: str = "models/onnx/speech_enhancer_fp32.onnx",
    int8_model: str = "models/onnx/speech_enhancer_int8.onnx",
    num_frames: int = 500,
    warmup_frames: int = 50,
    num_threads: int = 2,
    output_json: Optional[str] = "models/benchmark_results.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs latency benchmarks on both FP32 and INT8 models and prints comparative summary.
    """
    hw_info = get_system_hardware_info()

    if verbose:
        print("\n==========================================================================")
        print("     SIH26052 NOICELESSX — Streaming ONNX Latency & RTF Benchmark         ")
        print("==========================================================================")
        print(f"Hardware Platform:    {hw_info['platform']} ({hw_info['machine']})")
        print(f"Processor:            {hw_info['processor']}")
        print(f"Inference Mode:       Single-Frame Streaming (5.0 ms hop @ 16kHz)")
        print(f"Frames Measured:      {num_frames} frames ({warmup_frames} warmup)")
        print(f"Thread Allocation:    {num_threads} intra-op threads")
        print("--------------------------------------------------------------------------\n")

    res_fp32 = benchmark_single_model(
        fp32_model,
        num_frames=num_frames,
        warmup_frames=warmup_frames,
        num_threads=num_threads,
        verbose=verbose,
    )

    res_int8 = benchmark_single_model(
        int8_model,
        num_frames=num_frames,
        warmup_frames=warmup_frames,
        num_threads=num_threads,
        verbose=verbose,
    )

    speedup = res_fp32["mean_latency_ms"] / max(1e-6, res_int8["mean_latency_ms"])
    mem_reduction = (1.0 - res_int8["file_size_mb"] / res_fp32["file_size_mb"]) * 100.0

    if verbose:
        print("==========================================================================")
        print("                    Comparative Benchmark Summary                         ")
        print("==========================================================================")
        print(f"  Metric              | FP32 Baseline    | INT8 Quantized   | INT8 Advantage")
        print("  --------------------+------------------+------------------+-----------------")
        print(f"  File Size           | {res_fp32['file_size_mb']:13.2f} MB| {res_int8['file_size_mb']:13.2f} MB| {res_fp32['file_size_mb']/res_int8['file_size_mb']:.2f}x ({mem_reduction:.1f}% less)")
        print(f"  Mean Latency        | {res_fp32['mean_latency_ms']:13.3f} ms| {res_int8['mean_latency_ms']:13.3f} ms| {speedup:.2f}x speedup")
        print(f"  Median (p50)        | {res_fp32['median_latency_ms']:13.3f} ms| {res_int8['median_latency_ms']:13.3f} ms| {res_fp32['median_latency_ms']/max(1e-6, res_int8['median_latency_ms']):.2f}x speedup")
        print(f"  p95 Latency         | {res_fp32['p95_latency_ms']:13.3f} ms| {res_int8['p95_latency_ms']:13.3f} ms| {res_fp32['p95_latency_ms']/max(1e-6, res_int8['p95_latency_ms']):.2f}x speedup")
        print(f"  Real-Time Factor    | {res_fp32['rtf']:13.3f} x | {res_int8['rtf']:13.3f} x | {'Faster' if res_int8['rtf'] < res_fp32['rtf'] else 'Similar'}")
        print(f"  CPU Headroom (5ms)  | {max(0.0, (1.0 - res_fp32['rtf']) * 100):13.1f} % | {max(0.0, (1.0 - res_int8['rtf']) * 100):13.1f} % | {max(0.0, (1.0 - res_int8['rtf'])*100) - max(0.0, (1.0 - res_fp32['rtf'])*100):+.1f}% margin")
        print("==========================================================================\n")

    summary = {
        "hardware_info": hw_info,
        "fp32_benchmark": res_fp32,
        "int8_benchmark": res_int8,
        "speedup_ratio": speedup,
        "memory_reduction_percent": mem_reduction,
    }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"Saved benchmark results to: {out_p}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark single-frame streaming ONNX models.")
    parser.add_argument("--fp32-model", type=str, default="models/onnx/speech_enhancer_fp32.onnx", help="Path to FP32 model")
    parser.add_argument("--int8-model", type=str, default="models/onnx/speech_enhancer_int8.onnx", help="Path to INT8 model")
    parser.add_argument("--frames", type=int, default=500, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--threads", type=int, default=2, help="Intra-op thread count (default: 2 for Pi 4/5)")
    parser.add_argument("--output", type=str, default="models/benchmark_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    run_benchmark_comparison(
        fp32_model=args.fp32_model,
        int8_model=args.int8_model,
        num_frames=args.frames,
        warmup_frames=args.warmup,
        num_threads=args.threads,
        output_json=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
