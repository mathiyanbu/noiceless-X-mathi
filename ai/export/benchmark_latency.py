#!/usr/bin/env python3
import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import onnxruntime as ort

def get_soc_temperature() -> Optional[float]:
    """Query Raspberry Pi Broadcom SoC temperature via vcgencmd or sysfs."""
    try:
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and "temp=" in res.stdout:
            # Output format: temp=48.2'C
            temp_str = res.stdout.strip().replace("temp=", "").replace("'C", "")
            return float(temp_str)
    except Exception:
        pass

    # Fallback to sysfs thermal zone
    sysfs_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if sysfs_path.exists():
        try:
            return float(sysfs_path.read_text().strip()) / 1000.0
        except Exception:
            pass

    return None

def benchmark_single_model(
    model_path: str,
    num_warmup: int = 30,
    num_iterations: int = 200,
    num_bins: int = 257,
    threads: int = 2
) -> Dict[str, float]:
    """
    Measures streaming step latency on a single ONNX model frame-by-frame.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    # 1 frame input: (1, 2, 1, 257)
    dummy_frame = np.random.randn(1, 2, 1, num_bins).astype(np.float32)
    hidden = np.zeros((2, 1, 256), dtype=np.float32)

    # Warmup runs
    for _ in range(num_warmup):
        outs = session.run(None, {"noisy_stft": dummy_frame, "hidden_in": hidden})
        hidden = outs[2]

    latencies_ms: List[float] = []

    for _ in range(num_iterations):
        t0 = time.perf_counter_ns()
        outs = session.run(None, {"noisy_stft": dummy_frame, "hidden_in": hidden})
        t1 = time.perf_counter_ns()
        hidden = outs[2]
        latencies_ms.append((t1 - t0) / 1e6)

    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
        "rtf": float(np.mean(arr) / 5.0), # Relative to 5.0ms hop size
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark streaming inference latency on Raspberry Pi.")
    parser.add_argument("--fp32", type=str, default="models/onnx/speech_enhancer_fp32.onnx", help="Path to FP32 model")
    parser.add_argument("--int8", type=str, default="models/onnx/speech_enhancer_int8.onnx", help="Path to INT8 model")
    parser.add_argument("--iterations", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--threads", type=int, default=2, help="ORT intra-op threads")
    args = parser.parse_args()

    print("==========================================================================")
    print("          SIH26052 NOICELESSX — Raspberry Pi Inference Benchmark          ")
    print("==========================================================================")
    print(f"  Platform:    {platform.machine()} ({platform.system()} {platform.release()})")
    print(f"  Processor:   {platform.processor() or 'ARM64'}")
    print(f"  ORT Threads: {args.threads}")

    temp_start = get_soc_temperature()
    if temp_start is not None:
        print(f"  SoC Temp:    {temp_start:.1f} °C")
    print("--------------------------------------------------------------------------")

    results = {}
    if Path(args.fp32).exists():
        print(f"Benchmarking FP32 model ({args.fp32})...")
        results["FP32"] = benchmark_single_model(args.fp32, num_iterations=args.iterations, threads=args.threads)

    if Path(args.int8).exists():
        print(f"Benchmarking INT8 model ({args.int8})...")
        results["INT8"] = benchmark_single_model(args.int8, num_iterations=args.iterations, threads=args.threads)

    temp_end = get_soc_temperature()

    print("\n========================= Measured Latency Summary =========================")
    print("  Model | Mean Latency |   p50   |   p95   |   Max   | RTF (5ms Hop) | Realtime?")
    print("  ------+--------------+---------+---------+---------+---------------+----------")
    for name, r in results.items():
        is_rt = "YES (PASS)" if r["mean_ms"] <= 5.0 else "NO (FAIL)"
        print(f"  {name:5} | {r['mean_ms']:7.2f} ms   | {r['p50_ms']:5.2f} ms | {r['p95_ms']:5.2f} ms | {r['max_ms']:5.2f} ms | {r['rtf']:11.2f}x  | {is_rt}")
    print("============================================================================")

    if temp_end is not None:
        print(f"  Final SoC Temp: {temp_end:.1f} °C (Delta: {temp_end - (temp_start or temp_end):+.1f} °C)\n")


if __name__ == "__main__":
    main()
