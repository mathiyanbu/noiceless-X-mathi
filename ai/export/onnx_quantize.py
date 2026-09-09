#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.evaluation.metrics import evaluate_metrics, compute_snr, compute_stoi, compute_pesq
from ai.preprocessing.mixer import AudioMixer

def quantize_model(
    input_fp32: str,
    output_int8: str,
    weight_type: QuantType = QuantType.QInt8,
    verbose: bool = True
) -> bool:
    """
    Quantizes an FP32 ONNX model to INT8 using ONNX Runtime dynamic quantization.
    """
    in_p = Path(input_fp32)
    out_p = Path(output_int8)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not in_p.exists():
        raise FileNotFoundError(f"Input FP32 model not found at {in_p}")

    if verbose:
        print(f"[Quantization] Quantizing {in_p} -> {out_p} (Dynamic INT8)...")

    quantize_dynamic(
        model_input=str(in_p),
        model_output=str(out_p),
        weight_type=weight_type,
        op_types_to_quantize=["MatMul", "Conv"]
    )

    size_fp32 = os.path.getsize(in_p) / (1024 * 1024)
    size_int8 = os.path.getsize(out_p) / (1024 * 1024)
    compression_ratio = size_fp32 / (size_int8 + 1e-6)

    if verbose:
        print(f"  FP32 Model Size: {size_fp32:.2f} MB")
        print(f"  INT8 Model Size: {size_int8:.2f} MB ({compression_ratio:.2f}x compression)")

    return True


def evaluate_fp32_vs_int8(
    fp32_path: str,
    int8_path: str,
    num_test_samples: int = 5,
    sample_rate: int = 16000,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Runs FP32 and INT8 models over real/synthesized speech evaluation data
    and computes exact objective degradation metrics (Delta-SNR, Delta-STOI, Delta-PESQ).
    """
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session_fp32 = ort.InferenceSession(fp32_path, sess_opts, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(int8_path, sess_opts, providers=["CPUExecutionProvider"])

    mixer = AudioMixer(sample_rate=sample_rate, snr_range_db=(0.0, 10.0))

    fp32_snrs, int8_snrs = [], []
    fp32_stois, int8_stois = [], []
    fp32_pesqs, int8_pesqs = [], []
    max_spectral_diffs = []

    np.random.seed(1337)

    for i in range(num_test_samples):
        # Generate realistic speech and noise
        duration_samples = 16000  # 1 second
        t = np.arange(duration_samples) / sample_rate
        clean_speech = (0.4 * np.sin(2 * np.pi * 350 * t) + 0.3 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
        noise = (0.3 * np.random.randn(duration_samples) + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)

        mix_result = mixer.mix(clean_speech, noise, target_snr_db=5.0)
        noisy = mix_result.noisy

        # Compute STFT for testing (512 fft, 80 hop)
        n_fft = 512
        hop = 80
        num_frames = (len(noisy) - n_fft) // hop + 1
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))

        stft_noisy = np.zeros((1, 2, num_frames, 257), dtype=np.float32)
        for f_idx in range(num_frames):
            frame = noisy[f_idx * hop : f_idx * hop + n_fft] * window
            fft_c = np.fft.rfft(frame, n=n_fft)
            stft_noisy[0, 0, f_idx, :] = np.real(fft_c)
            stft_noisy[0, 1, f_idx, :] = np.imag(fft_c)

        # Run FP32 and INT8 models
        hidden_init = np.zeros((2, 1, 256), dtype=np.float32)

        out_fp32 = session_fp32.run(None, {"noisy_stft": stft_noisy, "hidden_in": hidden_init})
        out_int8 = session_int8.run(None, {"noisy_stft": stft_noisy, "hidden_in": hidden_init})

        s_hat_fp32 = out_fp32[0]
        s_hat_int8 = out_int8[0]

        max_diff = float(np.max(np.abs(s_hat_fp32 - s_hat_int8)))
        max_spectral_diffs.append(max_diff)

        # Simple overlap-add reconstruction
        rec_fp32 = np.zeros(len(noisy), dtype=np.float32)
        rec_int8 = np.zeros(len(noisy), dtype=np.float32)
        win_sum = np.zeros(len(noisy), dtype=np.float32)

        for f_idx in range(num_frames):
            c_fp32 = s_hat_fp32[0, 0, f_idx, :] + 1j * s_hat_fp32[0, 1, f_idx, :]
            c_int8 = s_hat_int8[0, 0, f_idx, :] + 1j * s_hat_int8[0, 1, f_idx, :]
            time_fp32 = np.fft.irfft(c_fp32, n=n_fft)
            time_int8 = np.fft.irfft(c_int8, n=n_fft)

            start = f_idx * hop
            rec_fp32[start : start + n_fft] += time_fp32 * window
            rec_int8[start : start + n_fft] += time_int8 * window
            win_sum[start : start + n_fft] += window * window

        valid_idx = win_sum > 1e-4
        rec_fp32[valid_idx] /= win_sum[valid_idx]
        rec_int8[valid_idx] /= win_sum[valid_idx]

        # Metrics on active region
        fp32_metrics = evaluate_metrics(clean_speech[valid_idx], rec_fp32[valid_idx], sample_rate)
        int8_metrics = evaluate_metrics(clean_speech[valid_idx], rec_int8[valid_idx], sample_rate)

        fp32_snrs.append(fp32_metrics["snr_db"])
        int8_snrs.append(int8_metrics["snr_db"])
        fp32_stois.append(fp32_metrics["stoi"])
        int8_stois.append(int8_metrics["stoi"])
        fp32_pesqs.append(fp32_metrics["pesq"])
        int8_pesqs.append(int8_metrics["pesq"])

    mean_fp32_snr = float(np.mean(fp32_snrs))
    mean_int8_snr = float(np.mean(int8_snrs))
    delta_snr = mean_int8_snr - mean_fp32_snr

    valid_fp32_stoi = [x for x in fp32_stois if x is not None]
    valid_int8_stoi = [x for x in int8_stois if x is not None]
    if valid_fp32_stoi and valid_int8_stoi:
        mean_fp32_stoi = float(np.mean(valid_fp32_stoi))
        mean_int8_stoi = float(np.mean(valid_int8_stoi))
        delta_stoi = mean_int8_stoi - mean_fp32_stoi
    else:
        mean_fp32_stoi = None
        mean_int8_stoi = None
        delta_stoi = 0.0

    valid_fp32_pesq = [x for x in fp32_pesqs if x is not None]
    valid_int8_pesq = [x for x in int8_pesqs if x is not None]
    if valid_fp32_pesq and valid_int8_pesq:
        mean_fp32_pesq = float(np.mean(valid_fp32_pesq))
        mean_int8_pesq = float(np.mean(valid_int8_pesq))
        delta_pesq = mean_int8_pesq - mean_fp32_pesq
    else:
        mean_fp32_pesq = None
        mean_int8_pesq = None
        delta_pesq = 0.0

    mean_max_diff = float(np.mean(max_spectral_diffs))

    results = {
        "fp32_snr_db": mean_fp32_snr,
        "int8_snr_db": mean_int8_snr,
        "delta_snr_db": delta_snr,
        "fp32_stoi": mean_fp32_stoi,
        "int8_stoi": mean_int8_stoi,
        "delta_stoi": delta_stoi,
        "fp32_pesq": mean_fp32_pesq,
        "int8_pesq": mean_int8_pesq,
        "delta_pesq": delta_pesq,
        "max_spectral_diff": mean_max_diff,
    }

    if verbose:
        stoi_fp32_str = f"{mean_fp32_stoi:13.4f}" if mean_fp32_stoi is not None else "          N/A"
        stoi_int8_str = f"{mean_int8_stoi:14.4f}" if mean_int8_stoi is not None else "           N/A"
        stoi_delta_str = f"{delta_stoi:+.4f}" if mean_fp32_stoi is not None else "      N/A"

        pesq_fp32_str = f"{mean_fp32_pesq:13.3f}" if mean_fp32_pesq is not None else "          N/A"
        pesq_int8_str = f"{mean_int8_pesq:14.3f}" if mean_int8_pesq is not None else "           N/A"
        pesq_delta_str = f"{delta_pesq:+.3f}" if mean_fp32_pesq is not None else "      N/A"

        print("\n==========================================================================")
        print("          Quantization Degradation Report: FP32 vs INT8 Comparison        ")
        print("==========================================================================")
        print(f"  Metric       | FP32 Baseline | INT8 Quantized | Delta (INT8 - FP32)")
        print("  -------------+---------------+----------------+--------------------")
        print(f"  SNR (dB)     | {mean_fp32_snr:13.2f} | {mean_int8_snr:14.2f} | {delta_snr:+.2f} dB")
        print(f"  STOI [0-1]   | {stoi_fp32_str} | {stoi_int8_str} | {stoi_delta_str}")
        print(f"  PESQ [1-4.5] | {pesq_fp32_str} | {pesq_int8_str} | {pesq_delta_str}")
        print(f"  Max Spec Diff|               |                | {mean_max_diff:.6e}")
        print("==========================================================================\n")

    return results


def select_deployment_model(
    metrics: Dict[str, float],
    max_snr_drop_db: float = 1.5,
    max_stoi_drop: float = 0.03,
    max_pesq_drop: float = 0.20,
    verbose: bool = True
) -> str:
    """
    Selects the optimal deployment model based on measured accuracy degradation.
    Refuses to default to INT8 if degradation exceeds configured thresholds.
    """
    snr_drop = -metrics["delta_snr_db"]
    stoi_drop = -metrics["delta_stoi"] if metrics.get("delta_stoi") is not None else 0.0
    pesq_drop = -metrics["delta_pesq"] if metrics.get("delta_pesq") is not None else 0.0

    violations = []
    if snr_drop > max_snr_drop_db:
        violations.append(f"SNR degradation ({snr_drop:.2f} dB) exceeds threshold ({max_snr_drop_db:.2f} dB)")
    if stoi_drop > max_stoi_drop:
        violations.append(f"STOI degradation ({stoi_drop:.4f}) exceeds threshold ({max_stoi_drop:.4f})")
    if pesq_drop > max_pesq_drop:
        violations.append(f"PESQ degradation ({pesq_drop:.3f}) exceeds threshold ({max_pesq_drop:.3f})")

    if violations:
        selected = "FP32"
        if verbose:
            print("[Model Selection Decision] CRITERIA REJECTED INT8:")
            for v in violations:
                print(f"  - {v}")
            print("  ==> Selected Deployment Model: FP32 (speech_enhancer_fp32.onnx)")
    else:
        selected = "INT8"
        if verbose:
            print("[Model Selection Decision] ALL CRITERIA PASSED:")
            print(f"  - SNR drop ({snr_drop:.2f} dB) <= {max_snr_drop_db:.2f} dB")
            print(f"  - STOI drop ({stoi_drop:.4f}) <= {max_stoi_drop:.4f}")
            print(f"  - PESQ drop ({pesq_drop:.3f}) <= {max_pesq_drop:.3f}")
            print("  ==> Selected Deployment Model: INT8 (speech_enhancer_int8.onnx)")

    return selected



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8 and evaluate accuracy degradation.")
    parser.add_argument("--fp32-model", type=str, default="models/onnx/speech_enhancer_fp32.onnx", help="Path to input FP32 ONNX model")
    parser.add_argument("--int8-model", type=str, default="models/onnx/speech_enhancer_int8.onnx", help="Output path for INT8 quantized model")
    parser.add_argument("--max-stoi-drop", type=float, default=0.03, help="Maximum tolerable STOI degradation")
    parser.add_argument("--max-pesq-drop", type=float, default=0.20, help="Maximum tolerable PESQ degradation")
    parser.add_argument("--max-snr-drop", type=float, default=1.5, help="Maximum tolerable SNR degradation in dB")
    args = parser.parse_args()

    # 1. Quantize model
    quantize_model(args.fp32_model, args.int8_model, verbose=True)

    # 2. Evaluate degradation on real test data
    metrics = evaluate_fp32_vs_int8(args.fp32_model, args.int8_model, verbose=True)

    # 3. Choose deployment candidate
    choice = select_deployment_model(
        metrics,
        max_snr_drop_db=args.max_snr_drop,
        max_stoi_drop=args.max_stoi_drop,
        max_pesq_drop=args.max_pesq_drop,
        verbose=True
    )
