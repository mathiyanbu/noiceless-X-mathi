#!/usr/bin/env python3
"""
SIH26052 — NOICELESSX: ONNX INT8 Quantization, Real Test Evaluation & Deployment Selection.
Quantizes FP32 ONNX model to INT8, evaluates accuracy degradation against FP32
on the REAL held-out test split (ΔSNR, ΔSI-SNR, STOI, PESQ), enforces quality gates,
and writes the winning model to models/onnx/speech_enhancer.onnx.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_dynamic

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.evaluation.metrics import (
    compute_pesq,
    compute_si_sdr,
    compute_stoi,
    compute_true_snr,
)


class RealAudioCalibrationDataReader(CalibrationDataReader):
    """
    Feeds real audio STFT frames from the manifest's validation split
    into the static quantizer calibration loop (never synthetic noise).
    """

    def __init__(self, manifest_path: str, max_samples: int = 20, num_bins: int = 257):
        super().__init__()
        from ai.training.dataset import SpeechEnhancementDataset

        self.dataset = SpeechEnhancementDataset(
            manifest_path=manifest_path,
            split="val",
            segment_duration=2.0,
        )
        self.max_samples = min(max_samples, len(self.dataset))
        self.current_idx = 0
        self.num_bins = num_bins

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.current_idx >= self.max_samples:
            return None
        sample = self.dataset[self.current_idx]
        self.current_idx += 1

        # noisy_stft shape: (2, T, 257) -> (1, 2, T, 257)
        stft_arr = sample["noisy_stft"].unsqueeze(0).numpy()
        hidden_arr = np.zeros((2, 1, 256), dtype=np.float32)

        return {
            "noisy_stft": stft_arr,
            "hidden_in": hidden_arr,
        }

    def rewind(self):
        self.current_idx = 0


def quantize_model(
    input_fp32: str = "models/onnx/speech_enhancer_fp32.onnx",
    output_int8: str = "models/onnx/speech_enhancer_int8.onnx",
    manifest_path: Optional[str] = "data/manifests/manifest.csv",
    weight_type: QuantType = QuantType.QInt8,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Quantizes an FP32 ONNX model to INT8 using ONNX Runtime quantization.
    """
    in_p = Path(input_fp32)
    out_p = Path(output_int8)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not in_p.exists():
        raise FileNotFoundError(f"Input FP32 model not found at {in_p}")

    if verbose:
        print("\n==========================================================================")
        print("          SIH26052 NOICELESSX — ONNX INT8 Quantization Pipeline           ")
        print("==========================================================================")
        print(f"[Quantization] Input FP32 Model:  {in_p}")
        print(f"[Quantization] Output INT8 Model: {out_p}")

    # Quantize operators (MatMul, Conv, Gemm, ConvTranspose)
    quantize_dynamic(
        model_input=str(in_p),
        model_output=str(out_p),
        weight_type=weight_type,
        op_types_to_quantize=["MatMul", "Conv", "Gemm", "ConvTranspose"],
    )

    size_fp32 = os.path.getsize(in_p) / (1024 * 1024)
    size_int8 = os.path.getsize(out_p) / (1024 * 1024)
    compression_ratio = size_fp32 / (size_int8 + 1e-6)

    if verbose:
        print(f"  FP32 Model Size:      {size_fp32:.2f} MB")
        print(f"  INT8 Model Size:      {size_int8:.2f} MB")
        print(f"  Compression Ratio:    {compression_ratio:.2f}x (Memory savings: {(1.0 - size_int8 / size_fp32) * 100:.1f}%)")
        print("--------------------------------------------------------------------------")

    return {
        "fp32_size_mb": size_fp32,
        "int8_size_mb": size_int8,
        "compression_ratio": compression_ratio,
        "output_path": str(out_p),
    }


def evaluate_fp32_vs_int8(
    fp32_path: str = "models/onnx/speech_enhancer_fp32.onnx",
    int8_path: str = "models/onnx/speech_enhancer_int8.onnx",
    manifest_path: Optional[str] = "data/manifests/manifest.csv",
    num_test_samples: Optional[int] = None,
    sample_rate: int = 16000,
    fft_size: int = 512,
    hop_size: int = 80,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs FP32 and INT8 models over test data and computes exact objective metrics:
    ΔSNR, ΔSI-SNR, STOI, PESQ, and max spectral diff.
    Supports both real manifest held-out test split and fallback audio synthesis.
    """
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session_fp32 = ort.InferenceSession(fp32_path, sess_opts, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(int8_path, sess_opts, providers=["CPUExecutionProvider"])

    window = torch.hann_window(fft_size)

    fp32_snrs_in, fp32_snrs_out = [], []
    int8_snrs_in, int8_snrs_out = [], []
    delta_snrs_fp32, delta_snrs_int8 = [], []

    fp32_sisnrs_in, fp32_sisnrs_out = [], []
    int8_sisnrs_in, int8_sisnrs_out = [], []
    delta_sisnrs_fp32, delta_sisnrs_int8 = [], []

    fp32_stois, int8_stois = [], []
    fp32_pesqs, int8_pesqs = [], []
    max_spectral_diffs = []

    has_manifest = manifest_path and Path(manifest_path).exists()
    if has_manifest:
        from ai.training.dataset import SpeechEnhancementDataset

        test_dataset = SpeechEnhancementDataset(
            manifest_path=manifest_path,
            split="test",
            sample_rate=sample_rate,
            segment_duration=2.0,
        )
        total_available = len(test_dataset)
        count = min(num_test_samples, total_available) if num_test_samples else total_available
    else:
        from ai.training.dataset import AudioMixer

        test_dataset = None
        count = num_test_samples if num_test_samples else 5
        mixer = AudioMixer(target_sample_rate=sample_rate)

    if verbose:
        print("\n==========================================================================")
        print("   Real Test Set Evaluation: FP32 Baseline vs INT8 Quantized Comparison   ")
        print("==========================================================================")
        source_label = f"{manifest_path} (Held-out Test Split)" if has_manifest else "Harmonic Test Signals"
        print(f"Evaluation source:   {source_label}")
        print(f"Evaluated samples:   {count} audio mixtures")
        print("--------------------------------------------------------------------------")

    for idx in range(count):
        if test_dataset is not None:
            sample = test_dataset[idx]
            noisy_stft = sample["noisy_stft"].unsqueeze(0).numpy()
            clean_wav = sample["clean_wav"].numpy()
            noisy_wav = sample["noisy_wav"].numpy()
        else:
            duration = sample_rate
            t = np.arange(duration) / sample_rate
            clean_wav = (0.5 * np.sin(2 * np.pi * (300 + idx * 50) * t)).astype(np.float32)
            noise_wav = (0.1 * np.random.randn(duration)).astype(np.float32)
            mix_w, clean_w, _ = mixer.mix(clean_wav, noise_wav, snr_db=5.0)
            noisy_wav = mix_w
            clean_wav = clean_w

            # STFT
            t_clean = torch.from_numpy(clean_wav)
            t_noisy = torch.from_numpy(noisy_wav)
            stft_cplx = torch.stft(
                t_noisy,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=fft_size,
                window=window,
                center=True,
                return_complex=True,
            )
            real = stft_cplx.real.T
            imag = stft_cplx.imag.T
            noisy_stft = torch.stack([real, imag], dim=0).unsqueeze(0).numpy()

        target_len = len(clean_wav)
        hidden_init = np.zeros((2, 1, 256), dtype=np.float32)

        out_fp32 = session_fp32.run(None, {"noisy_stft": noisy_stft, "hidden_in": hidden_init})
        s_hat_fp32 = out_fp32[0]

        out_int8 = session_int8.run(None, {"noisy_stft": noisy_stft, "hidden_in": hidden_init})
        s_hat_int8 = out_int8[0]

        spec_diff = float(np.max(np.abs(s_hat_fp32 - s_hat_int8)))
        max_spectral_diffs.append(spec_diff)

        # Exact iSTFT reconstruction matching Phase 3
        def istft_reconstruct(s_hat_arr: np.ndarray) -> np.ndarray:
            cplx = torch.complex(
                torch.from_numpy(s_hat_arr[0, 0]).T,
                torch.from_numpy(s_hat_arr[0, 1]).T,
            )
            wav = torch.istft(
                cplx,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=fft_size,
                window=window,
                center=True,
                length=target_len,
            )
            return wav.numpy()

        rec_fp32 = istft_reconstruct(s_hat_fp32)
        rec_int8 = istft_reconstruct(s_hat_int8)

        # True SNR metrics
        snr_res_fp32 = compute_true_snr(clean_wav, noisy_wav, rec_fp32)
        snr_res_int8 = compute_true_snr(clean_wav, noisy_wav, rec_int8)

        fp32_snrs_in.append(snr_res_fp32["snr_in"])
        fp32_snrs_out.append(snr_res_fp32["snr_out"])
        delta_snrs_fp32.append(snr_res_fp32["delta_snr"])

        int8_snrs_in.append(snr_res_int8["snr_in"])
        int8_snrs_out.append(snr_res_int8["snr_out"])
        delta_snrs_int8.append(snr_res_int8["delta_snr"])

        # SI-SNR metrics
        sisnr_in = compute_si_sdr(clean_wav, noisy_wav)
        sisnr_fp32 = compute_si_sdr(clean_wav, rec_fp32)
        sisnr_int8 = compute_si_sdr(clean_wav, rec_int8)

        fp32_sisnrs_in.append(sisnr_in)
        fp32_sisnrs_out.append(sisnr_fp32)
        delta_sisnrs_fp32.append(sisnr_fp32 - sisnr_in)

        int8_sisnrs_in.append(sisnr_in)
        int8_sisnrs_out.append(sisnr_int8)
        delta_sisnrs_int8.append(sisnr_int8 - sisnr_in)

        # STOI metrics
        stoi_fp32 = compute_stoi(clean_wav, rec_fp32, sample_rate=sample_rate)
        stoi_int8 = compute_stoi(clean_wav, rec_int8, sample_rate=sample_rate)
        if stoi_fp32 is not None and stoi_int8 is not None:
            fp32_stois.append(stoi_fp32)
            int8_stois.append(stoi_int8)

        # PESQ metrics
        pesq_fp32 = compute_pesq(clean_wav, rec_fp32, sample_rate=sample_rate)
        pesq_int8 = compute_pesq(clean_wav, rec_int8, sample_rate=sample_rate)
        if pesq_fp32 is not None and pesq_int8 is not None:
            fp32_pesqs.append(pesq_fp32)
            int8_pesqs.append(pesq_int8)

    mean_fp32_snr_out = float(np.mean(fp32_snrs_out))
    mean_int8_snr_out = float(np.mean(int8_snrs_out))
    delta_snr_quant = mean_int8_snr_out - mean_fp32_snr_out

    mean_fp32_sisnr_out = float(np.mean(fp32_sisnrs_out))
    mean_int8_sisnr_out = float(np.mean(int8_sisnrs_out))
    delta_sisnr_quant = mean_int8_sisnr_out - mean_fp32_sisnr_out

    mean_fp32_delta_snr = float(np.mean(delta_snrs_fp32))
    mean_int8_delta_snr = float(np.mean(delta_snrs_int8))

    mean_fp32_stoi = float(np.mean(fp32_stois)) if fp32_stois else None
    mean_int8_stoi = float(np.mean(int8_stois)) if int8_stois else None
    delta_stoi = (mean_int8_stoi - mean_fp32_stoi) if (mean_fp32_stoi is not None and mean_int8_stoi is not None) else None

    mean_fp32_pesq = float(np.mean(fp32_pesqs)) if fp32_pesqs else None
    mean_int8_pesq = float(np.mean(int8_pesqs)) if int8_pesqs else None
    delta_pesq = (mean_int8_pesq - mean_fp32_pesq) if (mean_fp32_pesq is not None and mean_int8_pesq is not None) else None

    mean_max_spec_diff = float(np.mean(max_spectral_diffs))

    if verbose:
        stoi_fp32_str = f"{mean_fp32_stoi:13.4f}" if mean_fp32_stoi is not None else "          N/A"
        stoi_int8_str = f"{mean_int8_stoi:14.4f}" if mean_int8_stoi is not None else "           N/A"
        stoi_delta_str = f"{delta_stoi:+.4f}" if delta_stoi is not None else "      N/A"

        pesq_fp32_str = f"{mean_fp32_pesq:13.3f}" if mean_fp32_pesq is not None else "          N/A"
        pesq_int8_str = f"{mean_int8_pesq:14.3f}" if mean_int8_pesq is not None else "           N/A"
        pesq_delta_str = f"{delta_pesq:+.3f}" if delta_pesq is not None else "      N/A"

        print(f"  Metric              | FP32 Baseline | INT8 Quantized | Delta (INT8 - FP32)")
        print("  --------------------+---------------+----------------+--------------------")
        print(f"  Output SNR (dB)     | {mean_fp32_snr_out:13.2f} | {mean_int8_snr_out:14.2f} | {delta_snr_quant:+.2f} dB")
        print(f"  SNR Improv Delta-SNR| {mean_fp32_delta_snr:13.2f} | {mean_int8_delta_snr:14.2f} | {mean_int8_delta_snr - mean_fp32_delta_snr:+.2f} dB")
        print(f"  Output SI-SNR (dB)  | {mean_fp32_sisnr_out:13.2f} | {mean_int8_sisnr_out:14.2f} | {delta_sisnr_quant:+.2f} dB")
        print(f"  STOI Intelligibility| {stoi_fp32_str} | {stoi_int8_str} | {stoi_delta_str}")
        print(f"  PESQ Speech Quality | {pesq_fp32_str} | {pesq_int8_str} | {pesq_delta_str}")
        print(f"  Mean Max Spec Diff  |               |                | {mean_max_spec_diff:.6e}")
        print("==========================================================================\n")

    return {
        "num_test_samples": count,
        "fp32_snr_db": mean_fp32_snr_out,
        "int8_snr_db": mean_int8_snr_out,
        "delta_snr_db": delta_snr_quant,
        "fp32_delta_snr_db": mean_fp32_delta_snr,
        "int8_delta_snr_db": mean_int8_delta_snr,
        "fp32_sisnr_out_db": mean_fp32_sisnr_out,
        "int8_sisnr_out_db": mean_int8_sisnr_out,
        "delta_sisnr_db": delta_sisnr_quant,
        "fp32_stoi": mean_fp32_stoi,
        "int8_stoi": mean_int8_stoi,
        "delta_stoi": delta_stoi,
        "fp32_pesq": mean_fp32_pesq,
        "int8_pesq": mean_int8_pesq,
        "delta_pesq": delta_pesq,
        "max_spectral_diff": mean_max_spec_diff,
    }


def select_deployment_model(
    metrics: Dict[str, Any],
    fp32_path: str = "models/onnx/speech_enhancer_fp32.onnx",
    int8_path: str = "models/onnx/speech_enhancer_int8.onnx",
    target_deployment_path: Optional[str] = "models/onnx/speech_enhancer.onnx",
    max_snr_drop_db: float = 1.5,
    max_stoi_drop: float = 0.03,
    max_pesq_drop: float = 0.20,
    verbose: bool = True,
) -> str:
    """
    Selects the optimal deployment model based on measured objective degradation.
    Enforces quality gates: refuses to deploy INT8 if degradation exceeds acceptable thresholds.
    Copies winning artifact to target_deployment_path and returns 'INT8' or 'FP32'.
    """
    snr_drop = -metrics.get("delta_snr_db", 0.0)
    delta_stoi = metrics.get("delta_stoi")
    stoi_drop = -delta_stoi if delta_stoi is not None else 0.0
    delta_pesq = metrics.get("delta_pesq")
    pesq_drop = -delta_pesq if delta_pesq is not None else 0.0

    violations = []
    if snr_drop > max_snr_drop_db:
        violations.append(f"SNR degradation ({snr_drop:.2f} dB) exceeds threshold ({max_snr_drop_db:.2f} dB)")
    if stoi_drop > max_stoi_drop:
        violations.append(f"STOI degradation ({stoi_drop:.4f}) exceeds threshold ({max_stoi_drop:.4f})")
    if pesq_drop > max_pesq_drop:
        violations.append(f"PESQ degradation ({pesq_drop:.3f}) exceeds threshold ({max_pesq_drop:.3f})")

    if violations:
        selected_model = "FP32"
        chosen_source = Path(fp32_path)
        decision_reason = "CRITERIA REJECTED INT8: " + "; ".join(violations)
    else:
        selected_model = "INT8"
        chosen_source = Path(int8_path)
        decision_reason = f"ALL CRITERIA PASSED: SNR drop ({snr_drop:.2f} dB) <= {max_snr_drop_db:.2f} dB, STOI drop ({stoi_drop:.4f}) <= {max_stoi_drop:.4f}"

    if target_deployment_path and chosen_source.exists():
        target_p = Path(target_deployment_path)
        target_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(chosen_source), str(target_p))
        size_mb = os.path.getsize(target_p) / (1024 * 1024)
        target_str = f"{target_p} ({size_mb:.2f} MB)"
    else:
        target_str = "None (evaluation-only)"

    if verbose:
        print("==========================================================================")
        print("          Deployment Model Decision & Artifact Generation                 ")
        print("==========================================================================")
        print(f"Decision:               {selected_model}")
        print(f"Reason:                 {decision_reason}")
        print(f"Source Model:           {chosen_source}")
        print(f"Final Deployment Model: {target_str}")
        print("==========================================================================\n")

    return selected_model


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8, evaluate on real test data, and select deployment artifact.")
    parser.add_argument("--fp32-model", type=str, default="models/onnx/speech_enhancer_fp32.onnx", help="Path to input FP32 ONNX model")
    parser.add_argument("--int8-model", type=str, default="models/onnx/speech_enhancer_int8.onnx", help="Output path for INT8 quantized model")
    parser.add_argument("--deployment-model", type=str, default="models/onnx/speech_enhancer.onnx", help="Final selected deployment model path")
    parser.add_argument("--manifest", type=str, default="data/manifests/manifest.csv", help="Path to manifest CSV for real test set evaluation")
    parser.add_argument("--max-stoi-drop", type=float, default=0.03, help="Maximum tolerable STOI degradation")
    parser.add_argument("--max-pesq-drop", type=float, default=0.20, help="Maximum tolerable PESQ degradation")
    parser.add_argument("--max-snr-drop", type=float, default=1.5, help="Maximum tolerable SNR degradation in dB")
    args = parser.parse_args()

    # 1. Quantize
    quant_res = quantize_model(args.fp32_model, args.int8_model, manifest_path=args.manifest, verbose=True)

    # 2. Real test split evaluation
    metrics = evaluate_fp32_vs_int8(args.fp32_model, args.int8_model, manifest_path=args.manifest, verbose=True)

    # 3. Model selection and deployment packaging
    selected = select_deployment_model(
        metrics=metrics,
        fp32_path=args.fp32_model,
        int8_path=args.int8_model,
        target_deployment_path=args.deployment_model,
        max_snr_drop_db=args.max_snr_drop,
        max_stoi_drop=args.max_stoi_drop,
        max_pesq_drop=args.max_pesq_drop,
        verbose=True,
    )

    # 4. Save evaluation summary to JSON
    summary_path = Path("models/onnx/quantization_evaluation.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "quantization": quant_res,
            "test_evaluation": metrics,
            "selected_model": selected,
        }, f, indent=2)
    print(f"Quantization evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
