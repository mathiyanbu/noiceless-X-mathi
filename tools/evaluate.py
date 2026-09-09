#!/usr/bin/env python3
"""
SIH26052 — NOICELESSX: Offline Objective Evaluation Tool.

Computes real evaluation metrics from actual recorded clean/noisy/enhanced triples:
  - True SNR_in  = 10*log10( Σs^2 / (Σ(x-s)^2 + eps) )
  - True SNR_out = 10*log10( Σs^2 / (Σ(s_hat-s)^2 + eps) )
  - ΔSNR = SNR_out - SNR_in
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
  - STOI (Short-Time Objective Intelligibility) using pystoi
  - PESQ (Perceptual Evaluation of Speech Quality) using pesq (WB/NB)

Strict Zero-Mock Policy: Never fabricates or invents placeholder scores.
Missing libraries return 'N/A' rather than estimated proxy numbers.
"""

import os
import sys
import argparse
import json
import csv
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import soundfile as sf

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ai.evaluation.metrics import (
    compute_true_snr,
    compute_si_sdr,
    compute_stoi,
    compute_pesq,
    _HAS_PYSTOI,
    _HAS_PESQ
)


def load_and_validate_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file, convert to mono float32, and check sample rate."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    data, sr = sf.read(path, dtype="float32")

    # If stereo/multichannel, take first channel
    if data.ndim > 1:
        data = data[:, 0]

    # Check for NaN / Inf
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

    return data, sr


def evaluate_triple(
    clean_path: str,
    noisy_path: str,
    enhanced_path: str,
    target_sr: int = 16000,
    eps: float = 1e-10
) -> Dict[str, Any]:
    """
    Evaluate a single clean/noisy/enhanced audio triple.
    """
    clean_data, sr_clean = load_and_validate_audio(clean_path, target_sr)
    noisy_data, sr_noisy = load_and_validate_audio(noisy_path, target_sr)
    enhanced_data, sr_enh = load_and_validate_audio(enhanced_path, target_sr)

    # Sample rate verification
    if sr_clean != target_sr or sr_noisy != target_sr or sr_enh != target_sr:
        print(f"[Warning] Sample rate mismatch: clean={sr_clean}, noisy={sr_noisy}, enhanced={sr_enh}. Expected {target_sr} Hz.")

    # Align lengths
    min_len = min(len(clean_data), len(noisy_data), len(enhanced_data))
    if min_len == 0:
        raise ValueError(f"Empty audio file detected in triple: {clean_path}")

    s = clean_data[:min_len]
    x = noisy_data[:min_len]
    s_hat = enhanced_data[:min_len]

    # 1. Exact True SNR calculation
    snr_res = compute_true_snr(s, x, s_hat, eps=eps)

    # 2. Scale-Invariant SDR
    si_sdr = compute_si_sdr(s, s_hat, eps=eps)
    si_sdr_in = compute_si_sdr(s, x, eps=eps)

    # 3. STOI (pystoi)
    stoi_enhanced = compute_stoi(s, s_hat, sample_rate=sr_clean)
    stoi_noisy = compute_stoi(s, x, sample_rate=sr_clean)
    delta_stoi = (stoi_enhanced - stoi_noisy) if (stoi_enhanced is not None and stoi_noisy is not None) else None

    # 4. PESQ (pesq library)
    pesq_enhanced = compute_pesq(s, s_hat, sample_rate=sr_clean)
    pesq_noisy = compute_pesq(s, x, sample_rate=sr_clean)
    delta_pesq = (pesq_enhanced - pesq_noisy) if (pesq_enhanced is not None and pesq_noisy is not None) else None

    duration_sec = min_len / float(sr_clean)

    return {
        "clean_file": os.path.basename(clean_path),
        "noisy_file": os.path.basename(noisy_path),
        "enhanced_file": os.path.basename(enhanced_path),
        "duration_sec": duration_sec,
        "sample_rate": sr_clean,
        "samples": min_len,
        "snr_in_db": snr_res["snr_in"],
        "snr_out_db": snr_res["snr_out"],
        "delta_snr_db": snr_res["delta_snr"],
        "si_sdr_in_db": si_sdr_in,
        "si_sdr_out_db": si_sdr,
        "delta_si_sdr_db": si_sdr - si_sdr_in,
        "stoi_noisy": stoi_noisy,
        "stoi_enhanced": stoi_enhanced,
        "delta_stoi": delta_stoi,
        "pesq_noisy": pesq_noisy,
        "pesq_enhanced": pesq_enhanced,
        "delta_pesq": delta_pesq
    }


def format_metric(val: Optional[float], fmt: str = "{:.2f}") -> str:
    """Format float or display honest N/A."""
    if val is None or np.isnan(val):
        return "N/A"
    return fmt.format(val)


def print_evaluation_table(results: List[Dict[str, Any]]):
    """Render comprehensive engineering evaluation table to stdout."""
    print("\n" + "=" * 105)
    print("                    SIH26052 NOICELESSX — OFFLINE OBJECTIVE EVALUATION                   ")
    print("=" * 105)
    header = (
        f"{'Item':<4} {'File':<24} {'SNR_in':<9} {'SNR_out':<9} {'ΔSNR (dB)':<10} "
        f"{'SI-SDR':<9} {'STOI':<8} {'ΔSTOI':<8} {'PESQ':<8} {'ΔPESQ':<8}"
    )
    print(header)
    print("-" * 105)

    snr_in_list = []
    snr_out_list = []
    delta_snr_list = []
    si_sdr_list = []
    stoi_list = []
    pesq_list = []

    for idx, r in enumerate(results, 1):
        name = r["enhanced_file"]
        if len(name) > 22:
            name = name[:19] + "..."

        snr_in = r["snr_in_db"]
        snr_out = r["snr_out_db"]
        d_snr = r["delta_snr_db"]
        si_sdr = r["si_sdr_out_db"]
        stoi_enh = r["stoi_enhanced"]
        d_stoi = r["delta_stoi"]
        pesq_enh = r["pesq_enhanced"]
        d_pesq = r["delta_pesq"]

        snr_in_list.append(snr_in)
        snr_out_list.append(snr_out)
        delta_snr_list.append(d_snr)
        si_sdr_list.append(si_sdr)
        if stoi_enh is not None:
            stoi_list.append(stoi_enh)
        if pesq_enh is not None:
            pesq_list.append(pesq_enh)

        row = (
            f"{idx:<4} {name:<24} "
            f"{format_metric(snr_in):<9} "
            f"{format_metric(snr_out):<9} "
            f"{format_metric(d_snr, '+{:.2f}'):<10} "
            f"{format_metric(si_sdr):<9} "
            f"{format_metric(stoi_enh, '{:.3f}'):<8} "
            f"{format_metric(d_stoi, '+{:.3f}'):<8} "
            f"{format_metric(pesq_enh, '{:.2f}'):<8} "
            f"{format_metric(d_pesq, '+{:.2f}'):<8}"
        )
        print(row)

    print("-" * 105)
    # Average Summary Row
    avg_snr_in = np.mean(snr_in_list) if snr_in_list else 0.0
    avg_snr_out = np.mean(snr_out_list) if snr_out_list else 0.0
    avg_delta_snr = np.mean(delta_snr_list) if delta_snr_list else 0.0
    avg_si_sdr = np.mean(si_sdr_list) if si_sdr_list else 0.0
    avg_stoi = np.mean(stoi_list) if stoi_list else None
    avg_pesq = np.mean(pesq_list) if pesq_list else None

    summary_row = (
        f"{'AVG':<4} {'[ALL SAMPLES]':<24} "
        f"{format_metric(avg_snr_in):<9} "
        f"{format_metric(avg_snr_out):<9} "
        f"{format_metric(avg_delta_snr, '+{:.2f}'):<10} "
        f"{format_metric(avg_si_sdr):<9} "
        f"{format_metric(avg_stoi, '{:.3f}'):<8} "
        f"{'--':<8} "
        f"{format_metric(avg_pesq, '{:.2f}'):<8} "
        f"{'--':<8}"
    )
    print(summary_row)
    print("=" * 105)

    if not _HAS_PYSTOI:
        print("[Notice] pystoi is not installed. To calculate STOI, install with: pip install pystoi")
    if not _HAS_PESQ:
        print("[Notice] pesq library is not installed on this host. Run on Raspberry Pi / Linux or install C++ build tools for PESQ.")
    print()


def main():
    parser = argparse.ArgumentParser(description="NOICELESSX Offline Objective Speech Enhancement Evaluator")
    parser.add_argument("--clean", help="Path to clean ground-truth speech WAV file")
    parser.add_argument("--noisy", help="Path to noisy microphone input WAV file")
    parser.add_argument("--enhanced", help="Path to enhanced speech output WAV file")
    parser.add_argument("--clean_dir", help="Directory containing clean ground-truth WAV files")
    parser.add_argument("--noisy_dir", help="Directory containing noisy microphone input WAV files")
    parser.add_argument("--enhanced_dir", help="Directory containing enhanced speech output WAV files")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate in Hz (default: 16000)")
    parser.add_argument("--output_json", help="Path to export evaluation metrics as JSON")
    parser.add_argument("--output_csv", help="Path to export evaluation metrics as CSV")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []

    # 1. Single triple evaluation
    if args.clean and args.noisy and args.enhanced:
        print(f"[evaluate] Evaluating single triple:")
        print(f"  Clean:    {args.clean}")
        print(f"  Noisy:    {args.noisy}")
        print(f"  Enhanced: {args.enhanced}")
        res = evaluate_triple(args.clean, args.noisy, args.enhanced, target_sr=args.sample_rate)
        results.append(res)

    # 2. Batch directory evaluation
    elif args.clean_dir and args.noisy_dir and args.enhanced_dir:
        print(f"[evaluate] Batch evaluation across directories:")
        print(f"  Clean Dir:    {args.clean_dir}")
        print(f"  Noisy Dir:    {args.noisy_dir}")
        print(f"  Enhanced Dir: {args.enhanced_dir}")

        clean_files = sorted([f for f in os.listdir(args.clean_dir) if f.lower().endswith(".wav")])
        if not clean_files:
            print(f"[Error] No .wav files found in clean directory: {args.clean_dir}")
            sys.exit(1)

        for cf in clean_files:
            c_path = os.path.join(args.clean_dir, cf)
            n_path = os.path.join(args.noisy_dir, cf)
            e_path = os.path.join(args.enhanced_dir, cf)

            if not os.path.exists(n_path) or not os.path.exists(e_path):
                print(f"[Warning] Skipping {cf}: matching noisy or enhanced file not found.")
                continue

            try:
                res = evaluate_triple(c_path, n_path, e_path, target_sr=args.sample_rate)
                results.append(res)
            except Exception as ex:
                print(f"[Error] Failed to evaluate {cf}: {ex}")

    else:
        print("[Error] Please specify either single triple (--clean, --noisy, --enhanced) or directory batch (--clean_dir, --noisy_dir, --enhanced_dir).")
        parser.print_help()
        sys.exit(1)

    if not results:
        print("[Error] No valid audio triples could be evaluated.")
        sys.exit(1)

    # Render results table
    print_evaluation_table(results)

    # Export JSON if requested
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"evaluation_results": results}, f, indent=2)
        print(f"[evaluate] JSON metrics saved to: {args.output_json}")

    # Export CSV if requested
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        headers = list(results[0].keys())
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"[evaluate] CSV metrics saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
