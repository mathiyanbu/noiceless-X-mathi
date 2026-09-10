#!/usr/bin/env python3
"""
SIH26052 — NOICELESSX: Audio Decode Verification, Taxonomy Indexing & Disjoint Manifest Generator.
Scans audio directories, decodes every file to verify audio integrity (zero corruption),
categorizes into stationary/nonstationary/impulsive noise, clean speech, and RIRs,
enforces strict speaker and noise recording disjointness across train/val/test splits,
and generates structured CSV/JSON manifests with comprehensive summary telemetry.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.datasets.sources import (
    DEMAND_CLASSES,
    ESC50_CLASSES,
    FSD50K_EVENT_CLASSES,
    IMPULSIVE_CLASSES,
    STEADY_STATE_NOISE_CLASSES,
    TAU_SCENES,
    URBANSOUND_CLASSES,
    is_impulsive_class,
)

# Valid categories enforced across NOICELESSX pipeline
VALID_CATEGORIES = {
    "clean_speech",
    "noise_stationary",
    "noise_nonstationary",
    "noise_impulsive",
    "rir",
}


@dataclass
class ManifestEntry:
    filepath: str
    dataset_source: str
    category: str  # clean_speech, noise_stationary, noise_nonstationary, noise_impulsive, rir
    duration_sec: float
    sample_rate: int
    class_label: str
    split: str  # train, val, test
    speaker_id: Optional[str] = None
    channels: int = 1
    rms_dbfs: float = -96.0


def verify_and_decode_audio_file(filepath: Path) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Actually opens and decodes the audio file to verify integrity.
    Rejects corrupted headers, non-finite values (NaN/Inf), and empty files.
    Returns: (is_valid, metadata_dict, error_message)
    """
    try:
        # 1. Inspect header
        info = sf.info(str(filepath))
        if info.frames == 0 or info.duration == 0:
            return False, None, "File contains zero frames/duration"

        # 2. Decode audio samples (entire file or up to 60s to confirm decoding)
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        if len(data) == 0:
            return False, None, "Decoded sample buffer is empty"

        # 3. Check for numerical validity (NaN/Inf)
        if not np.all(np.isfinite(data)):
            return False, None, "File contains NaN or infinite values"

        # 4. Compute RMS dBFS
        rms = float(np.sqrt(np.mean(data ** 2) + 1e-12))
        rms_dbfs = float(20.0 * math.log10(rms))

        meta = {
            "duration_sec": float(info.duration),
            "sample_rate": int(sr),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "rms_dbfs": rms_dbfs,
        }
        return True, meta, None

    except Exception as e:
        return False, None, f"Decoding failed: {str(e)}"


def infer_dataset_source(filepath: Path) -> str:
    """Infers the source research dataset from path heuristics."""
    parts = [p.lower() for p in filepath.parts]
    name = filepath.stem.lower()

    if any("voicebank" in p for p in parts) or name.startswith("p225_") or name.startswith("p226_") or name.startswith("p227_") or name.startswith("p228_"):
        return "voicebank"
    if any("vctk" in p for p in parts):
        return "vctk"
    if any("librispeech" in p for p in parts):
        return "librispeech"
    if any("demand" in p for p in parts) or any(c.lower() in parts for c in DEMAND_CLASSES):
        return "demand"
    if any("esc50" in p or "esc-50" in p for p in parts):
        return "esc50"
    if any("urbansound" in p or "urbansound8k" in p for p in parts):
        return "urbansound8k"
    if any("fsd50k" in p for p in parts):
        return "fsd50k"
    if any("musan" in p for p in parts):
        return "musan"
    if any("tau" in p for p in parts):
        return "tau2020"
    if any("rir" in p for p in parts) or any("openslr" in p for p in parts):
        return "rirs_noises"
    if any("dns" in p for p in parts):
        return "dns5"

    return "general_corpus"


def classify_acoustic_category(class_label: str, source: str, filepath: Path) -> str:
    """
    Categorizes the audio into one of the 5 canonical classes:
    clean_speech, noise_stationary, noise_nonstationary, noise_impulsive, rir
    """
    norm_label = class_label.lower().strip().replace(" ", "_").replace("-", "_")
    norm_path = str(filepath).lower().replace("\\", "/")

    # 1. RIR check
    if source == "rirs_noises" or "/rir" in norm_path or "rir" in norm_label:
        return "rir"

    # 2. Clean Speech check
    if source in {"voicebank", "vctk", "librispeech"} or "/clean_speech" in norm_path:
        return "clean_speech"

    # 3. Impulsive Noise check
    if is_impulsive_class(norm_label) or is_impulsive_class(filepath.stem):
        return "noise_impulsive"

    # 4. Stationary vs Nonstationary Noise
    if (
        norm_label in STEADY_STATE_NOISE_CLASSES
        or any(k in norm_label for k in ["fan", "air_conditioner", "hvac", "hum", "vacuum", "washing", "rain", "wind", "engine_idling"])
        or (source == "demand" and norm_label in {"DKITCHEN", "DWASHING", "DLIVING", "OOFFICE"})
    ):
        return "noise_stationary"

    # Default environmental/urban noise is nonstationary
    return "noise_nonstationary"


def extract_speaker_id(filepath: Path, source: str) -> Optional[str]:
    """Extracts speaker ID from filename or directory structure for speech datasets."""
    stem = filepath.stem
    parts = stem.split("_")

    if source in {"voicebank", "vctk"} and len(parts) >= 2 and parts[0].startswith("p"):
        return parts[0]
    elif source == "librispeech":
        # LibriSpeech format: speaker_id-chapter_id-utterance_id.flac
        spk_parts = stem.split("-")
        if len(spk_parts) >= 2:
            return spk_parts[0]

    # Check parent folder name for speaker ID
    parent_name = filepath.parent.name
    if parent_name.startswith("p") and parent_name[1:].isdigit():
        return parent_name

    return None


class DatasetManifestProcessor:
    """
    Scans, decode-verifies, classifies, and creates disjoint splits for audio datasets.
    """

    def __init__(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def scan_directory(self, input_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
        """Scans and decode-verifies all audio files under input_dir."""
        valid_files: List[Tuple[Path, Dict[str, Any]]] = []
        extensions = {".wav", ".flac", ".ogg", ".mp3"}

        print(f"Scanning directory: {input_dir}")
        all_candidates = [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions]
        print(f"Found {len(all_candidates)} audio candidate files. Decoding and verifying...")

        for idx, f in enumerate(all_candidates, 1):
            is_valid, meta, err = verify_and_decode_audio_file(f)
            if is_valid and meta:
                valid_files.append((f, meta))
            else:
                print(f"  [CORRUPT/REJECTED] {f.name}: {err}")

            if idx % 100 == 0 or idx == len(all_candidates):
                print(f"\r  Verified {idx}/{len(all_candidates)} files...", end="", flush=True)

        print(f"\nSuccessfully verified and decoded {len(valid_files)} valid audio files.")
        return valid_files

    def process_manifest(
        self,
        valid_files: List[Tuple[Path, Dict[str, Any]]],
    ) -> Tuple[List[ManifestEntry], Dict[str, Any]]:
        """
        Assigns categories, extracts speakers/labels, and enforces strict disjoint splits.
        """
        rng = random.Random(self.seed)

        # 1. Initial parsing
        speech_items: List[Tuple[Path, Dict[str, Any], str, str, str]] = []  # (path, meta, source, spk, label)
        noise_items: List[Tuple[Path, Dict[str, Any], str, str, str]] = []   # (path, meta, source, cat, label)
        rir_items: List[Tuple[Path, Dict[str, Any], str, str, str]] = []

        for fpath, meta in valid_files:
            source = infer_dataset_source(fpath)
            class_label = fpath.parent.name if fpath.parent.name != fpath.stem else "ambient"
            category = classify_acoustic_category(class_label, source, fpath)
            speaker_id = extract_speaker_id(fpath, source) if category == "clean_speech" else None

            if category == "clean_speech":
                spk = speaker_id or f"spk_{fpath.stem.split('_')[0]}"
                speech_items.append((fpath, meta, source, spk, class_label))
            elif category == "rir":
                rir_items.append((fpath, meta, source, "rir", class_label))
            else:
                noise_items.append((fpath, meta, source, category, class_label))

        # 2. Strict Disjoint Splitting for Clean Speech (By Speaker ID)
        spk_to_items: Dict[str, List[Tuple[Path, Dict[str, Any], str, str, str]]] = {}
        for item in speech_items:
            spk = item[3]
            spk_to_items.setdefault(spk, []).append(item)

        unique_speakers = sorted(list(spk_to_items.keys()))
        rng.shuffle(unique_speakers)

        n_spk = len(unique_speakers)
        n_test_spk = max(1, int(round(n_spk * self.test_ratio))) if n_spk > 2 else 0
        n_val_spk = max(1, int(round(n_spk * self.val_ratio))) if n_spk > 2 else 0

        test_speakers = set(unique_speakers[:n_test_spk])
        val_speakers = set(unique_speakers[n_test_spk : n_test_spk + n_val_spk])
        train_speakers = set(unique_speakers[n_test_spk + n_val_spk:])

        # Guard against empty train split
        if not train_speakers and unique_speakers:
            train_speakers = set(unique_speakers)
            val_speakers = set(unique_speakers)
            test_speakers = set(unique_speakers)

        entries: List[ManifestEntry] = []

        for spk, items in spk_to_items.items():
            split = "train"
            if spk in test_speakers and len(test_speakers) < len(unique_speakers):
                split = "test"
            elif spk in val_speakers and len(val_speakers) < len(unique_speakers):
                split = "val"

            for fpath, meta, source, spk_id, label in items:
                entries.append(ManifestEntry(
                    filepath=str(fpath.resolve()),
                    dataset_source=source,
                    category="clean_speech",
                    duration_sec=meta["duration_sec"],
                    sample_rate=meta["sample_rate"],
                    class_label=label,
                    split=split,
                    speaker_id=spk_id,
                    channels=meta["channels"],
                    rms_dbfs=meta["rms_dbfs"],
                ))

        # 3. Strict Disjoint Splitting for Noise Recordings (By Recording File Path)
        # Each distinct noise recording file is uniquely assigned to train, val, or test
        noise_files = sorted(noise_items, key=lambda x: str(x[0]))
        rng.shuffle(noise_files)

        n_noise = len(noise_files)
        n_test_noise = int(round(n_noise * self.test_ratio)) if n_noise > 2 else 0
        n_val_noise = int(round(n_noise * self.val_ratio)) if n_noise > 2 else 0

        for idx, (fpath, meta, source, cat, label) in enumerate(noise_files):
            if idx < n_test_noise and n_test_noise > 0:
                split = "test"
            elif idx < n_test_noise + n_val_noise and n_val_noise > 0:
                split = "val"
            else:
                split = "train"

            entries.append(ManifestEntry(
                filepath=str(fpath.resolve()),
                dataset_source=source,
                category=cat,
                duration_sec=meta["duration_sec"],
                sample_rate=meta["sample_rate"],
                class_label=label,
                split=split,
                speaker_id=None,
                channels=meta["channels"],
                rms_dbfs=meta["rms_dbfs"],
            ))

        # 4. Strict Disjoint Splitting for RIRs
        rir_files = sorted(rir_items, key=lambda x: str(x[0]))
        rng.shuffle(rir_files)
        n_rir = len(rir_files)
        n_test_rir = int(round(n_rir * self.test_ratio)) if n_rir > 2 else 0
        n_val_rir = int(round(n_rir * self.val_ratio)) if n_rir > 2 else 0

        for idx, (fpath, meta, source, cat, label) in enumerate(rir_files):
            if idx < n_test_rir and n_test_rir > 0:
                split = "test"
            elif idx < n_test_rir + n_val_rir and n_val_rir > 0:
                split = "val"
            else:
                split = "train"

            entries.append(ManifestEntry(
                filepath=str(fpath.resolve()),
                dataset_source=source,
                category="rir",
                duration_sec=meta["duration_sec"],
                sample_rate=meta["sample_rate"],
                class_label=label,
                split=split,
                speaker_id=None,
                channels=meta["channels"],
                rms_dbfs=meta["rms_dbfs"],
            ))

        # 5. Compute Real Summary Metrics
        summary = self._compute_summary_metrics(entries)
        return entries, summary

    def _compute_summary_metrics(self, entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Calculates precise hours, category breakdowns, and distinct classes."""
        clean_hours = sum(e.duration_sec for e in entries if e.category == "clean_speech") / 3600.0
        stationary_noise_hours = sum(e.duration_sec for e in entries if e.category == "noise_stationary") / 3600.0
        nonstationary_noise_hours = sum(e.duration_sec for e in entries if e.category == "noise_nonstationary") / 3600.0
        impulsive_noise_hours = sum(e.duration_sec for e in entries if e.category == "noise_impulsive") / 3600.0
        total_noise_hours = stationary_noise_hours + nonstationary_noise_hours + impulsive_noise_hours

        clean_sources = {e.dataset_source for e in entries if e.category == "clean_speech"}
        noise_sources = {e.dataset_source for e in entries if e.category.startswith("noise_")}
        distinct_noise_classes = {e.class_label for e in entries if e.category.startswith("noise_")}
        rir_count = sum(1 for e in entries if e.category == "rir")

        demand_classes = {e.class_label for e in entries if e.dataset_source == "demand"}
        esc50_classes = {e.class_label for e in entries if e.dataset_source == "esc50"}
        urbansound_classes = {e.class_label for e in entries if e.dataset_source == "urbansound8k"}
        fsd50k_classes = {e.class_label for e in entries if e.dataset_source == "fsd50k"}

        train_count = sum(1 for e in entries if e.split == "train")
        val_count = sum(1 for e in entries if e.split == "val")
        test_count = sum(1 for e in entries if e.split == "test")

        summary_text = (
            f"{clean_hours:.2f} hours clean speech across {len(clean_sources)} sources / "
            f"{len(demand_classes)} DEMAND environments + "
            f"{len(fsd50k_classes)} FSD50K + "
            f"{len(esc50_classes)} ESC-50 + "
            f"{len(urbansound_classes)} UrbanSound8K noise categories = "
            f"{len(distinct_noise_classes)} distinct noise categories"
        )

        return {
            "total_records": len(entries),
            "clean_speech_hours": round(clean_hours, 3),
            "clean_sources_count": len(clean_sources),
            "noise_stationary_hours": round(stationary_noise_hours, 3),
            "noise_nonstationary_hours": round(nonstationary_noise_hours, 3),
            "noise_impulsive_hours": round(impulsive_noise_hours, 3),
            "total_noise_hours": round(total_noise_hours, 3),
            "distinct_noise_classes_count": len(distinct_noise_classes),
            "rir_count": rir_count,
            "train_records": train_count,
            "val_records": val_count,
            "test_records": test_count,
            "summary_statement": summary_text,
        }


def print_summary_table(summary: Dict[str, Any]):
    """Prints a beautiful, real numbers summary table to standard output."""
    print("\n==========================================================================")
    print("        SIH26052 NOICELESSX — Dataset Manifest Telemetry & Real Counts    ")
    print("==========================================================================")
    print(f"Total Verified Audio Records:     {summary['total_records']}")
    print(f"Clean Speech Total Hours:         {summary['clean_speech_hours']} hrs ({summary['clean_sources_count']} speech source(s))")
    print(f"Stationary Noise Total Hours:     {summary['noise_stationary_hours']} hrs")
    print(f"Nonstationary Noise Total Hours:  {summary['noise_nonstationary_hours']} hrs")
    print(f"Impulsive Noise Total Hours:      {summary['noise_impulsive_hours']} hrs")
    print(f"Total Noise Pool Duration:        {summary['total_noise_hours']} hrs")
    print(f"Distinct Acoustic Noise Classes:  {summary['distinct_noise_classes_count']}")
    print(f"Room Impulse Responses (RIRs):    {summary['rir_count']}")
    print("--------------------------------------------------------------------------")
    print(f"Split Distribution:               Train: {summary['train_records']} | Val: {summary['val_records']} | Test: {summary['test_records']}")
    print("--------------------------------------------------------------------------")
    print(f"Defensible Summary Statement:")
    print(f"  \"{summary['summary_statement']}\"")
    print("==========================================================================\n")


def export_manifests(entries: List[ManifestEntry], summary: Dict[str, Any], output_dir: Path):
    """Exports unified CSV, JSON, and summary telemetry."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "manifest.csv"
    json_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    # 1. Export CSV
    fieldnames = [
        "filepath", "dataset_source", "category", "duration_sec",
        "sample_rate", "class_label", "split", "speaker_id", "channels", "rms_dbfs"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow(asdict(e))

    # 2. Export JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)

    # 3. Export Summary JSON
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 4. Split-specific JSON manifests for PyTorch loaders
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]
    test_entries = [e for e in entries if e.split == "test"]

    with open(output_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in train_entries], f, indent=2)
    with open(output_dir / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in val_entries], f, indent=2)
    with open(output_dir / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in test_entries], f, indent=2)

    print(f"Manifests exported successfully:")
    print(f"  - CSV:     {csv_path}")
    print(f"  - JSON:    {json_path}")
    print(f"  - Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan, decode-verify, classify, and generate disjoint manifests for audio datasets."
    )
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Input root directory containing audio")
    parser.add_argument("--output-dir", type=str, default="data/manifests", help="Output directory for manifests")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        print(f"[ERROR] Input directory '{input_path}' does not exist.")
        sys.exit(1)

    processor = DatasetManifestProcessor(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    valid_files = processor.scan_directory(input_path)
    if not valid_files:
        print(f"[ERROR] No valid decodable audio files found in '{input_path}'.")
        sys.exit(1)

    entries, summary = processor.process_manifest(valid_files)
    print_summary_table(summary)
    export_manifests(entries, summary, output_path)


if __name__ == "__main__":
    main()
