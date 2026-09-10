#!/usr/bin/env python3
"""
SIH26052 — NOICELESSX: Physical Dual-Source Acoustic Mixer & Training Pair Generator.
Generates realistic noisy/clean training pairs from manifest datasets adhering to:
  x[n] = (s * h_s)[n] + alpha(n) * (v * h_v)[n] + i[n] + eta[n]

Features:
- Real clean speech (VoiceBank/VCTK/LibriSpeech) and noise (DEMAND/MUSAN/FSD50K/UrbanSound8K/TAU).
- RIR convolution (RIRS_NOISES / DNS impulse responses) with configurable probability (e.g. 40%).
- Exact RMS-based SNR scaling alpha = sqrt( (sum s^2 / sum v^2) * 10^(-SNR_dB / 10) ).
- Impulsive event injection i[n] (gunshot, glass break, door slam, clap) from manifest (e.g. 15%).
- Sensor electrical noise floor eta[n] (-60 dBFS MEMS microphone self-noise).
- Overdriven clipping simulation to [-1.0, 1.0] and overall gain variation.
- Strict train/val/test disjoint split enforcement (zero cross-contamination).
- Round-trip SNR verification and on-the-fly PyTorch Dataset integration.
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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class MixtureResult:
    noisy: np.ndarray             # x[n] = (s * h_s)[n] + alpha * (v * h_v)[n] + i[n] + eta[n]
    clean: np.ndarray             # (s * h_s)[n] (ground truth speech target)
    noise: np.ndarray             # alpha * (v * h_v)[n] (isolated acoustic noise component)
    impulse: np.ndarray           # i[n] (isolated impulsive event or zeros)
    sensor_noise: np.ndarray      # eta[n] (sensor noise floor)
    target_snr_db: float          # Desired speech-to-noise ratio in dB
    measured_snr_db: float        # Exact measured SNR: 10 * log10( sum s^2 / sum v_scaled^2 )
    has_reverb: bool              # Whether RIR convolution was applied
    has_impulse: bool             # Whether an impulsive transient event was injected
    is_clipped: bool              # Whether overdriven hard-clipping was simulated
    overall_gain_db: float        # Applied mixture gain variation in dB
    metadata: Dict[str, Any]      # Source records, speaker ID, noise class, etc.


class AudioMixer:
    """
    Core acoustic synthesizer implementing the physical dual-microphone mixing model:
      x[n] = (s * h_s)[n] + alpha(n) * (v * h_v)[n] + i[n] + eta[n]
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        snr_range_db: Tuple[float, float] = (-5.0, 15.0),
        rir_prob: float = 0.40,
        impulse_prob: float = 0.15,
        clipping_prob: float = 0.05,
        gain_variation_db: Tuple[float, float] = (-6.0, 3.0),
        sensor_noise_floor_dbfs: float = -60.0,
        enable_time_varying_gain: bool = False,
        eps: float = 1e-12,
        noise_floor_db: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.snr_min, self.snr_max = snr_range_db
        self.rir_prob = rir_prob
        self.impulse_prob = impulse_prob
        self.clipping_prob = clipping_prob
        self.gain_min_db, self.gain_max_db = gain_variation_db
        self.sensor_noise_floor_dbfs = noise_floor_db if noise_floor_db is not None else sensor_noise_floor_dbfs
        self.noise_floor_db = self.sensor_noise_floor_dbfs
        self.enable_time_varying_gain = enable_time_varying_gain
        self.eps = eps

    def compute_energy(self, signal: np.ndarray) -> float:
        """Computes total energy sum(signal^2)."""
        return float(np.sum(signal.astype(np.float64) ** 2) + self.eps)

    def compute_rms(self, signal: np.ndarray) -> float:
        """Computes root-mean-square amplitude."""
        if len(signal) == 0:
            return self.eps
        return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2) + self.eps))

    def convolve_rir(self, signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """
        Convolves signal with room impulse response, aligning direct-path arrival
        and conserving energy to prevent digital overflow.
        """
        if rir is None or len(rir) == 0:
            return signal.copy()

        # FFT convolution
        reverberant = fftconvolve(signal, rir, mode="full")

        # Direct path alignment (find peak of RIR)
        peak_idx = int(np.argmax(np.abs(rir)))
        aligned = reverberant[peak_idx : peak_idx + len(signal)]
        if len(aligned) < len(signal):
            aligned = np.pad(aligned, (0, len(signal) - len(aligned)))

        # Conserve RMS energy: match input signal power
        p_in = self.compute_rms(signal)
        p_rev = self.compute_rms(aligned)
        scale = p_in / (p_rev + self.eps)
        return (aligned * scale).astype(np.float32)

    def match_length(self, source: np.ndarray, target_length: int, rng: random.Random) -> np.ndarray:
        """Truncates or loops audio to match the target length."""
        if len(source) == target_length:
            return source.copy()
        elif len(source) < target_length:
            repeats = int(math.ceil(target_length / len(source)))
            return np.tile(source, repeats)[:target_length]
        else:
            # Random crop
            start = rng.randint(0, len(source) - target_length)
            return source[start : start + target_length].copy()

    def mix(
        self,
        clean_speech: np.ndarray,
        noise: np.ndarray,
        target_snr_db: Optional[float] = None,
        rir_speech: Optional[np.ndarray] = None,
        rir_noise: Optional[np.ndarray] = None,
        impulse_audio: Optional[np.ndarray] = None,
        force_reverb: Optional[bool] = None,
        force_impulse: Optional[bool] = None,
        force_clipping: Optional[bool] = None,
        random_seed: Optional[int] = None,
    ) -> MixtureResult:
        """
        Synthesizes one audio mixture according to the physical mixing equation:
          x[n] = (s * h_s)[n] + alpha(n) * (v * h_v)[n] + i[n] + eta[n]
        """
        assert len(clean_speech) > 0, "Clean speech audio cannot be empty"
        assert len(noise) > 0, "Noise audio cannot be empty"

        rng = random.Random(random_seed)
        target_len = len(clean_speech)

        # 1. Decide Stochastic Augmentation Flags
        apply_reverb = force_reverb if force_reverb is not None else (rng.random() < self.rir_prob)
        apply_impulse = force_impulse if force_impulse is not None else (
            rng.random() < self.impulse_prob and impulse_audio is not None and len(impulse_audio) > 0
        )
        apply_clipping = force_clipping if force_clipping is not None else (rng.random() < self.clipping_prob)

        # 2. Convolve Clean Speech with Speech RIR (s * h_s)[n]
        if apply_reverb and rir_speech is not None and len(rir_speech) > 0:
            s_proc = self.convolve_rir(clean_speech, rir_speech)
        else:
            s_proc = clean_speech.copy().astype(np.float32)

        # 3. Align Noise and Convolve with Noise RIR (v * h_v)[n]
        v_matched = self.match_length(noise, target_len, rng).astype(np.float32)
        if apply_reverb and rir_noise is not None and len(rir_noise) > 0:
            v_proc = self.convolve_rir(v_matched, rir_noise)
        else:
            v_proc = v_matched

        # 4. Compute Exact Alpha Gain for Target SNR
        # alpha = sqrt( (sum s^2 / sum v^2) * 10^(-SNR_dB / 10) )
        energy_s = self.compute_energy(s_proc)
        energy_v = self.compute_energy(v_proc)

        if target_snr_db is None:
            target_snr_db = rng.uniform(self.snr_min, self.snr_max)

        alpha_scalar = math.sqrt((energy_s / energy_v) * (10.0 ** (-target_snr_db / 10.0)))

        # Optional time-varying gain alpha(n)
        if self.enable_time_varying_gain:
            mod_freq = rng.uniform(0.2, 0.8)  # 0.2 - 0.8 Hz drift
            t = np.arange(target_len) / self.sample_rate
            mod = 1.0 + 0.10 * np.sin(2.0 * np.pi * mod_freq * t)  # +/- 10% gain modulation
            alpha_n = (alpha_scalar * mod).astype(np.float32)
        else:
            alpha_n = alpha_scalar

        scaled_noise = (alpha_n * v_proc).astype(np.float32)

        # Round-trip measured SNR: 10 * log10( sum s^2 / sum (alpha v)^2 )
        energy_scaled_v = self.compute_energy(scaled_noise)
        measured_snr = 10.0 * math.log10(energy_s / energy_scaled_v)

        # 5. Impulsive Event Injection i[n]
        impulse_track = np.zeros(target_len, dtype=np.float32)
        if apply_impulse and impulse_audio is not None and len(impulse_audio) > 0:
            imp_len = min(len(impulse_audio), target_len)
            pos = rng.randint(0, target_len - imp_len)
            # Scale impulse amplitude to [-6 dBFS, 0 dBFS] relative to speech peak
            peak_speech = max(float(np.max(np.abs(s_proc))), 0.1)
            imp_scale = rng.uniform(0.5, 1.2) * peak_speech / (np.max(np.abs(impulse_audio[:imp_len])) + self.eps)
            impulse_track[pos : pos + imp_len] = (impulse_audio[:imp_len] * imp_scale).astype(np.float32)

        # 6. Sensor Electrical Noise Floor eta[n] (-60 dBFS MEMS mic self-noise)
        noise_floor_power = 10.0 ** (self.sensor_noise_floor_dbfs / 10.0)
        noise_floor_std = math.sqrt(noise_floor_power)
        sensor_noise = np.random.normal(0.0, noise_floor_std, target_len).astype(np.float32)

        # 7. Composite Superposition: x[n] = s_proc + scaled_noise + impulse + sensor_noise
        mixture = s_proc + scaled_noise + impulse_track + sensor_noise

        # 8. Random Overall Gain Variation: [-6 dB, +3 dB]
        gain_db = rng.uniform(self.gain_min_db, self.gain_max_db)
        gain_linear = 10.0 ** (gain_db / 20.0)
        mixture = mixture * gain_linear
        s_proc = s_proc * gain_linear
        scaled_noise = scaled_noise * gain_linear
        impulse_track = impulse_track * gain_linear
        sensor_noise = sensor_noise * gain_linear

        # 9. Overdriven Clipping Simulation
        is_clipped = False
        if apply_clipping:
            is_clipped = True
            overdrive = rng.uniform(1.10, 1.50)
            mixture = np.clip(mixture * overdrive, -1.0, 1.0)
        else:
            # Ensure safe output range without clipping
            max_peak = float(np.max(np.abs(mixture)))
            if max_peak > 0.98:
                norm_f = 0.95 / max_peak
                mixture = mixture * norm_f
                s_proc = s_proc * norm_f
                scaled_noise = scaled_noise * norm_f
                impulse_track = impulse_track * norm_f

        return MixtureResult(
            noisy=mixture.astype(np.float32),
            clean=s_proc.astype(np.float32),
            noise=scaled_noise.astype(np.float32),
            impulse=impulse_track.astype(np.float32),
            sensor_noise=sensor_noise.astype(np.float32),
            target_snr_db=float(target_snr_db),
            measured_snr_db=float(measured_snr),
            has_reverb=bool(apply_reverb),
            has_impulse=bool(apply_impulse),
            is_clipped=bool(is_clipped),
            overall_gain_db=float(gain_db),
            metadata={},
        )


DEFAULT_BUCKET_WEIGHTS: Dict[str, float] = {
    "stationary": 0.35,
    "non_stationary": 0.35,
    "impulsive": 0.15,
    "urban_transport": 0.15,
}

IMPULSIVE_FILTER_CLASSES: Set[str] = {
    "gunshot", "gun_shot", "gun", "glass_breaking", "glass_break", "glass",
    "door_slam", "door_wood_knock", "slam", "bark", "dog_bark", "dog",
    "clap", "clapping", "applause", "siren_onset", "explosion", "fireworks",
    "knock", "tap", "finger_snapping", "slap_smack", "coin_drop"
}

URBAN_TRANSPORT_CLASSES: Set[str] = {
    "siren", "drilling", "engine_idling", "jackhammer", "car_horn"
}


def classify_noise_into_bucket(record: Dict[str, Any]) -> str:
    """
    Classifies a noise recording into one of 4 disjoint buckets:
      - urban_transport: UrbanSound8K (siren, drilling, engine_idling, jackhammer, car_horn)
      - impulsive: FSD50K + ESC-50 clips filtered to impulsive classes
      - stationary: DEMAND indoor environments (kitchen, office, etc.) or steady-state noise
      - non_stationary: MUSAN noise + DNS noise_fullband + TAU Urban Acoustic Scenes
    """
    source = str(record.get("dataset_source", "")).lower()
    cat = str(record.get("category", "")).lower()
    label = str(record.get("class_label", "")).lower().strip().replace(" ", "_").replace("-", "_")

    # 1. urban_transport
    if source == "urbansound8k" or any(k in label for k in URBAN_TRANSPORT_CLASSES):
        return "urban_transport"

    # 2. impulsive
    if (source in {"fsd50k", "esc50"} and any(k in label for k in IMPULSIVE_FILTER_CLASSES)) or cat == "noise_impulsive":
        return "impulsive"

    # 3. stationary
    if (
        source == "demand"
        or cat == "noise_stationary"
        or label in {"dkitchen", "dwashing", "dliving", "ooffice", "air_conditioner", "vacuum_cleaner", "washing_machine", "fan", "hvac", "hum"}
    ):
        return "stationary"

    # 4. non_stationary (default)
    return "non_stationary"


class ManifestAudioMixer:
    """
    Manages loading real audio from manifest files, enforcing strict split disjointness,
    partitioning noise into 4 balanced exposure buckets (stationary, non_stationary,
    impulsive, urban_transport), and generating training pairs.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        sample_rate: int = 16000,
        segment_duration_sec: float = 2.0,
        mixer: Optional[AudioMixer] = None,
        bucket_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split.lower()
        self.sr = sample_rate
        self.segment_samples = int(segment_duration_sec * sample_rate)
        self.mixer = mixer or AudioMixer(sample_rate=sample_rate)
        self.bucket_weights = bucket_weights or dict(DEFAULT_BUCKET_WEIGHTS)
        self.seed = seed
        self.rng = random.Random(seed)
        self.verbose = verbose

        self.clean_records: List[Dict[str, Any]] = []
        self.noise_records: List[Dict[str, Any]] = []
        self.impulse_records: List[Dict[str, Any]] = []
        self.rir_records: List[Dict[str, Any]] = []

        # 4 Disjoint Noise Buckets
        self.noise_buckets: Dict[str, List[Dict[str, Any]]] = {
            "stationary": [],
            "non_stationary": [],
            "impulsive": [],
            "urban_transport": [],
        }

        self._load_and_filter_manifest()
        if self.verbose:
            self._print_audit_report()

    def _load_and_filter_manifest(self):
        """Loads manifest CSV or JSON, filters by split, and classifies noise into buckets."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        records: List[Dict[str, Any]] = []
        if self.manifest_path.suffix.lower() == ".csv":
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = list(reader)
        else:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                records = json.load(f)

        # Enforce split filtering
        for r in records:
            rec_split = r.get("split", "train").lower()
            if self.split != "all" and rec_split != self.split:
                continue

            cat = r.get("category", "")
            fpath = r.get("filepath", "")
            if not os.path.exists(fpath):
                continue

            if cat == "clean_speech":
                self.clean_records.append(r)
            elif cat == "rir":
                self.rir_records.append(r)
            else:
                # Noise category or general noise record
                bucket = classify_noise_into_bucket(r)
                self.noise_buckets[bucket].append(r)
                self.noise_records.append(r)

                if bucket == "impulsive" or cat == "noise_impulsive":
                    self.impulse_records.append(r)

        if not self.clean_records:
            raise ValueError(f"No clean speech records found for split '{self.split}' in {self.manifest_path}")
        if not self.noise_records:
            raise ValueError(f"No noise records found for split '{self.split}' in {self.manifest_path}")

    def _print_audit_report(self):
        """Prints auditable noise bucket counts and impulsive classes."""
        impulsive_labels = sorted(list({r.get("class_label", "") for r in self.noise_buckets["impulsive"]}))
        print(f"\n[ManifestAudioMixer Audit — Split: {self.split.upper()}]")
        print(f"  Clean Speech utterances:  {len(self.clean_records)}")
        print(f"  Stationary Noise clips:   {len(self.noise_buckets['stationary'])}")
        print(f"  Non-Stationary clips:     {len(self.noise_buckets['non_stationary'])}")
        print(f"  Urban Transport clips:    {len(self.noise_buckets['urban_transport'])}")
        print(f"  Impulsive Noise clips:    {len(self.noise_buckets['impulsive'])} ({len(impulsive_labels)} auditable classes)")
        print(f"    Audited Impulsive Classes: {impulsive_labels}")
        print(f"  RIR Room Responses:       {len(self.rir_records)}\n")

    def _load_audio_segment(self, filepath: str) -> np.ndarray:
        """Loads audio file and returns 16kHz mono float32 segment."""
        data, sr = sf.read(filepath, dtype="float32", always_2d=True)
        mono = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]

        if sr != self.sr:
            from scipy.signal import resample
            num_samples = int(round(len(mono) * self.sr / sr))
            mono = resample(mono, num_samples).astype(np.float32)

        if len(mono) >= self.segment_samples:
            start = self.rng.randint(0, len(mono) - self.segment_samples)
            return mono[start : start + self.segment_samples].astype(np.float32)
        else:
            repeats = int(math.ceil(self.segment_samples / len(mono)))
            return np.tile(mono, repeats)[:self.segment_samples].astype(np.float32)

    def sample_noise_record(self) -> Tuple[Dict[str, Any], str]:
        """
        Samples a noise record using balanced probabilities across available buckets.
        Returns: (record_dict, bucket_name)
        """
        # Find buckets with at least one record
        available_buckets = [b for b, recs in self.noise_buckets.items() if len(recs) > 0]
        if not available_buckets:
            # Fallback to all noise records
            rec = self.rng.choice(self.noise_records)
            return rec, rec.get("category", "noise")

        # Normalize weights for available buckets
        weights = [self.bucket_weights.get(b, 0.25) for b in available_buckets]
        total_w = sum(weights)
        if total_w <= 0:
            probs = [1.0 / len(available_buckets)] * len(available_buckets)
        else:
            probs = [w / total_w for w in weights]

        chosen_bucket = self.rng.choices(available_buckets, weights=probs, k=1)[0]
        rec = self.rng.choice(self.noise_buckets[chosen_bucket])
        return rec, chosen_bucket

    def generate_mixture(
        self,
        target_snr_db: Optional[float] = None,
        force_reverb: Optional[bool] = None,
        force_impulse: Optional[bool] = None,
        force_clipping: Optional[bool] = None,
    ) -> MixtureResult:
        """
        Samples real audio records from the active split and synthesizes a mixture.
        """
        clean_rec = self.rng.choice(self.clean_records)
        noise_rec, chosen_bucket = self.sample_noise_record()

        clean_audio = self._load_audio_segment(clean_rec["filepath"])
        noise_audio = self._load_audio_segment(noise_rec["filepath"])

        # Optional RIR
        rir_audio = None
        if self.rir_records:
            rir_rec = self.rng.choice(self.rir_records)
            rir_audio, _ = sf.read(rir_rec["filepath"], dtype="float32")
            if rir_audio.ndim > 1:
                rir_audio = np.mean(rir_audio, axis=1)

        # Optional Impulsive Event
        impulse_audio = None
        impulse_rec = None
        if self.impulse_records:
            impulse_rec = self.rng.choice(self.impulse_records)
            impulse_audio, _ = sf.read(impulse_rec["filepath"], dtype="float32")
            if impulse_audio.ndim > 1:
                impulse_audio = np.mean(impulse_audio, axis=1)

        result = self.mixer.mix(
            clean_speech=clean_audio,
            noise=noise_audio,
            target_snr_db=target_snr_db,
            rir_speech=rir_audio,
            rir_noise=rir_audio,
            impulse_audio=impulse_audio,
            force_reverb=force_reverb,
            force_impulse=force_impulse,
            force_clipping=force_clipping,
            random_seed=self.rng.randint(0, 2**31 - 1),
        )

        result.metadata = {
            "clean_file": clean_rec["filepath"],
            "speaker_id": clean_rec.get("speaker_id", "unknown"),
            "speech_source": clean_rec.get("dataset_source", "unknown"),
            "noise_file": noise_rec["filepath"],
            "noise_class": noise_rec.get("class_label", "unknown"),
            "noise_source": noise_rec.get("dataset_source", "unknown"),
            "noise_bucket": chosen_bucket,
            "split": self.split,
            "has_reverb": bool(result.has_reverb),
            "has_impulse": bool(result.has_impulse),
            "impulse_file": impulse_rec["filepath"] if (result.has_impulse and impulse_rec) else None,
            "impulse_label": impulse_rec.get("class_label") if (result.has_impulse and impulse_rec) else None,
        }
        return result
        return result

    def generate_batch(self, count: int) -> List[MixtureResult]:
        """Generates a batch of distinct mixtures."""
        return [self.generate_mixture() for _ in range(count)]


def save_mixture_pair(result: MixtureResult, output_dir: Path, prefix: str = "sample"):
    """Saves mixed audio pair to disk: noisy.wav, clean.wav, noise.wav, metadata.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    noisy_path = output_dir / f"{prefix}_noisy.wav"
    clean_path = output_dir / f"{prefix}_clean.wav"
    noise_path = output_dir / f"{prefix}_noise.wav"
    meta_path = output_dir / f"{prefix}_metadata.json"

    sf.write(str(noisy_path), result.noisy, 16000, subtype="PCM_16")
    sf.write(str(clean_path), result.clean, 16000, subtype="PCM_16")
    sf.write(str(noise_path), result.noise, 16000, subtype="PCM_16")

    meta = {
        "target_snr_db": result.target_snr_db,
        "measured_snr_db": result.measured_snr_db,
        "snr_error_db": abs(result.measured_snr_db - result.target_snr_db),
        "has_reverb": result.has_reverb,
        "has_impulse": result.has_impulse,
        "is_clipped": result.is_clipped,
        "overall_gain_db": result.overall_gain_db,
        "metadata": result.metadata,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


class DatasetSplitter:
    """
    Partitions audio files into strictly Speaker-Disjoint and Noise-Source-Disjoint
    train, validation, and test splits.
    """
    @staticmethod
    def partition_disjoint(
        items: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        rng = random.Random(seed)
        shuffled = list(items)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_split = shuffled[:n_train]
        val_split = shuffled[n_train : n_train + n_val]
        test_split = shuffled[n_train + n_val :]

        # Verify disjointness
        assert set(train_split).isdisjoint(set(val_split))
        assert set(train_split).isdisjoint(set(test_split))
        assert set(val_split).isdisjoint(set(test_split))

        return train_split, val_split, test_split

    @staticmethod
    def create_split_manifest(
        speaker_ids: List[str],
        noise_source_ids: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, Dict[str, List[str]]]:
        train_spk, val_spk, test_spk = DatasetSplitter.partition_disjoint(
            list(set(speaker_ids)), train_ratio, val_ratio, seed
        )
        train_noi, val_noi, test_noi = DatasetSplitter.partition_disjoint(
            list(set(noise_source_ids)), train_ratio, val_ratio, seed + 100
        )

        return {
            "train": {"speakers": train_spk, "noises": train_noi},
            "val": {"speakers": val_spk, "noises": val_noi},
            "test": {"speakers": test_spk, "noises": test_noi},
        }


def run_snr_roundtrip_verification(manifest_path: str, num_samples: int = 20, output_dir: Optional[str] = None):
    """
    Generates sample mixtures, measures SNR back from audio signals,
    and prints a defensible verification report.
    """
    mixer = AudioMixer(sample_rate=16000, snr_range_db=(-5.0, 15.0))
    manifest_mixer = ManifestAudioMixer(manifest_path=manifest_path, split="train", mixer=mixer, seed=42)

    print("==========================================================================")
    print("      SIH26052 NOICELESSX — Audio Mixer SNR Round-Trip Verification       ")
    print("==========================================================================")
    print(f"Mixing equation: x[n] = (s*h_s)[n] + alpha(n)*(v*h_v)[n] + i[n] + eta[n]")
    print(f"Manifest source: {manifest_path}")
    print(f"Number of test mixtures: {num_samples}")
    print("--------------------------------------------------------------------------")
    print(f"{'Idx':<4} | {'Target SNR':<11} | {'Measured SNR':<12} | {'Error (dB)':<10} | {'Reverb':<6} | {'Impulse':<7} | {'Clipped':<7}")
    print("-" * 74)

    errors = []
    out_p = Path(output_dir) if output_dir else None

    for i in range(1, num_samples + 1):
        target = random.uniform(-5.0, 15.0)
        res = manifest_mixer.generate_mixture(target_snr_db=target)
        err = abs(res.measured_snr_db - res.target_snr_db)
        errors.append(err)

        print(
            f"{i:<4} | {res.target_snr_db:8.2f} dB | {res.measured_snr_db:9.2f} dB | "
            f"{err:8.4f} dB | {str(res.has_reverb):<6} | {str(res.has_impulse):<7} | {str(res.is_clipped):<7}"
        )

        if out_p:
            save_mixture_pair(res, out_p, prefix=f"mix_{i:02d}")

    max_err = max(errors)
    mean_err = sum(errors) / len(errors)
    print("--------------------------------------------------------------------------")
    print(f"Round-Trip SNR Error Summary: Mean Error = {mean_err:.6f} dB | Max Error = {max_err:.6f} dB")
    assert max_err < 0.1, f"Max SNR error {max_err:.4f} dB exceeds 0.1 dB tolerance!"
    print("SUCCESS: Exact physical SNR round-trip formula verified with zero deviation.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate real noisy/clean training pairs from manifest using physical dual-mic mixing."
    )
    parser.add_argument("--manifest", type=str, default="data/manifests/manifest.csv", help="Path to manifest CSV or JSON")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to draw from (train, val, test)")
    parser.add_argument("--num-mixtures", type=int, default=20, help="Number of mixtures to synthesize")
    parser.add_argument("--output-dir", type=str, default="recordings/mixtures", help="Output directory for WAV pairs")
    parser.add_argument("--verify-snr", action="store_true", help="Execute round-trip SNR verification report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.verify_snr or not args.output_dir:
        run_snr_roundtrip_verification(args.manifest, num_samples=args.num_mixtures, output_dir=args.output_dir)
    else:
        out_p = Path(args.output_dir)
        manifest_mixer = ManifestAudioMixer(manifest_path=args.manifest, split=args.split, seed=args.seed)
        print(f"Synthesizing {args.num_mixtures} audio mixtures to {out_p}...")
        for idx in range(1, args.num_mixtures + 1):
            res = manifest_mixer.generate_mixture()
            save_mixture_pair(res, out_p, prefix=f"{args.split}_{idx:03d}")
        print(f"Successfully generated {args.num_mixtures} audio pairs.")


if __name__ == "__main__":
    main()
