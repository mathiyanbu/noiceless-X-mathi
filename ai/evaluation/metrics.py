"""
Real Evaluation Metrics & Live Noise-Floor SNR Tracker for NOICELESSX.

Computes:
1. True Ground-Truth SNR (Offline with clean reference):
   SNR_in  = 10*log10( Σs^2 / (Σ(x-s)^2 + eps) )
   SNR_out = 10*log10( Σs^2 / (Σ(s_hat-s)^2 + eps) )
   ΔSNR    = SNR_out - SNR_in
2. Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
3. STOI (Short-Time Objective Intelligibility) via pystoi (never invented)
4. PESQ (Perceptual Evaluation of Speech Quality) via pesq (never invented)
5. Live Noise-Floor SNR Tracker: Recursive minimum/noise-floor power tracking
   during non-speech/VAD-negative frames, clearly labeled as estimated.
"""

from typing import Dict, Optional, Tuple
import numpy as np

try:
    from pystoi import stoi as pystoi_stoi
    _HAS_PYSTOI = True
except ImportError:
    _HAS_PYSTOI = False

try:
    from pesq import pesq as c_pesq
    _HAS_PESQ = True
except ImportError:
    _HAS_PESQ = False


def compute_true_snr(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    eps: float = 1e-10
) -> Dict[str, float]:
    """
    Computes exact ground-truth SNR metrics from actual signals:
      SNR_in  = 10*log10( Σs^2 / (Σ(x-s)^2 + eps) )
      SNR_out = 10*log10( Σs^2 / (Σ(s_hat-s)^2 + eps) )
      ΔSNR    = SNR_out - SNR_in
    """
    min_len = min(len(clean), len(noisy), len(enhanced))
    if min_len == 0:
        return {"snr_in": 0.0, "snr_out": 0.0, "delta_snr": 0.0}

    s = clean[:min_len].astype(np.float64)
    x = noisy[:min_len].astype(np.float64)
    s_hat = enhanced[:min_len].astype(np.float64)

    s_power = np.sum(s**2)
    in_noise_power = np.sum((x - s)**2) + eps
    out_noise_power = np.sum((s_hat - s)**2) + eps

    snr_in = float(10.0 * np.log10((s_power + eps) / in_noise_power))
    snr_out = float(10.0 * np.log10((s_power + eps) / out_noise_power))
    delta_snr = float(snr_out - snr_in)

    return {
        "snr_in": snr_in,
        "snr_out": snr_out,
        "delta_snr": delta_snr
    }


def compute_snr(clean: np.ndarray, enhanced: np.ndarray, eps: float = 1e-10) -> float:
    """Standard SNR between clean reference and enhanced signal in dB."""
    min_len = min(len(clean), len(enhanced))
    if min_len == 0:
        return 0.0
    c = clean[:min_len].astype(np.float64)
    e = enhanced[:min_len].astype(np.float64)
    noise = e - c
    p_signal = np.sum(c**2) + eps
    p_noise = np.sum(noise**2) + eps
    return float(10.0 * np.log10(p_signal / p_noise))


def compute_si_sdr(reference: np.ndarray, estimated: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    Rigorously scales the reference to find optimal projection.
    """
    min_len = min(len(reference), len(estimated))
    if min_len == 0:
        return 0.0

    s = reference[:min_len].astype(np.float64)
    s_hat = estimated[:min_len].astype(np.float64)

    # Zero-mean normalization
    s = s - np.mean(s)
    s_hat = s_hat - np.mean(s_hat)

    dot = np.dot(s_hat, s)
    s_energy = np.dot(s, s) + eps
    alpha = dot / s_energy

    e_target = alpha * s
    e_res = s_hat - e_target

    target_energy = np.sum(e_target**2) + eps
    res_energy = np.sum(e_res**2) + eps

    return float(10.0 * np.log10(target_energy / res_energy))


# Alias for backwards compatibility
compute_sisnr = compute_si_sdr


def compute_stoi(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int = 16000) -> Optional[float]:
    """
    Compute Short-Time Objective Intelligibility (STOI) [0.0, 1.0] using pystoi.
    Returns None if pystoi is not installed — NEVER returns an invented placeholder score.
    """
    min_len = min(len(clean), len(enhanced))
    if min_len < 256:
        return 0.0

    if not _HAS_PYSTOI:
        return None

    try:
        val = pystoi_stoi(clean[:min_len], enhanced[:min_len], sample_rate, extended=False)
        return float(np.clip(val, 0.0, 1.0))
    except Exception:
        return None


def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int = 16000) -> Optional[float]:
    """
    Compute Perceptual Evaluation of Speech Quality (PESQ) [-0.5, 4.5] using pesq library.
    Returns None if pesq is not installed — NEVER returns an invented placeholder score.
    """
    min_len = min(len(clean), len(enhanced))
    if min_len < 512:
        return None

    if not _HAS_PESQ:
        return None

    if sample_rate not in (8000, 16000):
        return None

    try:
        mode = "wb" if sample_rate == 16000 else "nb"
        val = c_pesq(sample_rate, clean[:min_len], enhanced[:min_len], mode)
        return float(val)
    except Exception:
        return None


def evaluate_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int = 16000
) -> Dict[str, Optional[float]]:
    """
    Compute full objective evaluation metrics on clean/noisy/enhanced triple.
    Only computes real metrics from actual signals; missing libraries return None.
    """
    true_snr = compute_true_snr(clean, noisy, enhanced)
    si_sdr = compute_si_sdr(clean, enhanced)
    stoi_val = compute_stoi(clean, enhanced, sample_rate)
    pesq_val = compute_pesq(clean, enhanced, sample_rate)

    return {
        "snr_in_db": true_snr["snr_in"],
        "snr_out_db": true_snr["snr_out"],
        "delta_snr_db": true_snr["delta_snr"],
        "si_sdr_db": si_sdr,
        "stoi": stoi_val,
        "pesq": pesq_val,
    }


class LiveNoiseFloorTracker:
    """
    Real-time noise-floor tracker and estimated SNR estimator for live operation.
    
    Since clean speech s[n] is mathematically unavailable in real-time operational environments,
    this tracker monitors incoming energy and tracks the noise floor during non-speech/VAD-negative
    frames (p_vad < vad_threshold) via recursive exponential smoothing.
    
    All computed live SNR values are explicitly identified as 'estimated'.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_threshold: float = 0.30,
        noise_alpha: float = 0.95,
        signal_alpha: float = 0.85,
        eps: float = 1e-10
    ):
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.noise_alpha = noise_alpha
        self.signal_alpha = signal_alpha
        self.eps = eps

        self.reset()

    def reset(self):
        """Reset internal power accumulators to baseline acoustic state."""
        # Noise floor variance estimates (starts at conservative quiet baseline)
        self.p_noise_in: float = 1e-4
        self.p_noise_out: float = 1e-4

        # Smoothed signal power estimates
        self.p_sig_in: float = 1e-4
        self.p_sig_out: float = 1e-4

        # Latest levels in dBFS
        self.primary_dbfs: float = -96.0
        self.reference_dbfs: float = -96.0
        self.output_dbfs: float = -96.0

        # Latest estimated SNR (dB)
        self.estimated_input_snr_db: float = 0.0
        self.estimated_output_snr_db: float = 0.0
        self.estimated_snr_improvement_db: float = 0.0
        self.frames_tracked: int = 0
        self.is_estimated: bool = True

    def update(
        self,
        primary_samples: np.ndarray,
        output_samples: np.ndarray,
        vad_prob: float,
        reference_samples: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Process a new audio hop/frame and update live noise-floor and SNR estimates.
        """
        self.frames_tracked += 1

        # 1. Compute frame mean-square powers
        p_frame_in = float(np.mean(primary_samples**2)) if len(primary_samples) > 0 else 0.0
        p_frame_out = float(np.mean(output_samples**2)) if len(output_samples) > 0 else 0.0
        p_frame_ref = (float(np.mean(reference_samples**2)) if (reference_samples is not None and len(reference_samples) > 0) else 0.0)

        # 2. Compute real-time dBFS levels: 20 * log10(RMS)
        rms_in = np.sqrt(max(p_frame_in, self.eps))
        rms_out = np.sqrt(max(p_frame_out, self.eps))
        rms_ref = np.sqrt(max(p_frame_ref, self.eps))

        self.primary_dbfs = float(np.clip(20.0 * np.log10(rms_in), -96.0, 0.0))
        self.output_dbfs = float(np.clip(20.0 * np.log10(rms_out), -96.0, 0.0))
        self.reference_dbfs = float(np.clip(20.0 * np.log10(rms_ref), -96.0, 0.0))

        # 3. Update overall smoothed signal power
        self.p_sig_in = self.signal_alpha * self.p_sig_in + (1.0 - self.signal_alpha) * p_frame_in
        self.p_sig_out = self.signal_alpha * self.p_sig_out + (1.0 - self.signal_alpha) * p_frame_out

        # 4. Noise-floor tracking during VAD-negative frames (non-speech)
        # When speech is absent (vad_prob < threshold), the energy is primarily acoustic background noise
        if vad_prob < self.vad_threshold:
            self.p_noise_in = self.noise_alpha * self.p_noise_in + (1.0 - self.noise_alpha) * p_frame_in
            self.p_noise_out = self.noise_alpha * self.p_noise_out + (1.0 - self.noise_alpha) * p_frame_out
        else:
            # Slow minimum tracking during active speech to avoid upward noise tracking drift
            if p_frame_in < self.p_noise_in:
                self.p_noise_in = 0.90 * self.p_noise_in + 0.10 * p_frame_in
            if p_frame_out < self.p_noise_out:
                self.p_noise_out = 0.90 * self.p_noise_out + 0.10 * p_frame_out

        # Ensure noise floors have valid positive floor
        self.p_noise_in = max(self.p_noise_in, self.eps)
        self.p_noise_out = max(self.p_noise_out, self.eps)

        # 5. Compute estimated SNRs
        # Speech signal power estimate: S = max(P_total - P_noise, eps)
        s_in_est = max(self.p_sig_in - self.p_noise_in, 1e-6 * self.p_noise_in)
        s_out_est = max(self.p_sig_out - self.p_noise_out, 1e-6 * self.p_noise_out)

        snr_in = 10.0 * np.log10(s_in_est / self.p_noise_in)
        snr_out = 10.0 * np.log10(s_out_est / self.p_noise_out)

        self.estimated_input_snr_db = float(np.clip(snr_in, -20.0, 45.0))
        self.estimated_output_snr_db = float(np.clip(snr_out, -20.0, 45.0))
        self.estimated_snr_improvement_db = float(np.clip(self.estimated_output_snr_db - self.estimated_input_snr_db, -10.0, 35.0))

        return {
            "estimated_input_snr_db": self.estimated_input_snr_db,
            "estimated_output_snr_db": self.estimated_output_snr_db,
            "estimated_snr_improvement_db": self.estimated_snr_improvement_db,
            "primary_level_dbfs": self.primary_dbfs,
            "reference_level_dbfs": self.reference_dbfs,
            "output_level_dbfs": self.output_dbfs,
            "snr_is_estimated": True
        }
