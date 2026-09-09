import numpy as np
from typing import Optional, Tuple

def extract_impulse_features(
    time_frame: np.ndarray,
    spectral_magnitude: np.ndarray,
    prev_magnitude: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the 8-element feature vector matching C++ ImpulseDetector:
    1. RMS: sqrt((1/N) * sum(x^2))
    2. Spectral Flux: sum_k (max(0, |X(k, m)| - |X(k, m-1)|))^2
    3. Crest Factor: max|x| / (RMS + 1e-7)
    4. ZCR: (1/(N-1)) * sum |sign(x[n]) - sign(x[n-1])| / 2
    5-8. Subband Energies: average |X(k)|^2 across 4 frequency bands
         - Band 0 (Low): 0 - 500 Hz (bins 0..16)
         - Band 1 (Mid-Low): 500 - 2000 Hz (bins 16..64)
         - Band 2 (Mid-High): 2000 - 4000 Hz (bins 64..128)
         - Band 3 (High): 4000 - 8000 Hz (bins 128..256)

    Returns:
        features: np.ndarray of shape (8,) and dtype float32
        new_magnitude: np.ndarray of shape (257,) to persist for next frame
    """
    time_frame = np.asarray(time_frame, dtype=np.float32)
    spectral_magnitude = np.asarray(spectral_magnitude, dtype=np.float32)
    N = len(time_frame)

    if prev_magnitude is None:
        prev_magnitude = np.zeros_like(spectral_magnitude)

    # 1. RMS
    rms = float(np.sqrt(np.mean(time_frame ** 2)))

    # 2. Crest Factor
    max_abs = float(np.max(np.abs(time_frame))) if N > 0 else 0.0
    crest_factor = max_abs / (rms + 1e-7)

    # 3. Zero Crossing Rate
    if N > 1:
        signs = (time_frame >= 0.0).astype(np.int32)
        zcr = float(np.sum(np.abs(np.diff(signs))) / (N - 1))
    else:
        zcr = 0.0

    # 4. Spectral Flux (Half-Wave Rectified)
    diff = spectral_magnitude - prev_magnitude
    pos_diff = np.maximum(0.0, diff)
    spectral_flux = float(np.sum(pos_diff ** 2))

    # 5. Subband Energies
    band_splits = [0, 16, 64, 128, 256]
    band_energies = []
    for b in range(4):
        start = band_splits[b]
        end = min(band_splits[b + 1], len(spectral_magnitude))
        if end > start:
            band_slice = spectral_magnitude[start:end]
            band_energy = float(np.mean(band_slice ** 2))
        else:
            band_energy = 0.0
        band_energies.append(band_energy)

    features = np.array([
        rms,
        spectral_flux,
        crest_factor,
        zcr,
        band_energies[0],
        band_energies[1],
        band_energies[2],
        band_energies[3]
    ], dtype=np.float32)

    return features, spectral_magnitude.copy()
