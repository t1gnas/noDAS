import numpy as np
from scipy.ndimage import uniform_filter1d

from das.data import UNKNOWN


def argmax_with_threshold(probs: np.ndarray, thr: float, background: int = 0) -> np.ndarray:
    missing = np.isnan(probs[..., 0])
    p = np.nan_to_num(probs, nan=0.0)
    cls, conf = p.argmax(-1), p.max(-1)
    cls = np.where((cls != background) & (conf < thr), background, cls).astype(np.uint8)
    cls[missing] = UNKNOWN
    return cls


def smooth_time(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return probs
    missing = np.isnan(probs)
    out = uniform_filter1d(np.nan_to_num(probs, nan=0.0), size=k, axis=0, mode="nearest")
    out[missing] = np.nan
    return out


def confirm(history: list[np.ndarray], k: int, n: int) -> np.ndarray:
    recent = history[-n:]
    return np.sum(recent, axis=0) >= k


def to_position_m(c: float, attrs: dict) -> float:
    return float((c + attrs["start_locus_index"]) * attrs["spatial_spacing_m"])


def activity(cls: np.ndarray, background: int = 0) -> np.ndarray:
    return (cls != background) & (cls != UNKNOWN)
