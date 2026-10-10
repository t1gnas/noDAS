import numpy as np
from scipy.ndimage import uniform_filter

from das.data import N_BANDS


def normalised_grid(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.nan_to_num(rec["features"], nan=0.0).astype(np.float32), rec["coverage"]


def context_mean(x: np.ndarray, rows: int, ch: int) -> np.ndarray:
    return uniform_filter(x, size=(rows, ch, 1), mode="nearest")


def context_std(x: np.ndarray, rows: int, ch: int) -> np.ndarray:
    m = context_mean(x, rows, ch)
    return np.sqrt(np.maximum(context_mean(x * x, rows, ch) - m * m, 0.0))


def cell_features(grid: np.ndarray, rows: int, ch: int) -> np.ndarray:
    parts = [grid, context_mean(grid, rows, ch), context_std(grid, rows, ch)]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def feature_names() -> list[str]:
    return [f"{p}{i}" for p in ("band", "ctxmean", "ctxstd") for i in range(N_BANDS)]


def rec_features(rec: dict, cfg_cell: dict) -> np.ndarray:
    grid, _ = normalised_grid(rec)
    return cell_features(grid, cfg_cell["context_rows"], cfg_cell["context_channels"])


def gather(rec: dict, mask: np.ndarray, cfg_cell: dict) -> tuple[np.ndarray, np.ndarray]:
    assert mask.shape == rec["targets"].shape, f"mask {mask.shape} != targets {rec['targets'].shape}"
    return rec_features(rec, cfg_cell)[mask], rec["targets"][mask].astype(np.int64)
