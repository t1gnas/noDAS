import numpy as np

from das.data import N_CLASSES, UNKNOWN, patch_region


def split_rows(rows: list[dict], split: str) -> list[dict]:
    return [r for r in rows if r["split"] == split]


def rows_for(rows: list[dict], rec_id: str) -> list[dict]:
    return [r for r in rows if r["recording_id"] == rec_id]


def patch_xy(rec: dict, row: dict) -> tuple[np.ndarray, np.ndarray]:
    x = patch_region(rec, row, "features")
    assert np.isfinite(x).all(), f"{rec['id']}: NaN inside indexed patch at row {row['time_row']} ch {row['channel']}"
    return x.transpose(2, 0, 1), patch_region(rec, row, "targets").astype(np.int64)


def patch_arrays(recs: dict[str, dict], rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    assert rows, "patch_arrays needs at least one patch row"
    pairs = [patch_xy(recs[r["recording_id"]], r) for r in rows]
    x = np.ascontiguousarray(np.stack([p[0] for p in pairs]), dtype=np.float32)
    y = np.stack([p[1] for p in pairs])
    return x, y, rows


def sampler_weights(rows: list[dict]) -> np.ndarray:
    bg = np.asarray([r["class_id"] == 0 for r in rows])
    if not bg.any() or bg.all():
        return np.full(len(rows), 1.0 / len(rows))
    return np.where(bg, 1.0 / bg.sum(), 1.0 / (~bg).sum()).astype(np.float64)


def cell_mask(rec: dict, rows: list[dict]) -> np.ndarray:
    mask = np.zeros(rec["targets"].shape, bool)
    for r in rows_for(rows, rec["id"]):
        mask[r["time_row"]:r["time_row"] + r["height"], r["channel"]:r["channel"] + r["width"]] = True
    return mask & (rec["targets"] != UNKNOWN) & rec["coverage"]


def labelled_mask(rec: dict) -> np.ndarray:
    return (rec["targets"] != UNKNOWN) & rec["coverage"]


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y[y != UNKNOWN].ravel(), minlength=N_CLASSES).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    return (w / w[counts > 0].mean()).astype(np.float32)
