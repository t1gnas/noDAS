import csv
import json
from pathlib import Path

import h5py
import numpy as np

SPLITS = ("train", "val", "test")
SAMPLE_RATE_HZ = 20000
STRIDE_SAMPLES = 2048
SUPPORT_SAMPLES = 8192
N_BANDS = 6
N_CLASSES = 9
UNKNOWN = 255
INT_COLUMNS = ("class_id", "time_row", "channel", "height", "width", "labeled_cells")


def load_metadata(root: Path) -> dict:
    return json.loads((Path(root) / "metadata.json").read_text())


def parse_patch(row: dict) -> dict:
    return {**row, **{k: int(row[k]) for k in INT_COLUMNS}}


def load_patches(root: Path) -> list[dict]:
    with (Path(root) / "patches.csv").open(newline="") as f:
        return [parse_patch(r) for r in csv.DictReader(f)]


def list_recordings(root: Path) -> list[str]:
    return sorted(p.stem for p in Path(root).glob("*.h5"))


def norm_stats(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(meta["normalization"]["mean"], np.float32)
    std = np.asarray(meta["normalization"]["std"], np.float32)
    assert mean.shape == (N_BANDS,) and std.shape == (N_BANDS,), f"normalisation needs {N_BANDS} values, got mean {mean.shape} std {std.shape}"
    assert (std > 0).all(), f"normalisation std must be positive, got {std}"
    return mean, std


def plain_attr(v: object) -> object:
    v = v.decode() if isinstance(v, bytes) else v
    return v.item() if isinstance(v, np.generic) else v


def read_h5(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        assert "features" in f and "targets" in f, f"{path} needs datasets features and targets, has {list(f)}"
        attrs = {k: plain_attr(v) for k, v in f.attrs.items()}
        return f["features"][:].astype(np.float32), f["targets"][:].astype(np.uint8), attrs


def coverage_mask(features: np.ndarray) -> np.ndarray:
    return np.isfinite(features).all(axis=-1)


def load_file(path: Path, mean: np.ndarray, std: np.ndarray) -> dict:
    features, targets, attrs = read_h5(Path(path))
    assert features.ndim == 3 and features.shape[2] == N_BANDS, f"{path}: features must be [T,C,{N_BANDS}], got {features.shape}"
    assert targets.shape == features.shape[:2], f"{path}: targets {targets.shape} != features {features.shape[:2]}"
    norm = ((features - mean) / std).astype(np.float32)
    return {"id": Path(path).stem, "features": norm, "targets": targets, "attrs": attrs, "coverage": coverage_mask(features), "row0": 0}


def load_recording(root: Path, rec_id: str, mean: np.ndarray, std: np.ndarray) -> dict:
    return load_file(Path(root) / f"{rec_id}.h5", mean, std)


def load_all(root: Path) -> dict[str, dict]:
    mean, std = norm_stats(load_metadata(root))
    ids = list_recordings(root)
    assert ids, f"no .h5 recordings found in {root}"
    return {i: load_recording(root, i, mean, std) for i in ids}


def crop_rows(rec: dict, r0: int, r1: int) -> dict:
    r0, r1 = max(0, r0), min(rec["targets"].shape[0], r1)
    assert r1 > r0, f"empty row range [{r0},{r1}) for {rec['id']}"
    keys = ("features", "targets", "coverage")
    return {**rec, **{k: rec[k][r0:r1] for k in keys}, "row0": rec["row0"] + r0}


def row_seconds(attrs: dict) -> float:
    return attrs["stride_samples"] / attrs["sample_rate_hz"]


def seconds_to_row(t: float | None, attrs: dict, default: int) -> int:
    return default if t is None else int(round(t / row_seconds(attrs)))


def split_ranges(meta: dict) -> dict[str, dict]:
    return {r["recording_id"]: r["split_ranges"] for r in meta["recordings"]}


def check_attrs(rec: dict) -> None:
    a = rec["attrs"]
    expect = {"sample_rate_hz": SAMPLE_RATE_HZ, "stride_samples": STRIDE_SAMPLES, "support_samples": SUPPORT_SAMPLES}
    for k, v in expect.items():
        assert k in a and float(a[k]) == v, f"{rec['id']}: attr {k} expected {v}, got {a.get(k)}"
    for k in ("spatial_spacing_m", "start_locus_index"):
        assert k in a, f"{rec['id']}: missing attr {k}, has {sorted(a)}"


def check_labels(rec: dict) -> None:
    vals = np.unique(rec["targets"])
    bad = [int(v) for v in vals if v >= N_CLASSES and v != UNKNOWN]
    assert not bad, f"{rec['id']}: target values {bad} outside 0..{N_CLASSES - 1} and {UNKNOWN}"


def patch_region(rec: dict, p: dict, key: str) -> np.ndarray:
    return rec[key][p["time_row"]:p["time_row"] + p["height"], p["channel"]:p["channel"] + p["width"]]


def check_patch_bounds(rec: dict, p: dict, shape: list[int]) -> None:
    T, C = rec["targets"].shape
    assert [p["height"], p["width"]] == list(shape[1:]), f"patch size {(p['height'], p['width'])} != metadata patch_shape {shape[1:]}"
    assert 0 <= p["time_row"] and p["time_row"] + p["height"] <= T, f"{rec['id']}: patch rows {p['time_row']}+{p['height']} outside T={T}"
    assert 0 <= p["channel"] and p["channel"] + p["width"] <= C, f"{rec['id']}: patch channels {p['channel']}+{p['width']} outside C={C}"


def check_patch_content(rec: dict, p: dict, ranges: dict) -> None:
    assert p["split"] in SPLITS, f"patch split {p['split']!r} not in {SPLITS}"
    lo, hi = ranges[p["split"]]
    assert lo <= p["time_row"] and p["time_row"] + p["height"] <= hi, f"{rec['id']}: patch rows {p['time_row']}+{p['height']} cross {p['split']} range [{lo},{hi})"
    assert patch_region(rec, p, "coverage").all(), f"{rec['id']}: patch at row {p['time_row']} ch {p['channel']} leaves coverage"
    n = int((patch_region(rec, p, "targets") != UNKNOWN).sum())
    assert n == p["labeled_cells"], f"{rec['id']}: patch labeled_cells {p['labeled_cells']} != counted {n}"
    assert 0 <= p["class_id"] < N_CLASSES, f"patch class_id {p['class_id']} outside 0..{N_CLASSES - 1}"


def assert_consistent(recs: dict[str, dict], patches: list[dict], meta: dict) -> None:
    norm_stats(meta)
    assert int(meta["ignore_index"]) == UNKNOWN, f"ignore_index {meta['ignore_index']} != {UNKNOWN}"
    assert len(meta["classes"]) == N_CLASSES, f"metadata lists {len(meta['classes'])} classes, expected {N_CLASSES}"
    for rec in recs.values():
        check_attrs(rec)
        check_labels(rec)
    ranges = split_ranges(meta)
    for p in patches:
        assert p["recording_id"] in recs, f"patch references unknown recording {p['recording_id']}"
        check_patch_bounds(recs[p["recording_id"]], p, meta["patch_shape"])
        check_patch_content(recs[p["recording_id"]], p, ranges[p["recording_id"]])
