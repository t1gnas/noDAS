import numpy as np
from scipy.ndimage import binary_closing, find_objects, label

from das.data import N_CLASSES, row_seconds
from das.postproc import activity, argmax_with_threshold, smooth_time, to_position_m

CONNECT8 = np.ones((3, 3), bool)


def close_channels(mask: np.ndarray, gap: int) -> np.ndarray:
    if gap <= 0:
        return mask
    return mask | binary_closing(mask, structure=np.ones((1, gap + 1), bool))


def components(mask: np.ndarray, merge_gap_channels: int) -> tuple[np.ndarray, int]:
    labels, n = label(close_channels(mask, merge_gap_channels), structure=CONNECT8)
    return np.where(mask, labels, 0), n


def region_cells(labels: np.ndarray, k: int, sl: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    r, c = np.nonzero(labels[sl] == k)
    return r + sl[0].start, c + sl[1].start


def event_class(cell_probs: np.ndarray) -> tuple[int, float]:
    mean = np.nanmean(cell_probs, axis=0)
    k = int(np.argmax(mean[1:]) + 1)
    return k, float(mean[k])


def event_times(r: np.ndarray, attrs: dict, row0: int) -> tuple[float, float]:
    end = ((row0 + r.max()) * attrs["stride_samples"] + attrs["support_samples"]) / attrs["sample_rate_hz"]
    return float((row0 + r.min()) * row_seconds(attrs)), float(end)


def make_event(k: int, r: np.ndarray, c: np.ndarray, probs: np.ndarray, attrs: dict, row0: int, names: list[str]) -> dict:
    cls, conf = event_class(probs[r, c])
    t0, t1 = event_times(r, attrs, row0)
    c0, c1 = int(c.min()), int(c.max())
    return {"label": k, "t_start_s": t0, "t_end_s": t1, "c0": c0, "c1": c1, "pos_start_m": to_position_m(c0, attrs), "pos_end_m": to_position_m(c1, attrs),
            "class_id": cls, "class_name": names[cls], "confidence": conf, "cells": int(r.size), "centroid_c": float(c.mean())}


def event_table(labels: np.ndarray, probs: np.ndarray, attrs: dict, row0: int, min_cells: int, names: list[str]) -> list[dict]:
    events = []
    for k, sl in enumerate(find_objects(labels), start=1):
        if sl is None:
            continue
        r, c = region_cells(labels, k, sl)
        if r.size >= min_cells:
            events.append(make_event(k, r, c, probs, attrs, row0, names))
    return sorted(events, key=lambda e: (e["t_start_s"], e["c0"]))


def detect(probs: np.ndarray, rec: dict, cfg_alert: dict, names: list[str], thr: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    probs = smooth_time(probs, cfg_alert.get("smooth_rows", 1))
    cls = argmax_with_threshold(probs, thr)
    labels, _ = components(activity(cls), cfg_alert["merge_gap_channels"])
    return cls, labels, event_table(labels, probs, rec["attrs"], rec["row0"], cfg_alert["min_cells"], names)


def gt_events(targets: np.ndarray, min_cells: int) -> list[dict]:
    out = []
    for k in range(1, N_CLASSES):
        labels, _ = label(targets == k, structure=CONNECT8)
        for j, sl in enumerate(find_objects(labels), start=1):
            r, c = region_cells(labels, j, sl)
            if r.size >= min_cells:
                out.append({"class_id": k, "rows": r, "chs": c})
    return out


def match_one(g: dict, cls: np.ndarray, pred_labels: np.ndarray, hit_fraction: float) -> dict:
    pred = cls[g["rows"], g["chs"]]
    act = activity(pred)
    detected = bool(act.mean() >= hit_fraction)
    mode = int(np.bincount(pred[act], minlength=N_CLASSES).argmax()) if act.any() else 0
    ids = np.setdiff1d(np.unique(pred_labels[g["rows"], g["chs"]]), [0])
    return {"class_id": g["class_id"], "detected": detected, "correct": detected and mode == g["class_id"], "ids": ids, "gt_c": float(g["chs"].mean())}


def localisation_m(m: dict, pred_labels: np.ndarray, attrs: dict) -> float:
    if not m["detected"] or m["ids"].size == 0:
        return float("nan")
    _, c = np.nonzero(np.isin(pred_labels, m["ids"]))
    return float(abs(c.mean() - m["gt_c"]) * attrs["spatial_spacing_m"])


def event_metrics(gt: list[dict], cls: np.ndarray, pred_labels: np.ndarray, attrs: dict, hit_fraction: float) -> list[dict]:
    matches = [match_one(g, cls, pred_labels, hit_fraction) for g in gt]
    return [{**m, "loc_err_m": localisation_m(m, pred_labels, attrs)} for m in matches]


def summarise_matches(matches: list[dict]) -> dict:
    det = np.array([m["detected"] for m in matches], bool)
    cor = np.array([m["correct"] for m in matches], bool)
    loc = np.array([m["loc_err_m"] for m in matches], float)
    pod = float(det.mean()) if det.size else float("nan")
    acc = float(cor[det].mean()) if det.any() else float("nan")
    med = float(np.nanmedian(loc)) if np.isfinite(loc).any() else float("nan")
    return {"n_gt": int(det.size), "pod": pod, "class_acc": acc, "loc_err_m": med}


def false_alarms(events: list[dict], pred_labels: np.ndarray, bg_mask: np.ndarray) -> int:
    return sum(bool(bg_mask[pred_labels == e["label"]].mean() > 0.5) for e in events)


def km_days(bg_cells: int, attrs: dict) -> float:
    return bg_cells * row_seconds(attrs) * attrs["spatial_spacing_m"] / 1000.0 / 86400.0


def alarms_per_km_day(n_alarms: int, bg_cells: int, attrs: dict) -> float:
    kd = km_days(bg_cells, attrs)
    return n_alarms / kd if kd > 0 else float("nan")
