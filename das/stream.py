import time

import numpy as np
from scipy.ndimage import find_objects, label

from das.config import Config
from das.data import UNKNOWN, crop_rows, row_seconds
from das.events import close_channels, gt_events
from das.postproc import activity, argmax_with_threshold, confirm, to_position_m
from das.predict import Predictor, class_names, crop_time, load_model, load_recording_for_inference
from das.report import print_table, print_title


def now_s(rec: dict, end: int) -> float:
    a = rec["attrs"]
    return ((rec["row0"] + end - 1) * a["stride_samples"] + a["support_samples"]) / a["sample_rate_hz"]


def hop_probs(fn: Predictor, rec: dict, end: int, buf_rows: int, hop: int) -> np.ndarray:
    return fn(crop_rows(rec, end - buf_rows, end))[-hop:]


def segment(sl: slice, act: np.ndarray, probs: np.ndarray, rec: dict) -> dict:
    a, p = act[:, sl], np.nan_to_num(probs[:, sl], nan=0.0)
    mean = p[a].mean(0) if a.any() else p.reshape(-1, p.shape[-1]).mean(0)
    k = int(np.argmax(mean[1:]) + 1)
    c = (sl.start + sl.stop - 1) / 2
    return {"c0": sl.start, "c1": sl.stop - 1, "pos_m": to_position_m(c, rec["attrs"]), "class_id": k, "confidence": float(mean[k]), "cells": int(a.sum())}


def segments(state: dict, cfg: Config, rec: dict) -> list[dict]:
    k, n = cfg.alert["confirm"]
    on = confirm([h.any(0) for h in state["acts"]], k, n)
    labels, _ = label(close_channels(on[None], cfg.alert["merge_gap_channels"])[0])
    act, probs = np.concatenate(state["acts"][-n:]), np.concatenate(state["probs"][-n:])
    segs = [segment(sl[0], act, probs, rec) for sl in find_objects(labels)]
    return [s for s in segs if s["cells"] >= cfg.alert["min_cells"]]


def nearest_track(state: dict, seg: dict, radius: float) -> dict | None:
    cands = [t for t in state["open"] if abs(t["pos_m"] - seg["pos_m"]) <= radius and not t["matched"]]
    return min(cands, key=lambda t: abs(t["pos_m"] - seg["pos_m"])) if cands else None


def open_track(state: dict, seg: dict, t: float, names: list[str]) -> None:
    track = {**seg, "id": len(state["alerts"]) + 1, "t_open": t, "t_last": t, "misses": 0, "matched": True, "votes": [seg["class_id"]]}
    state["open"].append(track)
    state["alerts"].append(track)
    print(f"[{t:8.2f}s] ALERT  #{track['id']:<3d} {names[seg['class_id']]:<13s} at {seg['pos_m']:7.1f} m (ch {seg['c0']}-{seg['c1']}) conf {seg['confidence']:.2f}")


def close_track(state: dict, track: dict, t: float, names: list[str]) -> None:
    track["class_id"] = int(np.bincount(track["votes"]).argmax())
    state["open"].remove(track)
    print(f"[{t:8.2f}s] CLEAR  #{track['id']:<3d} {names[track['class_id']]:<13s} at {track['pos_m']:7.1f} m after {track['t_last'] - track['t_open']:.1f} s")


def update_tracks(state: dict, segs: list[dict], t: float, cfg: Config, names: list[str]) -> None:
    for tr in state["open"]:
        tr["matched"] = False
    for s in segs:
        tr = nearest_track(state, s, cfg.alert["track_radius_m"])
        if tr is None:
            open_track(state, s, t, names)
            continue
        tr.update(pos_m=s["pos_m"], t_last=t, misses=0, matched=True, confidence=s["confidence"], c0=s["c0"], c1=s["c1"])
        tr["votes"].append(s["class_id"])
    for tr in [x for x in state["open"] if not x["matched"]]:
        tr["misses"] += 1
        if tr["misses"] >= cfg.alert["confirm"][1]:
            close_track(state, tr, t, names)


def step(state: dict, probs: np.ndarray, rec: dict, end: int, cfg: Config, names: list[str]) -> None:
    n = cfg.alert["confirm"][1]
    state["acts"] = (state["acts"] + [activity(argmax_with_threshold(probs, cfg.alert["threshold"]))])[-n:]
    state["probs"] = (state["probs"] + [probs])[-n:]
    update_tracks(state, segments(state, cfg, rec), now_s(rec, end), cfg, names)


def latency_rows(state: dict, rec: dict, cfg: Config, names: list[str]) -> list[list]:
    a, rows = rec["attrs"], []
    for g in gt_events(rec["targets"], cfg.eval["min_event_cells"]):
        onset = (rec["row0"] + g["rows"].min()) * a["stride_samples"] / a["sample_rate_hz"]
        lo, hi = to_position_m(g["chs"].min(), a) - cfg.alert["track_radius_m"], to_position_m(g["chs"].max(), a) + cfg.alert["track_radius_m"]
        hits = [t["t_open"] for t in state["alerts"] if lo <= t["pos_m"] <= hi and t["t_last"] >= onset]
        rows.append([names[g["class_id"]], f"{onset:.2f}", f"{to_position_m(g['chs'].mean(), a):.1f}", f"{min(hits) - onset:.2f}" if hits else "missed"])
    return rows


def summary(state: dict, rec: dict, cfg: Config, names: list[str]) -> None:
    print_title(f"stream summary: {len(state['alerts'])} alerts")
    counts = np.bincount([t["class_id"] for t in state["alerts"]], minlength=len(names))
    print_table([[names[i], int(c)] for i, c in enumerate(counts) if c], ["class", "alerts"])
    if (rec["targets"] != UNKNOWN).any():
        print_title("detection latency per ground-truth event (first alert time minus onset; negative = an alert was already open nearby)")
        print_table(latency_rows(state, rec, cfg, names), ["class", "onset_s", "pos_m", "latency_s"])


def replay(path: str, model_name: str, cfg: Config, speed: float, t0: float | None, t1: float | None) -> dict:
    fn, names = load_model(model_name, cfg), class_names(cfg)
    rec = crop_time(load_recording_for_inference(path, cfg), t0, t1)
    hop, buf = cfg.alert["hop_rows"], cfg.infer["tile_rows"]
    state = {"acts": [], "probs": [], "open": [], "alerts": []}
    print(f"streaming {rec['id']} with {model_name}, hop {hop} rows, speed x{speed}")
    for end in range(hop, rec["targets"].shape[0] + 1, hop):
        step(state, hop_probs(fn, rec, end, buf, hop), rec, end, cfg, names)
        time.sleep(hop * row_seconds(rec["attrs"]) / speed if speed > 0 else 0)
    for tr in list(state["open"]):
        close_track(state, tr, now_s(rec, rec["targets"].shape[0]), names)
    summary(state, rec, cfg, names)
    return state
