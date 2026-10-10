import numpy as np
from scipy.ndimage import find_objects, label

from das.data import UNKNOWN
from das.events import close_channels
from das.postproc import to_position_m

MERGE_GAP_CHANNELS = 4
MIN_WIDTH_CHANNELS = 2
MAX_JUMP_M = 25.0
DETECT_MARGIN_M = 10.0


def observe(row: np.ndarray, rec: dict, view: dict) -> list[dict]:
    out = []
    seg = row[view["c0"]:view["c1"]]
    for k in np.unique(seg[(seg != 0) & (seg != UNKNOWN)]):
        labels, _ = label(close_channels((seg == k)[None], MERGE_GAP_CHANNELS)[0])
        for sl in find_objects(labels):
            if sl[0].stop - sl[0].start >= MIN_WIDTH_CHANNELS:
                c = view["c0"] + (sl[0].start + sl[0].stop - 1) / 2
                out.append({"class_id": int(k), "pos_m": to_position_m(c, rec["attrs"])})
    return out


def nearest(prev: list[dict], o: dict, taken: set[int]) -> dict | None:
    cands = [p for p in prev if p["class_id"] == o["class_id"] and p["id"] not in taken and abs(p["pos_m"] - o["pos_m"]) <= MAX_JUMP_M]
    return min(cands, key=lambda p: abs(p["pos_m"] - o["pos_m"])) if cands else None


def track(state: dict, observed: list[dict]) -> list[dict]:
    taken: set[int] = set()
    out = []
    for o in sorted(observed, key=lambda x: x["pos_m"]):
        match = nearest(state["actors"], o, taken)
        aid = match["id"] if match else state["next_actor"]
        state["next_actor"] += 0 if match else 1
        taken.add(aid)
        out.append({**o, "id": aid})
    state["actors"] = out
    return out


def covers(t: dict, pos_m: float, attrs: dict) -> bool:
    lo, hi = to_position_m(t["c0"], attrs) - DETECT_MARGIN_M, to_position_m(t["c1"], attrs) + DETECT_MARGIN_M
    return lo <= pos_m <= hi


def detection(actor: dict, open_tracks: list[dict], attrs: dict) -> dict | None:
    near = [t for t in open_tracks if covers(t, actor["pos_m"], attrs)]
    return min(near, key=lambda t: abs(t["pos_m"] - actor["pos_m"])) if near else None


def actor_payload(a: dict, open_tracks: list[dict], attrs: dict, classes: list[dict]) -> dict:
    hit = detection(a, open_tracks, attrs)
    style = classes[a["class_id"]]
    base = {"id": a["id"], "pos_m": a["pos_m"], "label": style["label"], "color": style["color"], "detected": hit is not None}
    if hit is None:
        return base
    return {**base, "alert_id": hit["id"], "alert_pos_m": hit["pos_m"], "confidence": hit["confidence"]}


def actors_frame(state: dict, row: np.ndarray, rec: dict, view: dict, classes: list[dict]) -> list[dict]:
    return [actor_payload(a, state["open"], rec["attrs"], classes) for a in track(state, observe(row, rec, view))]
