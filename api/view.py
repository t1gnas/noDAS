import base64
import math

import numpy as np

from api.catalog import UNVERIFIED_ID
from das.data import UNKNOWN
from das.postproc import activity, to_position_m

MAX_WIDTH = 480
MARGIN_CHANNELS = 24
LOW, HIGH = -1.5, 3.5


def span(mask_cols: np.ndarray, what: str) -> tuple[int, int]:
    cols = np.where(mask_cols)[0]
    assert cols.size, f"recording has no {what} channels"
    return int(cols.min()), int(cols.max())


def view_window(rec: dict) -> dict:
    lo, hi = span((rec["targets"] != UNKNOWN).any(0), "labelled")
    clo, chi = span(rec["coverage"].any(0), "covered")
    c0, c1 = max(lo - MARGIN_CHANNELS, clo), min(hi + MARGIN_CHANNELS, chi) + 1
    f = math.ceil((c1 - c0) / MAX_WIDTH)
    return {"c0": c0, "c1": c1, "factor": f, "width": math.ceil((c1 - c0) / f)}


def restrict(rec: dict, view: dict) -> dict:
    cov = np.zeros_like(rec["coverage"])
    cov[:, view["c0"]:view["c1"]] = rec["coverage"][:, view["c0"]:view["c1"]]
    return {**rec, "coverage": cov}


def pool_max(a: np.ndarray, f: int) -> np.ndarray:
    pad = (-a.shape[1]) % f
    a = np.pad(a, ((0, 0), (0, pad))) if pad else a
    return a.reshape(a.shape[0], -1, f).max(-1)


def intensity(rec: dict, r0: int, r1: int, view: dict) -> np.ndarray:
    x = np.nan_to_num(rec["features"][r0:r1, view["c0"]:view["c1"], :3], nan=LOW).mean(-1)
    scaled = np.clip((x - LOW) / (HIGH - LOW), 0.0, 1.0) ** 0.8
    return pool_max((scaled * 255).astype(np.uint8), view["factor"])


def overlay(cls: np.ndarray, targets: np.ndarray, view: dict) -> np.ndarray:
    c, t = cls[:, view["c0"]:view["c1"]], targets[:, view["c0"]:view["c1"]]
    annotated = (t != 0) & (t != UNKNOWN)
    codes = np.where(activity(c) & annotated, c, np.where(activity(c), UNVERIFIED_ID, 0))
    return pool_max(codes.astype(np.uint8), view["factor"])


def truth(targets: np.ndarray, view: dict) -> np.ndarray:
    t = targets[:, view["c0"]:view["c1"]]
    return pool_max(np.where(t == UNKNOWN, 0, t).astype(np.uint8), view["factor"])


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def positions(rec: dict, view: dict) -> dict:
    a, n = rec["attrs"], rec["coverage"].shape[1]
    return {"view_start_m": to_position_m(view["c0"], a), "view_end_m": to_position_m(view["c1"] - 1, a), "fiber_start_m": to_position_m(0, a), "fiber_end_m": to_position_m(n - 1, a)}
