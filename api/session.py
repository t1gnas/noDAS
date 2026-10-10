import asyncio
import time
from collections import Counter
from collections.abc import Iterator

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from api.catalog import UNVERIFIED_ID, find_scenario
from api.view import b64, intensity, overlay, positions, truth
from das.data import UNKNOWN, row_seconds
from das.stream import alert_event, new_state, steps

SPEEDS = (1.0, 2.0, 5.0, 10.0)


def meta(ctx: dict, sc: dict, speed: float) -> dict:
    p = ctx["prepared"][sc["id"]]
    rec, view, cfg = p["rec"], p["view"], ctx["cfg"]
    rs = row_seconds(rec["attrs"])
    return {"type": "meta", "scenario": sc["id"], "title": sc["title"], "width": view["width"], "hop_rows": cfg.alert["hop_rows"], "row_seconds": rs,
            "duration_s": rec["targets"].shape[0] * rs, "speed": speed, "classes": ctx["classes"], **positions(rec, view)}


def annotated_class(t: np.ndarray, probs: np.ndarray) -> tuple[bool, int | None, int | None]:
    ann = (t != 0) & (t != UNKNOWN)
    if not ann.any():
        return False, None, None
    mean = np.nan_to_num(probs[ann], nan=0.0).mean(0)
    return True, int(np.bincount(t[ann]).argmax()), int(np.argmax(mean[1:]) + 1)


def verify_track(track: dict, rec: dict, end: int, state: dict) -> tuple[bool, int | None, int | None]:
    cols = slice(track["c0"], track["c1"] + 1)
    probs = np.concatenate(state["probs"])[:, cols]
    k = min(probs.shape[0], end)
    return annotated_class(rec["targets"][end - k:end, cols], probs[-k:])


def refresh(state: dict, ver: dict, rec: dict, end: int) -> list[dict]:
    upgraded = []
    for tr in state["open"]:
        was = ver.get(tr["id"])
        if was is not None and was[0]:
            continue
        ver[tr["id"]] = verify_track(tr, rec, end, state)
        if was is not None and ver[tr["id"]][0]:
            upgraded.append(tr)
    return upgraded


def payload(e: dict, ver: dict, classes: list[dict]) -> dict:
    verified, truth_id, model_id = ver.get(e["id"], (False, None, None))
    style = classes[model_id] if verified else classes[UNVERIFIED_ID]
    keys = ("kind", "id", "t", "pos_m", "c0", "c1", "confidence", "duration_s")
    return {**{k: e[k] for k in keys}, "verified": verified, "label": style["label"], "color": style["color"],
            "model_label": classes[model_id if verified else e["class_id"]]["label"], "truth_label": classes[truth_id]["label"] if truth_id else None}


def frame_rows(s: dict, p: dict) -> dict:
    rec, view, r1 = p["rec"], p["view"], s["end"]
    r0 = r1 - s["cls"].shape[0]
    return {"rows": int(r1 - r0), "intensity": b64(intensity(rec, r0, r1, view)), "overlay": b64(overlay(s["cls"], rec["targets"][r0:r1], view)),
            "truth": b64(truth(rec["targets"][r0:r1], view))}


def build_frame(s: dict, ctx: dict, p: dict, state: dict, ver: dict) -> dict:
    upgraded = refresh(state, ver, p["rec"], s["end"]) if s["cls"] is not None else []
    events = s["events"] + [alert_event("update", tr, s["t"], ctx["names"]) for tr in upgraded]
    body = frame_rows(s, p) if s["cls"] is not None else {"rows": 0}
    return {"type": "frame", "t": s["t"], "end": s["end"], **body, "events": [payload(e, ver, ctx["classes"]) for e in events]}


def summary(state: dict, ver: dict, classes: list[dict]) -> dict:
    verified = [ver[a["id"]] for a in state["alerts"] if ver.get(a["id"], (False,))[0]]
    by_class = Counter(classes[v[2]]["label"] for v in verified)
    correct = sum(v[1] == v[2] for v in verified)
    return {"type": "end", "alerts": len(state["alerts"]), "verified": len(verified), "unverified": len(state["alerts"]) - len(verified),
            "correct": correct, "by_class": dict(by_class)}


async def read_controls(ws: WebSocket, control: dict) -> None:
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("action") in ("pause", "resume"):
                control["paused"] = msg["action"] == "pause"
            if float(msg.get("speed", 0)) in SPEEDS:
                control["speed"] = float(msg["speed"])
    except (WebSocketDisconnect, RuntimeError):
        control["closed"] = True


async def wait_turn(control: dict, hop_s: float, started: float) -> None:
    while control["paused"] and not control["closed"]:
        await asyncio.sleep(0.05)
    delay = hop_s / control["speed"] - (time.perf_counter() - started)
    if delay > 0:
        await asyncio.sleep(delay)


async def pump(ws: WebSocket, ctx: dict, sc: dict, control: dict) -> dict:
    p, state, ver = ctx["prepared"][sc["id"]], new_state(), {}
    hop_s = ctx["cfg"].alert["hop_rows"] * row_seconds(p["rec"]["attrs"])
    gen: Iterator[dict] = steps(ctx["fn"], p["rec"], ctx["cfg"], ctx["names"], state)
    while not control["closed"]:
        started = time.perf_counter()
        s = await asyncio.to_thread(next, gen, None)
        if s is None:
            break
        await ws.send_json(build_frame(s, ctx, p, state, ver))
        await wait_turn(control, hop_s, started)
    return summary(state, ver, ctx["classes"])


async def run_live(ws: WebSocket, ctx: dict, scenario_id: str, speed: float) -> None:
    sc = find_scenario(scenario_id)
    control = {"paused": False, "closed": False, "speed": speed if speed in SPEEDS else 5.0}
    await ws.send_json(meta(ctx, sc, control["speed"]))
    reader = asyncio.create_task(read_controls(ws, control))
    try:
        result = await pump(ws, ctx, sc, control)
        if not control["closed"]:
            await ws.send_json(result)
    finally:
        reader.cancel()
