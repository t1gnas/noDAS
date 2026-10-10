import json

import numpy as np

from das.config import Config, model_path
from das.evaluate import REFERENCE, REFERENCE_F1, REFERENCE_RECALL, evaluate
from das.loro import REFERENCE_RECALL as LORO_REFERENCE
from das.loro import run_all
from das.predict import MODELS, class_names

RESULTS_FILE = "results.json"


def plain(v: object) -> object:
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    return int(v) if isinstance(v, np.integer) else v


def model_summary(s: dict) -> dict:
    keep = ("tvb", "events", "event_rows", "false_alarms", "bg_km_days", "fa_per_km_day", "threshold")
    return {"cells": s["cells"], "coarse": s["coarse"], **{k: s[k] for k in keep}}


def loro_summary(results: list[dict]) -> list[dict]:
    return [{"id": r["id"], "class": r["class"], "recall": r["recall"], "event_pod": r["event_pod"], "dist": r["dist"]} for r in results]


def references() -> dict:
    return {"test": REFERENCE, "f1": REFERENCE_F1, "recall": REFERENCE_RECALL, "loro_recall": LORO_REFERENCE}


def run(cfg: Config) -> None:
    out = {"classes": class_names(cfg), "test": {m: model_summary(evaluate(m, cfg)) for m in MODELS}}
    out["loro"] = {"cnn": loro_summary(run_all("cnn", cfg))}
    out["reference"] = references()
    model_path(cfg, RESULTS_FILE).write_text(json.dumps(plain(out), indent=1))
    print(f"wrote {model_path(cfg, RESULTS_FILE)}")
