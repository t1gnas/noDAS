from collections import Counter
from functools import partial

import numpy as np

from das import cnn, lgbm
from das.config import Config, data_dir
from das.data import N_CLASSES, UNKNOWN, load_all, load_metadata, load_patches
from das.evaluate import eval_recording
from das.events import summarise_matches
from das.patches import labelled_mask
from das.predict import Predictor, class_names, cnn_predict, lgbm_predict
from das.scope import excluded_ids, scope_all, scoped
from das.report import print_limits, print_table, print_title

REFERENCE_RECALL = {"car": 0.64, "fence": 0.73, "walk": 0.50, "longboard": 0.50}


def true_class(rec: dict) -> int:
    t = rec["targets"]
    counts = np.bincount(t[(t != UNKNOWN) & (t != 0)].ravel(), minlength=N_CLASSES)
    assert counts.sum() > 0, f"{rec['id']} has no labelled activity cells"
    return int(counts.argmax())


def has_activity(rec: dict) -> bool:
    t = rec["targets"]
    return bool(((t != UNKNOWN) & (t != 0)).any()) and not (t == 0).any()


def folds(recs: dict[str, dict]) -> list[str]:
    cls = {i: true_class(r) for i, r in recs.items() if has_activity(r)}
    multi = {k for k, n in Counter(cls.values()).items() if n > 1}
    return sorted(i for i, k in cls.items() if k in multi)


def train_predictor(model_name: str, cfg: Config, recs: dict[str, dict], rows: list[dict]) -> Predictor:
    if model_name == "lgbm":
        return scoped(partial(lgbm_predict, lgbm.train(cfg, recs, rows, None), cfg), excluded_ids(cfg))
    net = cnn.train(cfg, recs, rows, None, cfg.cnn.get("epochs_loro", 15))
    return scoped(partial(cnn_predict, net, next(net.parameters()).device, cfg), excluded_ids(cfg))


def run_fold(model_name: str, hold_id: str, cfg: Config, recs: dict[str, dict], rows: list[dict]) -> dict:
    names = class_names(cfg)
    train_rows = [r for r in rows if r["recording_id"] != hold_id]
    assert len(train_rows) < len(rows), f"held-out recording {hold_id} has no patches"
    fn = train_predictor(model_name, cfg, recs, train_rows)
    rec = scope_all({hold_id: recs[hold_id]}, excluded_ids(cfg))[hold_id]
    k = true_class(rec)
    res = eval_recording(fn, rec, labelled_mask(rec), cfg, names)
    pred = res["y_pred"][res["y_true"] == k]
    dist = np.bincount(pred, minlength=N_CLASSES) / max(pred.size, 1)
    return {"id": hold_id, "class": names[k], "recall": float(dist[k]), "dist": dist, "event_pod": summarise_matches(res["matches"])["pod"]}


def top_classes(dist: np.ndarray, names: list[str], n: int = 3) -> str:
    order = np.argsort(dist)[::-1][:n]
    return ", ".join(f"{names[i]} {dist[i]:.2f}" for i in order if dist[i] > 0)


def print_folds(results: list[dict], names: list[str]) -> None:
    rows = [[r["id"], r["class"], r["recall"], r["event_pod"], top_classes(r["dist"], names)] for r in results]
    print_table(rows, ["held_out", "true_class", "recall", "event_PoD", "predicted distribution"])


def print_class_means(results: list[dict]) -> None:
    classes = sorted({r["class"] for r in results})
    rows = [[c, float(np.mean([r["recall"] for r in results if r["class"] == c])), REFERENCE_RECALL.get(c, "")] for c in classes]
    print_title("mean recall per class vs reference (per-cell LightGBM)")
    print_table(rows, ["class", "recall", "reference"])


def run_all(model_name: str, cfg: Config, only: str | None = None) -> list[dict]:
    recs, rows = load_all(data_dir(cfg)), load_patches(data_dir(cfg))
    ids = folds(scope_all(recs, excluded_ids(cfg))) if only is None else [only]
    assert all(i in recs for i in ids), f"unknown recording {only}; choose from {folds(recs)}"
    results = []
    for i in ids:
        print_title(f"{model_name} fold: hold out {i}")
        results.append(run_fold(model_name, i, cfg, recs, rows))
    names = class_names(cfg)
    print_title(f"{model_name}: leave-one-recording-out")
    print_folds(results, names)
    print_class_means(results)
    print_limits()
    return results
