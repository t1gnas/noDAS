import numpy as np

from das.config import Config, data_dir
from das.data import N_CLASSES, UNKNOWN, load_all, load_patches
from das.events import alarms_per_km_day, detect, event_metrics, false_alarms, gt_events, km_days, summarise_matches
from das.metrics import COARSE_NAMES, cell_report, coarse, threat_vs_background
from das.patches import cell_mask, split_rows
from das.predict import Predictor, class_names, load_model
from das.scope import excluded_ids, scope_all
from das.report import print_cell_report, print_confusion, print_limits, print_table, print_title

REFERENCE = {"macro_f1": 0.71, "accuracy": 0.81, "bg_acc": 0.97}
REFERENCE_F1 = {"openclose": 0.40, "running": 0.49, "walk": 0.69}
REFERENCE_RECALL = {"construction": 0.48}


def eval_recording(fn: Predictor, rec: dict, where: np.ndarray, cfg: Config, names: list[str]) -> dict:
    probs = fn(rec, where)
    assert not np.isnan(probs[where]).any(), f"{rec['id']}: model returned NaN inside the scored region"
    cls, labels, events = detect(probs, rec, cfg.alert, names, cfg.alert["threshold"])
    gt = gt_events(np.where(where, rec["targets"], UNKNOWN), cfg.eval["min_event_cells"])
    bg = where & (rec["targets"] == 0)
    return {"y_true": rec["targets"][where], "y_pred": probs[where].argmax(-1).astype(np.uint8), "attrs": rec["attrs"],
            "matches": event_metrics(gt, cls, labels, rec["attrs"], cfg.eval["event_hit_fraction"]),
            "fa": false_alarms(events, labels, bg), "bg_cells": int(bg.sum()), "n_events": len(events)}


def per_class_events(matches: list[dict], names: list[str]) -> list[list]:
    rows = []
    for k in range(1, N_CLASSES):
        s = summarise_matches([m for m in matches if m["class_id"] == k])
        if s["n_gt"]:
            rows.append([names[k], s["n_gt"], s["pod"], s["class_acc"], s["loc_err_m"]])
    return rows


def false_alarm_summary(results: list[dict]) -> dict:
    fa, bg = sum(r["fa"] for r in results), sum(r["bg_cells"] for r in results)
    attrs = results[0]["attrs"]
    return {"false_alarms": fa, "bg_km_days": km_days(bg, attrs), "fa_per_km_day": alarms_per_km_day(fa, bg, attrs)}


def aggregate(results: list[dict], cfg: Config, names: list[str]) -> dict:
    yt, yp = np.concatenate([r["y_true"] for r in results]), np.concatenate([r["y_pred"] for r in results])
    matches = [m for r in results for m in r["matches"]]
    coarse_rep = cell_report(coarse(yt, names, cfg.coarse_map), coarse(yp, names, cfg.coarse_map), len(COARSE_NAMES))
    return {"cells": cell_report(yt, yp, N_CLASSES), "coarse": coarse_rep, "tvb": threat_vs_background(yt, yp),
            "events": summarise_matches(matches), "event_rows": per_class_events(matches, names), **false_alarm_summary(results),
            "threshold": cfg.alert["threshold"], "hit_fraction": cfg.eval["event_hit_fraction"]}


def evaluate(name: str, cfg: Config, split: str = "test") -> dict:
    fn, names = load_model(name, cfg), class_names(cfg)
    recs, rows = scope_all(load_all(data_dir(cfg)), excluded_ids(cfg)), split_rows(load_patches(data_dir(cfg)), split)
    ids = sorted({r["recording_id"] for r in rows})
    results = [eval_recording(fn, recs[i], cell_mask(recs[i], rows), cfg, names) for i in ids]
    return aggregate(results, cfg, names)


def print_tvb(s: dict) -> None:
    t = s["tvb"]
    print(f"PoD {t['pod']:.3f}  NAR {t['nar']:.3f}  FNR {t['fnr']:.3f}  background-vs-activity accuracy {t['accuracy']:.3f}")


def print_event_summary(s: dict) -> None:
    e = s["events"]
    print_table(s["event_rows"], ["class", "gt_events", "PoD", "class_acc", "loc_err_m"])
    print(f"all: {e['n_gt']} GT events  PoD {e['pod']:.3f}  classification accuracy {e['class_acc']:.3f}  localisation error median {e['loc_err_m']:.1f} m, mean {e['loc_err_mean_m']:.1f} m")


def print_false_alarms(s: dict) -> None:
    print(f"false alarms {s['false_alarms']} over {s['bg_km_days']:.2e} km-days of annotated background -> {s['fa_per_km_day']:.1f} per km-day")
    print("warning: false alarms are measured only inside annotated background cells (one recording)")


def print_eval(name: str, s: dict, names: list[str], show_coarse: bool) -> None:
    print_title(f"{name}: official test split, per-cell (argmax)")
    print_cell_report(s["cells"], names)
    print_confusion(s["cells"]["confusion"], names)
    if show_coarse:
        print_title(f"{name}: coarse classes")
        print_cell_report(s["coarse"], list(COARSE_NAMES))
        print_confusion(s["coarse"]["confusion"], list(COARSE_NAMES))
    print_title(f"{name}: threat vs background (cells)")
    print_tvb(s)
    print_title(f"{name}: events (threshold {s['threshold']}, hit fraction {s['hit_fraction']})")
    print_event_summary(s)
    print_false_alarms(s)
    print_limits()


def compare_rows(summaries: dict[str, dict], names: list[str]) -> list[list]:
    pick = {"macro-F1": lambda s: s["cells"]["macro_f1"], "accuracy": lambda s: s["cells"]["accuracy"], "bg-vs-activity acc": lambda s: s["tvb"]["accuracy"],
            "threat PoD": lambda s: s["tvb"]["pod"], "NAR": lambda s: s["tvb"]["nar"], "event PoD": lambda s: s["events"]["pod"],
            "event class acc": lambda s: s["events"]["class_acc"], "loc err m": lambda s: s["events"]["loc_err_m"], "FA per km-day": lambda s: s["fa_per_km_day"]}
    ref = {"macro-F1": REFERENCE["macro_f1"], "accuracy": REFERENCE["accuracy"], "bg-vs-activity acc": REFERENCE["bg_acc"]}
    rows = [[k, *[f(s) for s in summaries.values()], ref.get(k, "")] for k, f in pick.items()]
    return rows + class_rows(summaries, names)


def class_rows(summaries: dict[str, dict], names: list[str]) -> list[list]:
    support = next(iter(summaries.values()))["cells"]["support"]
    f1 = [[f"F1 {n}", *[s["cells"]["f1"][i] for s in summaries.values()], REFERENCE_F1.get(n, "")] for i, n in enumerate(names) if support[i] > 0]
    rec = [[f"recall {n}", *[s["cells"]["recall"][names.index(n)] for s in summaries.values()], v] for n, v in REFERENCE_RECALL.items() if support[names.index(n)] > 0]
    return f1 + rec
