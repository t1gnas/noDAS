import numpy as np

from das.data import UNKNOWN

COARSE_NAMES = ("background", "person", "vehicle", "intrusion")


def scored(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert y_true.shape == y_pred.shape, f"y_true {y_true.shape} != y_pred {y_pred.shape}"
    keep = y_true != UNKNOWN
    t, p = y_true[keep].astype(np.int64), y_pred[keep].astype(np.int64)
    assert not (p == UNKNOWN).any(), f"{int((p == UNKNOWN).sum())} labelled cells have no prediction"
    return t, p


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    t, p = scored(y_true, y_pred)
    return np.bincount(t * n + p, minlength=n * n).reshape(n, n)


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(b > 0, a / np.maximum(b, 1), 0.0)


def report_from_cm(cm: np.ndarray) -> dict:
    tp, support, predicted = np.diag(cm).astype(float), cm.sum(1).astype(float), cm.sum(0).astype(float)
    precision, recall = safe_div(tp, predicted), safe_div(tp, support)
    f1 = safe_div(2 * precision * recall, precision + recall)
    macro = float(f1[support > 0].mean()) if (support > 0).any() else float("nan")
    acc = float(tp.sum() / max(cm.sum(), 1))
    return {"precision": precision, "recall": recall, "f1": f1, "support": support, "macro_f1": macro, "accuracy": acc, "confusion": cm}


def cell_report(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> dict:
    return report_from_cm(confusion(y_true, y_pred, n))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
    return cell_report(y_true, y_pred, n)["macro_f1"]


def coarse_lookup(class_names: list[str], coarse_map: dict) -> np.ndarray:
    lut = np.full(256, UNKNOWN, np.uint8)
    for i, name in enumerate(class_names):
        assert name in coarse_map, f"class {name!r} missing from coarse_map {sorted(coarse_map)}"
        lut[i] = COARSE_NAMES.index(coarse_map[name])
    return lut


def coarse(y: np.ndarray, class_names: list[str], coarse_map: dict) -> np.ndarray:
    return coarse_lookup(class_names, coarse_map)[y.astype(np.uint8)]


def threat_vs_background(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    t, p = scored(y_true, y_pred)
    threat, alarm = t != 0, p != 0
    pod = float(alarm[threat].mean()) if threat.any() else float("nan")
    nar = float(alarm[~threat].mean()) if (~threat).any() else float("nan")
    return {"pod": pod, "nar": nar, "fnr": 1.0 - pod, "accuracy": float((threat == alarm).mean())}
