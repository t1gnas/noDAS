from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

from das import cnn, lgbm
from das.config import Config, data_dir
from das.data import N_CLASSES, crop_rows, load_file, load_metadata, norm_stats, seconds_to_row
from das.events import detect
from das.scope import excluded_ids, scoped

MODELS = ("lgbm", "cnn")
Predictor = Callable[..., np.ndarray]


def mask_where(probs: np.ndarray, where: np.ndarray | None) -> np.ndarray:
    if where is not None:
        probs[~where] = np.nan
    return probs


def cnn_predict(net: cnn.Net, dev: object, cfg: Config, rec: dict, where: np.ndarray | None = None) -> np.ndarray:
    return mask_where(cnn.predict_grid(net, dev, rec, cfg), where)


def lgbm_predict(model: object, cfg: Config, rec: dict, where: np.ndarray | None = None) -> np.ndarray:
    return lgbm.predict_grid(model, rec, cfg, where)


def load_model(name: str, cfg: Config) -> Predictor:
    assert name in MODELS, f"model {name!r} not in {MODELS}"
    if name == "lgbm":
        return scoped(partial(lgbm_predict, lgbm.load(cfg), cfg), excluded_ids(cfg))
    return scoped(partial(cnn_predict, *cnn.load(cfg), cfg), excluded_ids(cfg))


def class_names(cfg: Config) -> list[str]:
    meta = load_metadata(data_dir(cfg))
    return [meta["classes"][str(i)] for i in range(N_CLASSES)]


def load_recording_for_inference(path: str | Path, cfg: Config) -> dict:
    assert Path(path).exists(), f"file not found: {path}"
    mean, std = norm_stats(load_metadata(data_dir(cfg)))
    return load_file(Path(path), mean, std)


def crop_time(rec: dict, t0: float | None, t1: float | None) -> dict:
    a, n = rec["attrs"], rec["targets"].shape[0]
    return crop_rows(rec, seconds_to_row(t0, a, 0), seconds_to_row(t1, a, n))


def scan(path: str | Path, model_name: str, cfg: Config, t0: float | None, t1: float | None, thr: float | None) -> list[dict]:
    rec = crop_time(load_recording_for_inference(path, cfg), t0, t1)
    probs = load_model(model_name, cfg)(rec)
    threshold = cfg.alert["threshold"] if thr is None else thr
    return detect(probs, rec, cfg.alert, class_names(cfg), threshold)[2]
