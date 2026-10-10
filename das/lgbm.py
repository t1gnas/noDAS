import json

import joblib
import lightgbm as lgb
import numpy as np

from das.cellfeat import feature_names, gather, rec_features
from das.config import Config, data_dir, model_path
from das.data import N_CLASSES, load_all, load_patches
from das.patches import cell_mask, split_rows

MODEL_FILE = "lgbm.joblib"


def make_model(c: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=c["n_estimators"], learning_rate=c["learning_rate"], num_leaves=c["num_leaves"], class_weight=c["class_weight"],
                              subsample=c["subsample"], subsample_freq=1, colsample_bytree=c["colsample"], random_state=c["seed"], n_jobs=-1, verbose=-1)


def collect_cells(recs: dict[str, dict], rows: list[dict], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    parts = [gather(recs[i], cell_mask(recs[i], rows), cfg.cellfeat) for i in sorted({r["recording_id"] for r in rows})]
    X, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    assert X.shape[0] == y.shape[0] and X.shape[0] > 0, f"no cells gathered: X {X.shape} y {y.shape}"
    return X, y


def cap_cells(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= n:
        return X, y
    idx = np.sort(np.random.default_rng(seed).choice(len(y), n, replace=False))
    return X[idx], y[idx]


def train(cfg: Config, recs: dict[str, dict], train_rows: list[dict], val_rows: list[dict] | None = None) -> lgb.LGBMClassifier:
    X, y = cap_cells(*collect_cells(recs, train_rows, cfg), cfg.lgbm["max_train_cells"], cfg.lgbm["seed"])
    print(f"lgbm: {len(y)} training cells, class counts {np.bincount(y, minlength=N_CLASSES).tolist()}")
    model = make_model(cfg.lgbm)
    if val_rows is None:
        return model.fit(X, y)
    Xv, yv = collect_cells(recs, val_rows, cfg)
    callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(25)]
    return model.fit(X, y, eval_X=(Xv,), eval_y=(yv,), callbacks=callbacks)


def save(model: lgb.LGBMClassifier, cfg: Config) -> None:
    joblib.dump(model, model_path(cfg, MODEL_FILE))
    imp = dict(zip(feature_names(), model.booster_.feature_importance("gain").round(1).tolist()))
    model_path(cfg, "lgbm_importance.json").write_text(json.dumps(imp, indent=2))
    print(f"saved {model_path(cfg, MODEL_FILE)} (best iteration {model.best_iteration_ or model.n_estimators})")


def fit(cfg: Config) -> lgb.LGBMClassifier:
    rows, recs = load_patches(data_dir(cfg)), load_all(data_dir(cfg))
    model = train(cfg, recs, split_rows(rows, "train"), split_rows(rows, "val"))
    save(model, cfg)
    return model


def load(cfg: Config) -> lgb.LGBMClassifier:
    path = model_path(cfg, MODEL_FILE)
    if not path.exists():
        raise SystemExit(f"error: model file {path} not found; run `docker compose run --rm das train lgbm` first")
    return joblib.load(path)


def predict_grid(model: lgb.LGBMClassifier, rec: dict, cfg: Config, where: np.ndarray | None = None) -> np.ndarray:
    m = rec["coverage"] if where is None else rec["coverage"] & where
    probs = np.full((*m.shape, N_CLASSES), np.nan, np.float32)
    feats = rec_features(rec, cfg.cellfeat)[m]
    if feats.shape[0]:
        out = np.zeros((feats.shape[0], N_CLASSES), np.float32)
        out[:, model.classes_] = model.predict_proba(feats)
        probs[m] = out
    return probs
