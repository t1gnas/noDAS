from collections.abc import Callable
from functools import partial

import numpy as np

from das.config import Config, data_dir
from das.data import N_CLASSES, UNKNOWN, load_metadata


def excluded_ids(cfg: Config) -> list[int]:
    names = [load_metadata(data_dir(cfg))["classes"][str(i)] for i in range(N_CLASSES)]
    wanted = list(cfg.scope.get("exclude_classes", []))
    unknown = [n for n in wanted if n not in names]
    assert not unknown, f"scope.exclude_classes {unknown} not in classes {names}"
    return [names.index(n) for n in wanted]


def mask_probs(probs: np.ndarray, ids: list[int]) -> np.ndarray:
    if not ids:
        return probs
    p = probs.copy()
    p[..., ids] = 0.0
    total = p.sum(-1, keepdims=True)
    return p / np.where(total > 0, total, 1.0)


def scope_targets(rec: dict, ids: list[int]) -> dict:
    t = rec["targets"].copy()
    t[np.isin(t, ids)] = UNKNOWN
    return {**rec, "targets": t}


def scope_all(recs: dict[str, dict], ids: list[int]) -> dict[str, dict]:
    return {k: scope_targets(r, ids) for k, r in recs.items()} if ids else recs


def masked_call(fn: Callable[..., np.ndarray], ids: list[int], rec: dict, where: np.ndarray | None = None) -> np.ndarray:
    return mask_probs(fn(rec, where), ids)


def scoped(fn: Callable[..., np.ndarray], ids: list[int]) -> Callable[..., np.ndarray]:
    return partial(masked_call, fn, ids) if ids else fn
