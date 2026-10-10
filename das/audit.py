from pathlib import Path

import numpy as np

from das.data import N_CLASSES, SPLITS, UNKNOWN, assert_consistent, load_all, load_metadata, load_patches
from das.report import print_table, print_title


def class_names(meta: dict) -> list[str]:
    return [meta["classes"][str(i)] for i in range(N_CLASSES)]


def recording_row(rec: dict) -> list:
    t = rec["targets"]
    counts = np.bincount(t[t != UNKNOWN].ravel(), minlength=N_CLASSES)
    return [rec["id"], f"{t.shape[0]}x{t.shape[1]}", f"{100 * rec['coverage'].mean():.1f}", int((t == UNKNOWN).sum()), *counts.tolist()]


def patch_count_rows(patches: list[dict], names: list[str]) -> list[list]:
    grid = np.zeros((N_CLASSES, len(SPLITS)), int)
    for p in patches:
        grid[p["class_id"], SPLITS.index(p["split"])] += 1
    return [[names[i], *grid[i].tolist()] for i in range(N_CLASSES)] + [["total", *grid.sum(0).tolist()]]


def run(root: Path) -> None:
    meta, patches, recs = load_metadata(root), load_patches(root), load_all(root)
    names = class_names(meta)
    print_title(f"{len(recs)} recordings in {root}")
    print_table([recording_row(r) for r in recs.values()], ["recording", "TxC", "cover%", "unknown", *[n[:6] for n in names]])
    print_title("patches per split and class")
    print_table(patch_count_rows(patches, names), ["class", *SPLITS])
    assert_consistent(recs, patches, meta)
    counts = {s: sum(p["split"] == s for p in patches) for s in SPLITS}
    print(f"\nOK: {len(recs)} recordings, patches {counts}, all consistency checks passed")
