from collections.abc import Callable
from itertools import product

import numpy as np


def tile_starts(n: int, size: int, stride: int) -> list[int]:
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, stride))
    return starts if starts[-1] == n - size else starts + [n - size]


def pad_grid(grid: np.ndarray, rows: int, ch: int) -> np.ndarray:
    pt, pc = max(0, rows - grid.shape[0]), max(0, ch - grid.shape[1])
    return np.pad(grid, ((0, pt), (0, pc), (0, 0)), mode="edge") if pt or pc else grid


def tile_weight(rows: int, ch: int) -> np.ndarray:
    wr = 0.1 + np.hanning(rows + 2)[1:-1]
    wc = 0.1 + np.hanning(ch + 2)[1:-1]
    return np.outer(wr, wc).astype(np.float32)


def stack_tiles(grid: np.ndarray, origins: list[tuple[int, int]], rows: int, ch: int) -> np.ndarray:
    tiles = [grid[t:t + rows, c:c + ch].transpose(2, 0, 1) for t, c in origins]
    return np.ascontiguousarray(np.stack(tiles), dtype=np.float32)


def accumulate(acc: np.ndarray, wsum: np.ndarray, probs: np.ndarray, origins: list[tuple[int, int]], w: np.ndarray) -> None:
    rows, ch = w.shape
    for (t, c), p in zip(origins, probs):
        acc[t:t + rows, c:c + ch] += p.transpose(1, 2, 0) * w[..., None]
        wsum[t:t + rows, c:c + ch] += w


def mask_coverage(probs: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    T, C = coverage.shape
    out = probs[:T, :C].copy()
    out[~coverage] = np.nan
    return out


def predict_tiled(fn: Callable[[np.ndarray], np.ndarray], grid: np.ndarray, coverage: np.ndarray, rows: int, ch: int, stride_rows: int, stride_ch: int, batch: int = 64) -> np.ndarray:
    g = pad_grid(grid, rows, ch)
    origins = list(product(tile_starts(g.shape[0], rows, stride_rows), tile_starts(g.shape[1], ch, stride_ch)))
    w, acc, wsum = tile_weight(rows, ch), None, np.zeros(g.shape[:2], np.float32)
    for i in range(0, len(origins), batch):
        part = origins[i:i + batch]
        probs = fn(stack_tiles(g, part, rows, ch))
        acc = np.zeros((*g.shape[:2], probs.shape[1]), np.float32) if acc is None else acc
        accumulate(acc, wsum, probs, part, w)
    return mask_coverage(acc / wsum[..., None], coverage)
