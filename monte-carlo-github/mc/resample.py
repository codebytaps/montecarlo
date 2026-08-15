from __future__ import annotations

import numpy as np


def auto_block_length(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 8:
        return 1.0
    centred = r - r.mean()
    denom = float(centred @ centred)
    if denom <= 0:
        return 1.0
    rho = float(centred[:-1] @ centred[1:]) / denom
    rho = float(np.clip(abs(rho), 0.0, 0.95))
    if rho < 1e-3:
        return 1.0
    b = (np.sqrt(6.0) * rho / (1.0 - rho**2)) ** (2.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, max(1.0, n / 4.0)))


def _iid_indices(n: int, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=shape)


def _stationary_indices(
    n: int, shape: tuple[int, int], mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    n_paths, horizon = shape
    p = 1.0 / max(mean_block, 1.0)
    starts = rng.integers(0, n, size=shape)
    new_block = rng.random(shape) < p
    new_block[:, 0] = True

    t = np.arange(horizon)
    last_start = np.maximum.accumulate(np.where(new_block, t, -1), axis=1)
    offset = t - last_start
    base = np.take_along_axis(starts, last_start, axis=1)
    return (base + offset) % n


def _block_indices(
    n: int, shape: tuple[int, int], block: int, rng: np.random.Generator
) -> np.ndarray:
    n_paths, horizon = shape
    block = int(max(1, min(block, n)))
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_paths, n_blocks * block)[:, :horizon]


def resample(
    returns: np.ndarray,
    n_paths: int,
    horizon: int,
    method: str = "stationary",
    block: float | None = None,
    t_df: float = 4.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("returns must be a 1-D array with at least 2 observations")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns contain non-finite values")
    rng = rng if rng is not None else np.random.default_rng(0)
    shape = (n_paths, horizon)

    if method == "iid":
        return r[_iid_indices(r.size, shape, rng)]
    if method == "stationary":
        b = auto_block_length(r) if block is None else float(block)
        return r[_stationary_indices(r.size, shape, b, rng)]
    if method == "block":
        b = auto_block_length(r) if block is None else float(block)
        return r[_block_indices(r.size, shape, int(round(b)), rng)]
    if method == "normal":
        return rng.normal(r.mean(), r.std(ddof=1), size=shape)
    if method == "t":
        if t_df <= 2:
            raise ValueError("t_df must be > 2 for finite variance")
        raw = rng.standard_t(t_df, size=shape) / np.sqrt(t_df / (t_df - 2.0))
        return r.mean() + r.std(ddof=1) * raw
    raise ValueError(f"unknown method {method!r}")
