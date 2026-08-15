from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def equity_to_returns(equity: np.ndarray) -> np.ndarray:
    equity = np.asarray(equity, dtype=float)
    if equity.ndim != 1:
        raise ValueError("equity curve must be 1-D")
    if equity.size < 2:
        raise ValueError("need at least 2 equity points")
    if np.any(equity <= 0):
        raise ValueError("equity curve must be strictly positive to take returns")
    return equity[1:] / equity[:-1] - 1.0


def load_returns(
    path: str | Path,
    column: str | None = None,
    kind: str = "auto",
) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    numeric = df.select_dtypes("number")
    if numeric.shape[1] == 0:
        raise ValueError(f"no numeric column found in {path}")

    if column is not None:
        if column not in df.columns:
            raise ValueError(
                f"column {column!r} not in {path.name}; available: {list(df.columns)}"
            )
        series = pd.to_numeric(df[column], errors="coerce")
    else:
        series = numeric.iloc[:, -1]

    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("fewer than 2 usable values")

    if kind == "auto":
        looks_like_equity = bool(np.all(values > 0) and np.mean(values) > 0.5)
        kind = "equity" if looks_like_equity else "returns"

    if kind == "equity":
        return equity_to_returns(values)
    if kind == "returns":
        if np.nanmax(np.abs(values)) > 2.0:
            raise ValueError(
                "returns are too large; use decimals or set kind to equity"
            )
        return values
    raise ValueError(f"unknown kind {kind!r}")


def synthetic_returns(
    n: int = 1_260,
    mu_annual: float = 0.10,
    sigma_annual: float = 0.15,
    periods_per_year: float = 252.0,
    dist: str = "t",
    t_df: float = 4.0,
    autocorr: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be >= 2")
    rng = np.random.default_rng(seed)
    mu = mu_annual / periods_per_year
    sigma = sigma_annual / np.sqrt(periods_per_year)

    if dist == "normal":
        shocks = rng.standard_normal(n)
    elif dist == "t":
        if t_df <= 2:
            raise ValueError("t_df must be > 2 for finite variance")
        raw = rng.standard_t(t_df, size=n)
        shocks = raw / np.sqrt(t_df / (t_df - 2.0))
    else:
        raise ValueError(f"unknown dist {dist!r}")

    if autocorr:
        if not -1.0 < autocorr < 1.0:
            raise ValueError("autocorr must be in (-1, 1)")
        out = np.empty(n)
        out[0] = shocks[0]
        for i in range(1, n):
            out[i] = autocorr * out[i - 1] + np.sqrt(1 - autocorr**2) * shocks[i]
        shocks = out

    return mu + sigma * shocks
