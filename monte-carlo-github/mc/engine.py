from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .config import SimConfig
from .resample import auto_block_length, resample


def running_peak(equity: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(equity, axis=1)


def max_drawdown(equity: np.ndarray) -> np.ndarray:
    peak = running_peak(equity)
    return (1.0 - equity / peak).max(axis=1)


def max_underwater(equity: np.ndarray) -> np.ndarray:
    peak = running_peak(equity)
    at_peak = equity >= peak
    t = np.arange(equity.shape[1])
    last_peak = np.maximum.accumulate(np.where(at_peak, t, -1), axis=1)
    return (t - last_peak).max(axis=1).astype(float)


def time_underwater(equity: np.ndarray) -> np.ndarray:
    return (equity < running_peak(equity)).mean(axis=1)


def total_return(equity: np.ndarray) -> np.ndarray:
    return equity[:, -1] / equity[:, 0] - 1.0


def cagr(equity: np.ndarray, periods_per_year: float) -> np.ndarray:
    years = (equity.shape[1] - 1) / periods_per_year
    growth = equity[:, -1] / equity[:, 0]
    out = np.full(growth.shape, -1.0)
    alive = growth > 0
    out[alive] = growth[alive] ** (1.0 / years) - 1.0
    return out


def path_sharpe(returns: np.ndarray, periods_per_year: float) -> np.ndarray:
    mu = returns.mean(axis=1)
    sd = returns.std(axis=1, ddof=1)
    out = np.zeros_like(mu)
    ok = sd > 0
    out[ok] = mu[ok] / sd[ok] * np.sqrt(periods_per_year)
    return out


def compute_metrics(
    equity: np.ndarray,
    returns: np.ndarray,
    periods_per_year: float,
    ruin_period: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "terminal_equity": equity[:, -1].copy(),
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "max_underwater": max_underwater(equity),
        "time_underwater": time_underwater(equity),
        "sharpe": path_sharpe(returns, periods_per_year),
        "ruined": (ruin_period >= 0).astype(float),
        "ruin_period": ruin_period.astype(float),
    }


def build_equity(
    returns: np.ndarray,
    initial_capital: float,
    leverage: float,
    compound: bool,
) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    n_paths = r.shape[0]
    ones = np.ones((n_paths, 1))

    if compound:
        gross = np.clip(1.0 + leverage * r, 0.0, None)
        equity = initial_capital * np.cumprod(np.hstack([ones, gross]), axis=1)
    else:
        cum = np.cumsum(leverage * r, axis=1)
        equity = initial_capital * np.hstack([ones, 1.0 + cum])
        np.clip(equity, 0.0, None, out=equity)
    return equity


def apply_ruin_barrier(
    equity: np.ndarray,
    initial_capital: float,
    threshold: float,
    basis: str,
    stop_at_ruin: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if basis == "initial":
        hit = equity <= threshold * initial_capital
    elif basis == "peak":
        hit = equity <= threshold * running_peak(equity)
    else:
        raise ValueError(f"unknown ruin_basis {basis!r}")
    hit |= equity <= 0.0
    hit[:, 0] = False

    any_hit = hit.any(axis=1)
    ruin_period = np.where(any_hit, hit.argmax(axis=1), -1)

    if stop_at_ruin and any_hit.any():
        n_cols = equity.shape[1]
        stop_at = np.where(any_hit, ruin_period, n_cols - 1)
        cols = np.minimum(np.arange(n_cols)[None, :], stop_at[:, None])
        equity = np.take_along_axis(equity, cols, axis=1)
    return equity, ruin_period


def realised_returns(equity: np.ndarray) -> np.ndarray:
    prev = equity[:, :-1]
    out = np.zeros_like(prev)
    ok = prev > 0
    out[ok] = equity[:, 1:][ok] / prev[ok] - 1.0
    return out


@dataclass
class SimResult:
    config: SimConfig
    input_returns: np.ndarray
    metrics: dict[str, np.ndarray]
    equity_sample: np.ndarray
    block_used: float | None
    observed: dict[str, float] = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return int(self.metrics["terminal_equity"].size)


def _observed_metrics(returns: np.ndarray, cfg: SimConfig) -> dict[str, float]:
    eq = build_equity(returns[None, :], cfg.initial_capital, cfg.leverage, cfg.compound)
    eq, ruin = apply_ruin_barrier(
        eq, cfg.initial_capital, cfg.ruin_threshold, cfg.ruin_basis, cfg.stop_at_ruin
    )
    vals = compute_metrics(eq, realised_returns(eq), cfg.periods_per_year, ruin)
    out = {k: float(v[0]) for k, v in vals.items()}
    out["n_periods"] = float(returns.size)
    return out


def simulate(returns: np.ndarray, cfg: SimConfig) -> SimResult:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("returns must be a 1-D array with at least 2 observations")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns contain non-finite values")

    rng = np.random.default_rng(cfg.seed)
    block_used: float | None = None
    if cfg.method in {"stationary", "block"}:
        block_used = auto_block_length(r) if cfg.block is None else float(cfg.block)

    collected: dict[str, list[np.ndarray]] = {}
    kept: list[np.ndarray] = []
    kept_n = 0
    remaining = cfg.n_paths
    chunk = cfg.chunk_size

    while remaining > 0:
        size = min(chunk, remaining)
        sim = resample(
            r, size, cfg.horizon,
            method=cfg.method, block=block_used, t_df=cfg.t_df, rng=rng,
        )
        eq = build_equity(sim, cfg.initial_capital, cfg.leverage, cfg.compound)
        eq, ruin = apply_ruin_barrier(
            eq, cfg.initial_capital, cfg.ruin_threshold, cfg.ruin_basis, cfg.stop_at_ruin
        )
        vals = compute_metrics(eq, realised_returns(eq), cfg.periods_per_year, ruin)
        for key, arr in vals.items():
            collected.setdefault(key, []).append(arr)

        if kept_n < cfg.fan_paths:
            take = min(cfg.fan_paths - kept_n, size)
            kept.append(eq[:take].astype(np.float32))
            kept_n += take

        remaining -= size

    merged = {k: np.concatenate(v) for k, v in collected.items()}
    equity_sample = np.vstack(kept) if kept else np.empty((0, cfg.horizon + 1), np.float32)

    return SimResult(
        config=cfg,
        input_returns=r,
        metrics=merged,
        equity_sample=equity_sample,
        block_used=block_used,
        observed=_observed_metrics(r, cfg),
    )


def sweep_leverage(
    returns: np.ndarray,
    cfg: SimConfig,
    levels: list[float],
    n_paths: int | None = None,
) -> dict:
    n_paths = n_paths or min(cfg.n_paths, 5_000)
    out: dict[str, list[float]] = {
        "leverage": [], "p_ruin": [], "median_cagr": [], "p05_cagr": [],
        "median_max_dd": [], "p95_max_dd": [], "p_loss": [],
    }
    for lev in levels:
        sub = replace(cfg, leverage=float(lev), n_paths=n_paths, fan_paths=0,
                      label=f"{cfg.label}-lev{lev:g}")
        metrics = simulate(returns, sub).metrics
        out["leverage"].append(float(lev))
        out["p_ruin"].append(float(metrics["ruined"].mean()))
        out["median_cagr"].append(float(np.median(metrics["cagr"])))
        out["p05_cagr"].append(float(np.percentile(metrics["cagr"], 5)))
        out["median_max_dd"].append(float(np.median(metrics["max_drawdown"])))
        out["p95_max_dd"].append(float(np.percentile(metrics["max_drawdown"], 95)))
        out["p_loss"].append(float((metrics["total_return"] < 0).mean()))
    return out
