from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from .engine import SimResult

PCTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
DD_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50)

CAVEATS = [
    "Future returns may not look like this sample.",
    "Backtest bias carries into these results.",
    "This measures risk, not whether the strategy works.",
    "IID sampling can make drawdowns look smaller.",
    "Costs must already be included in the returns.",
]


def _pct(a: np.ndarray, q) -> dict[str, float]:
    vals = np.atleast_1d(np.percentile(a, q))
    return {f"p{int(p):02d}": float(v) for p, v in zip(q, vals)}


def describe_input(r: np.ndarray, periods_per_year: float) -> dict:
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    centred = r - mu
    denom = float(centred @ centred)
    rho1 = float(centred[:-1] @ centred[1:]) / denom if denom > 0 else 0.0
    years = r.size / periods_per_year
    geo = float(np.expm1(np.log1p(r).sum() / years)) if np.all(r > -1) else float("nan")
    return {
        "n_periods": int(r.size),
        "years": float(years),
        "mean_period_return": mu,
        "ann_return_arithmetic": mu * periods_per_year,
        "ann_return_geometric": geo,
        "ann_vol": sd * float(np.sqrt(periods_per_year)),
        "sharpe": (mu / sd * float(np.sqrt(periods_per_year))) if sd > 0 else 0.0,
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "autocorr_lag1": rho1,
        "worst_period": float(r.min()),
        "best_period": float(r.max()),
        "pct_positive": float((r > 0).mean()),
    }


def summarise(res: SimResult) -> dict:
    m = res.metrics
    cfg = res.config
    ruined = m["ruined"].astype(bool)
    ttr = m["ruin_period"][ruined]

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "block_used": res.block_used,
        "input": describe_input(res.input_returns, cfg.periods_per_year),
        "observed": res.observed,
        "distribution": {
            key: {"mean": float(m[key].mean()), **_pct(m[key], PCTILES)}
            for key in ("terminal_equity", "total_return", "cagr", "max_drawdown",
                        "max_underwater", "time_underwater", "sharpe")
        },
        "probabilities": {
            "ruin": float(ruined.mean()),
            "loss": float((m["total_return"] < 0).mean()),
            "underperform_flat": float((m["terminal_equity"] < cfg.initial_capital).mean()),
            "negative_sharpe": float((m["sharpe"] < 0).mean()),
            **{f"drawdown_gt_{int(lvl * 100)}pct": float((m["max_drawdown"] > lvl).mean())
               for lvl in DD_LEVELS},
        },
        "time_to_ruin": (
            {"n_ruined": int(ruined.sum()), **_pct(ttr, (5, 25, 50, 75, 95))}
            if ruined.any() else {"n_ruined": 0}
        ),
        "caveats": CAVEATS,
    }


def _row(label: str, values: list[str], width: int = 22) -> str:
    return f"  {label:<{width}}" + "".join(f"{v:>13}" for v in values)


def render_text(summary: dict) -> str:
    cfg = summary["config"]
    inp = summary["input"]
    dist = summary["distribution"]
    prob = summary["probabilities"]
    obs = summary["observed"]
    basis = "starting capital" if cfg["ruin_basis"] == "initial" else "running peak"

    lines: list[str] = []
    add = lines.append
    add(f"Monte Carlo: {cfg['label']}")
    block_note = f" (mean block {summary['block_used']:.2f})" if summary["block_used"] else ""
    add(f"  {cfg['n_paths']:,} paths over {cfg['horizon']} periods")
    add(f"  {cfg['method']} sampling{block_note}, seed {cfg['seed']}")
    sizing = "compounded" if cfg["compound"] else "fixed notional"
    add(f"  {cfg['initial_capital']:,.0f} starting capital, {cfg['leverage']:g}x leverage, "
        f"{sizing}")
    stop = "stop trading" if cfg["stop_at_ruin"] else "keep trading"
    add(f"  ruin below {cfg['ruin_threshold']:.0%} of {basis}; {stop}")
    add("")

    add("Input")
    add(f"  {inp['n_periods']:,} periods, {inp['ann_return_arithmetic']:+.2%} annualised "
        f"return, {inp['ann_vol']:.2%} volatility")
    add(f"  {obs['cagr']:+.2%} CAGR, {obs['max_drawdown']:.2%} max drawdown, "
        f"Sharpe {inp['sharpe']:.2f}")
    add("")

    add("Outcomes")
    add(_row("", ["low 5%", "median", "high 5%"]))

    def line(label: str, key: str, fmt: str) -> None:
        d = dist[key]
        add(_row(label, [format(d[k], fmt) for k in ("p05", "p50", "p95")]))

    line("ending balance", "terminal_equity", ",.0f")
    line("total return", "total_return", "+.1%")
    line("annual return", "cagr", "+.2%")
    line("max drawdown", "max_drawdown", ".1%")
    line("underwater periods", "max_underwater", ".0f")
    line("Sharpe", "sharpe", ".2f")
    add("")

    add("Chances")
    add("  ruin:".ljust(38)
        + f"{prob['ruin']:>8.2%}")
    add("  ending below the start:".ljust(38) + f"{prob['underperform_flat']:>8.2%}")
    for lvl in DD_LEVELS:
        add(f"  drawdown over {lvl:.0%}:".ljust(38)
            + f"{prob[f'drawdown_gt_{int(lvl * 100)}pct']:>8.2%}")
    ttr = summary["time_to_ruin"]
    if ttr["n_ruined"]:
        add(f"  median time to ruin: {ttr['p50']:.0f} periods")
    add("")

    add("Notes")
    for caveat in summary["caveats"]:
        add(f"  - {caveat}")
    return "\n".join(lines)


def save(summary: dict, out_dir: Path, name: str = "summary") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return path


def render_sweep(sweep: dict) -> str:
    header = (f"  {'leverage':>9}{'ruin':>10}{'median CAGR':>14}"
              f"{'p05 CAGR':>12}{'median DD':>12}")
    lines = ["Leverage sweep", header]
    for i, lev in enumerate(sweep["leverage"]):
        lines.append(
            f"  {lev:>9.2f}{sweep['p_ruin'][i]:>10.1%}"
            f"{sweep['median_cagr'][i]:>14.2%}{sweep['p05_cagr'][i]:>12.2%}"
            f"{sweep['median_max_dd'][i]:>12.1%}"
        )
    return "\n".join(lines)
