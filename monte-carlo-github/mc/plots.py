from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from .engine import SimResult


@dataclass(frozen=True)
class Theme:
    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    series_1: str
    series_2: str
    ramp: tuple[str, ...]


LIGHT = Theme(
    surface="#fcfcfb", text_primary="#0b0b0b", text_secondary="#52514e",
    muted="#8a8a85", grid="#e6e5e1", series_1="#2a78d6", series_2="#eb6834",
    ramp=("#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"),
)
DARK = Theme(
    surface="#1a1a19", text_primary="#ffffff", text_secondary="#c3c2b7",
    muted="#8a8a85", grid="#33332f", series_1="#3987e5", series_2="#d95926",
    ramp=("#104281", "#184f95", "#256abf", "#3987e5", "#86b6ef"),
)

_PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
_MONEY = FuncFormatter(lambda v, _: f"{v:,.0f}")


def _style(theme: Theme) -> dict:
    return {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "text.color": theme.text_primary,
        "axes.labelcolor": theme.text_secondary,
        "axes.edgecolor": theme.grid,
        "xtick.color": theme.text_secondary,
        "ytick.color": theme.text_secondary,
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "font.size": 10,
    }


def _title(ax, title: str, subtitle: str, theme: Theme) -> None:
    ax.set_title(title, loc="left", pad=18, color=theme.text_primary)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=theme.text_secondary)


def _hline(ax, y: float, label: str, theme: Theme, ha: str = "left") -> None:
    ax.axhline(y, color=theme.muted, linewidth=1.2, linestyle=(0, (5, 4)), zorder=1)
    x = 0.01 if ha == "left" else 0.99
    ax.text(x, y, f" {label} ", transform=ax.get_yaxis_transform(), ha=ha,
            va="bottom", fontsize=8, color=theme.text_secondary)


def _vline(ax, x: float, label: str, theme: Theme, align: str = "left") -> None:
    ax.axvline(x, color=theme.muted, linewidth=1.2, linestyle=(0, (5, 4)), zorder=1)
    pad = " " if align == "left" else ""
    ax.text(x, 0.985, f"{pad}{label}{'' if align == 'left' else ' '}",
            transform=ax.get_xaxis_transform(), ha=align, va="top",
            fontsize=8, color=theme.text_secondary)


def fan_chart(res: SimResult, ax, theme: Theme = LIGHT) -> None:
    eq = res.equity_sample.astype(float)
    cfg = res.config
    if eq.size == 0:
        ax.text(0.5, 0.5, "no retained paths", ha="center", va="center")
        return

    t = np.arange(eq.shape[1])
    qs = np.percentile(eq, [1, 5, 25, 50, 75, 95, 99], axis=0)
    p01, p05, p25, p50, p75, p95, p99 = qs

    ax.fill_between(t, p01, p99, color=theme.ramp[0], linewidth=0, label="1st-99th pct")
    ax.fill_between(t, p05, p95, color=theme.ramp[1], linewidth=0, label="5th-95th pct")
    ax.fill_between(t, p25, p75, color=theme.ramp[2], linewidth=0, label="25th-75th pct")
    ax.plot(t, p50, color=theme.ramp[4], linewidth=2.0, label="median path", zorder=4)

    _hline(ax, cfg.initial_capital, "starting capital", theme, ha="right")

    barrier = cfg.ruin_threshold * cfg.initial_capital
    if cfg.ruin_basis == "initial" and barrier > 0:
        if barrier > 0.85 * float(p01.min()):
            _hline(ax, barrier, f"ruin barrier ({cfg.ruin_threshold:.0%} of start)", theme)

    if p01.min() > 0 and p99.max() / p01.min() > 12:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(_MONEY)
        ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.yaxis.set_major_formatter(_MONEY)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(f"periods forward ({cfg.periods_per_year:g} per year)")
    ax.set_ylabel("equity")
    _title(ax, "Equity paths",
           f"{res.n_paths:,} paths, {cfg.horizon} periods, {cfg.leverage:g}x leverage",
           theme)
    ax.legend(loc="upper left", ncol=2)


def drawdown_distribution(res: SimResult, ax, theme: Theme = LIGHT) -> None:
    dd = res.metrics["max_drawdown"]
    ax.hist(dd, bins=60, color=theme.series_1, rwidth=0.9)
    med, p95 = float(np.median(dd)), float(np.percentile(dd, 95))
    _vline(ax, med, f"median {med:.0%}", theme, align="right")
    _vline(ax, p95, f"95th {p95:.0%}", theme)
    ax.margins(y=0.16)

    obs = res.observed["max_drawdown"]
    ax.axvline(obs, color=theme.series_2, linewidth=2.0, zorder=5,
               label=f"original: {obs:.0%}")
    ax.xaxis.set_major_formatter(_PCT)
    ax.set_xlabel("worst drawdown over the horizon")
    ax.set_ylabel("paths")
    pct_worse = float((dd > obs).mean())
    _title(ax, "Maximum drawdown",
           f"{pct_worse:.0%} are worse than the original", theme)
    ax.legend(loc="center right")


def drawdown_exceedance(res: SimResult, ax, theme: Theme = LIGHT) -> None:
    dd = np.sort(res.metrics["max_drawdown"])
    exceed = 1.0 - np.arange(dd.size) / dd.size
    ax.plot(dd, exceed, color=theme.series_1, linewidth=2.0, label="simulated")

    obs = res.observed["max_drawdown"]
    p_obs = float((dd > obs).mean())
    ax.plot([obs], [p_obs], marker="o", markersize=8, color=theme.series_2,
            markeredgecolor=theme.surface, markeredgewidth=2, zorder=5,
            linestyle="none", label="original")
    ax.annotate(f"original: {obs:.0%}\nchance: {p_obs:.0%}",
                xy=(obs, p_obs), xytext=(12, 12), textcoords="offset points",
                fontsize=8, color=theme.text_secondary)

    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlim(0, min(1.0, float(np.percentile(dd, 99.9)) * 1.15))
    ax.set_ylim(0, 1)
    ax.set_xlabel("drawdown depth")
    ax.set_ylabel("probability of seeing it or worse")
    _title(ax, "Drawdown exceedance", "chance of reaching each drawdown level", theme)
    ax.legend(loc="upper right")


def terminal_distribution(res: SimResult, ax, theme: Theme = LIGHT) -> None:
    eq = res.metrics["terminal_equity"]
    start = res.config.initial_capital
    positive = eq[eq > 0]
    if positive.size and positive.max() / max(positive.min(), 1e-9) > 50:
        bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 60)
        ax.set_xscale("log")
    else:
        bins = 60
    ax.hist(eq, bins=bins, color=theme.series_1, rwidth=0.9)
    _vline(ax, start, "starting capital", theme)
    ax.margins(y=0.16)

    ax.xaxis.set_major_formatter(_MONEY)
    ax.set_xlabel("terminal equity")
    ax.set_ylabel("paths")
    p_loss = float((eq < start).mean())
    _title(ax, "Terminal equity",
           f"{p_loss:.0%} end below the start", theme)


def underwater_distribution(res: SimResult, ax, theme: Theme = LIGHT) -> None:
    uw = res.metrics["max_underwater"]
    ppy = res.config.periods_per_year
    ax.hist(uw, bins=60, color=theme.series_1, rwidth=0.9)
    med = float(np.median(uw))
    _vline(ax, med, f"median {med:.0f} periods", theme)
    ax.margins(y=0.16)

    obs = res.observed["max_underwater"]
    if obs <= res.config.horizon:
        ax.axvline(obs, color=theme.series_2, linewidth=2.0, zorder=5,
                   label=f"original: {obs:.0f} periods")
        ax.legend(loc="center right")
    ax.set_xlabel(f"longest stretch underwater (periods; {ppy:g} = 1 year)")
    ax.set_ylabel("paths")
    still_under = float((uw >= res.config.horizon).mean())
    _title(ax, "Time underwater",
           f"median {med / ppy:.2f} years; {still_under:.0%} still underwater at the end",
           theme)


def leverage_sweep(sweep: dict, axes, theme: Theme = LIGHT) -> None:
    ax_top, ax_bot = axes
    lev = np.asarray(sweep["leverage"], dtype=float)

    ax_top.plot(lev, sweep["p_ruin"], color=theme.series_1, linewidth=2.0,
                marker="o", markersize=8, markeredgecolor=theme.surface,
                markeredgewidth=2)
    for x, y in zip(lev, sweep["p_ruin"]):
        if y == 0:
            text = "0%"
        elif y < 0.001:
            text = "<0.1%"
        elif y < 0.02:
            text = f"{y:.1%}"
        else:
            text = f"{y:.0%}"
        ax_top.annotate(text, (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8,
                        color=theme.text_secondary)
    ax_top.yaxis.set_major_formatter(_PCT)
    ax_top.set_ylim(bottom=0)
    ax_top.set_ylabel("P(ruin)")
    _title(ax_top, "Leverage sweep",
           "ruin probability and annual return by leverage", theme)

    ax_bot.plot(lev, sweep["median_cagr"], color=theme.series_1, linewidth=2.0,
                marker="o", markersize=8, markeredgecolor=theme.surface,
                markeredgewidth=2, label="median CAGR")
    ax_bot.plot(lev, sweep["p05_cagr"], color=theme.series_2, linewidth=2.0,
                linestyle=(0, (5, 3)), marker="s", markersize=7,
                markeredgecolor=theme.surface, markeredgewidth=2,
                label="5th-percentile CAGR")
    ax_bot.axhline(0, color=theme.muted, linewidth=1.2, linestyle=(0, (5, 4)))
    med = np.asarray(sweep["median_cagr"], dtype=float)
    peak = int(np.argmax(med))
    if 0 < peak < len(med) - 1:
        ax_bot.annotate(
            f"highest median return: {lev[peak]:g}x",
            xy=(lev[peak], med[peak]), xytext=(10, -46), textcoords="offset points",
            fontsize=8, color=theme.text_secondary,
            arrowprops=dict(arrowstyle="-", color=theme.muted, linewidth=1))
    ax_bot.yaxis.set_major_formatter(_PCT)
    ax_bot.set_xticks(lev)
    ax_bot.set_xticklabels([f"{x:g}x" for x in lev])
    ax_bot.set_xlabel("leverage multiple on every period's return")
    ax_bot.set_ylabel("CAGR")
    ax_bot.legend(loc="lower left")


def dashboard(res: SimResult, theme: Theme = LIGHT):
    with plt.rc_context(_style(theme)):
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1.25, 1],
                      hspace=0.42, wspace=0.26,
                      left=0.06, right=0.97, top=0.90, bottom=0.09)
        fan_chart(res, fig.add_subplot(gs[0, :]), theme)
        drawdown_distribution(res, fig.add_subplot(gs[1, 0]), theme)
        drawdown_exceedance(res, fig.add_subplot(gs[1, 1]), theme)
        terminal_distribution(res, fig.add_subplot(gs[1, 2]), theme)
        fig.suptitle(f"Monte Carlo: {res.config.label}", x=0.06, ha="left",
                     fontsize=15, fontweight="bold", color=theme.text_primary)
        fig.text(0.06, 0.945,
                 "based on the supplied return sample",
                 ha="left", fontsize=9.5, color=theme.text_secondary)
    return fig


def save_all(res: SimResult, out_dir: Path, dark: bool = False,
             sweep: dict | None = None) -> list[Path]:
    theme = DARK if dark else LIGHT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_dark" if dark else ""
    written: list[Path] = []

    fig = dashboard(res, theme)
    path = out_dir / f"dashboard{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    panels = {
        "fan": fan_chart,
        "drawdown_hist": drawdown_distribution,
        "drawdown_exceedance": drawdown_exceedance,
        "terminal_equity": terminal_distribution,
        "underwater": underwater_distribution,
    }
    with plt.rc_context(_style(theme)):
        for name, fn in panels.items():
            fig, ax = plt.subplots(figsize=(9, 5.5))
            fn(res, ax, theme)
            fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.13)
            path = out_dir / f"{name}{suffix}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

        if sweep:
            fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
            leverage_sweep(sweep, axes, theme)
            fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.09,
                                hspace=0.12)
            path = out_dir / f"leverage_sweep{suffix}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

    return written
