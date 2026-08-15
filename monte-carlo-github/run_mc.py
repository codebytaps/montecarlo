from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from mc import SimConfig, simulate
from mc.data import load_returns, synthetic_returns
from mc.engine import sweep_leverage
from mc.plots import save_all
from mc.report import render_sweep, render_text, save, summarise

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Monte Carlo simulation of strategy return paths and risk of ruin.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_argument_group("input")
    src.add_argument("--input", type=Path, help="CSV/parquet of returns or an equity curve")
    src.add_argument("--column", help="value column (default: last numeric column)")
    src.add_argument("--kind", choices=["auto", "returns", "equity"], default="auto")
    src.add_argument("--synthetic", action="store_true",
                     help="generate a stand-in return series instead of reading a file")
    src.add_argument("--synthetic-mu", type=float, default=0.10, help="annual mean return")
    src.add_argument("--synthetic-sigma", type=float, default=0.15, help="annual vol")
    src.add_argument("--synthetic-n", type=int, default=1260, help="periods to generate")
    src.add_argument("--synthetic-autocorr", type=float, default=0.05)

    sim = p.add_argument_group("simulation")
    sim.add_argument("--method", default="stationary",
                     choices=["stationary", "iid", "block", "normal", "t"])
    sim.add_argument("--block", type=float, help="mean block length (default: auto)")
    sim.add_argument("--paths", type=int, default=10_000)
    sim.add_argument("--horizon", type=int, default=252, help="periods to simulate forward")
    sim.add_argument("--periods-per-year", type=float, default=252.0)
    sim.add_argument("--seed", type=int, default=0)

    cap = p.add_argument_group("capital and ruin")
    cap.add_argument("--capital", type=float, default=100_000.0)
    cap.add_argument("--leverage", type=float, default=1.0)
    cap.add_argument("--no-compound", action="store_true",
                     help="risk a fixed notional every period instead of compounding")
    cap.add_argument("--ruin", type=float, default=0.5,
                     help="ruin when equity falls to this fraction (default 0.5)")
    cap.add_argument("--ruin-basis", choices=["initial", "peak"], default="initial")
    cap.add_argument("--no-stop-at-ruin", action="store_true",
                     help="keep trading through the barrier instead of stopping")
    cap.add_argument("--sweep-leverage", help="comma-separated levels, e.g. 0.5,1,2,3")

    out = p.add_argument_group("output")
    out.add_argument("--label", default="run", help="name for this run; also the output folder")
    out.add_argument("--out", type=Path, default=HERE / "results")
    out.add_argument("--dark", action="store_true", help="also write dark-mode charts")
    out.add_argument("--no-charts", action="store_true")
    return p


def resolve_returns(args) -> tuple[np.ndarray, str]:
    if args.synthetic or args.input is None:
        if args.input is None and not args.synthetic:
            print("no --input given; falling back to --synthetic\n", file=sys.stderr)
        r = synthetic_returns(
            n=args.synthetic_n, mu_annual=args.synthetic_mu,
            sigma_annual=args.synthetic_sigma, periods_per_year=args.periods_per_year,
            dist="t", autocorr=args.synthetic_autocorr, seed=args.seed,
        )
        return r, "synthetic"
    return load_returns(args.input, column=args.column, kind=args.kind), str(args.input)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    returns, source = resolve_returns(args)

    cfg = SimConfig(
        method=args.method, n_paths=args.paths, horizon=args.horizon,
        block=args.block, seed=args.seed, initial_capital=args.capital,
        leverage=args.leverage, compound=not args.no_compound,
        ruin_threshold=args.ruin, ruin_basis=args.ruin_basis,
        stop_at_ruin=not args.no_stop_at_ruin,
        periods_per_year=args.periods_per_year, label=args.label,
        notes=f"source={source}",
    )

    res = simulate(returns, cfg)
    summary = summarise(res)
    summary["source"] = source

    out_dir = Path(args.out) / args.label
    print(render_text(summary))

    sweep = None
    if args.sweep_leverage:
        levels = [float(x) for x in args.sweep_leverage.split(",") if x.strip()]
        sweep = sweep_leverage(returns, cfg, levels)
        summary["leverage_sweep"] = sweep
        print()
        print(render_sweep(sweep))

    json_path = save(summary, out_dir)
    written = [json_path]
    if not args.no_charts:
        written += save_all(res, out_dir, dark=False, sweep=sweep)
        if args.dark:
            written += save_all(res, out_dir, dark=True, sweep=sweep)

    print(f"\nwrote {len(written)} files to {out_dir}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
