from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from mc import SimConfig, simulate
from mc.engine import sweep_leverage
from mc.plots import save_all
from mc.report import render_text, save, summarise
from mc.strategy import StrategyError, load_strategy


def progress(text: str) -> None:
    print(f"PROGRESS {text}", flush=True)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "This file is used by the desktop app.\n"
            "Run mc_ui.py or run_mc.py instead.\n"
        )
        return 0

    job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    path = Path(job["path"])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    outcome: dict = {"ok": False, "name": path.stem, "error": ""}

    try:
        progress(f"loading {path.name}")
        loaded = load_strategy(path)

        kwargs = dict(job["config"])
        if loaded.periods_per_year:
            kwargs["periods_per_year"] = loaded.periods_per_year
        kwargs["label"] = loaded.name
        cfg = SimConfig(**kwargs)
        outcome["name"] = loaded.name

        progress(f"simulating {cfg.n_paths:,} paths x {cfg.horizon} periods")
        res = simulate(loaded.returns, cfg)
        summary = summarise(res)
        summary["source"] = str(path)

        sweep = None
        if job.get("sweep_levels"):
            progress("sweeping leverage")
            sweep = sweep_leverage(loaded.returns, cfg, job["sweep_levels"])
            summary["leverage_sweep"] = sweep

        save(summary, out_dir)
        (out_dir / "report.txt").write_text(render_text(summary), encoding="utf-8")

        for theme in job.get("themes", ["light"]):
            progress(f"drawing {theme} charts")
            save_all(res, out_dir, dark=(theme == "dark"), sweep=sweep)

        outcome.update(ok=True, error="")
    except StrategyError as exc:
        outcome["error"] = str(exc)
    except Exception as exc:
        outcome["error"] = (f"{type(exc).__name__}: {exc}\n\n"
                            f"{traceback.format_exc(limit=4)}")

    (out_dir / "run.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    progress("done")
    if not outcome["ok"]:
        print(outcome["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
