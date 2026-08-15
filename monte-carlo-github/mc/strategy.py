from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import equity_to_returns, load_returns

RETURN_NAMES = ("RETURNS", "returns")
EQUITY_NAMES = ("EQUITY", "equity")
RETURN_FUNCS = ("get_returns",)
EQUITY_FUNCS = ("get_equity",)

CONTRACT_HELP = (
    "Add RETURNS, get_returns(), EQUITY, or get_equity().\n"
    "PERIODS_PER_YEAR and LABEL are optional."
)


class StrategyError(RuntimeError):
    pass


@dataclass
class LoadedStrategy:
    name: str
    returns: np.ndarray
    source_path: Path
    periods_per_year: float | None = None
    kind: str = "returns"


def _to_array(value, where: str) -> np.ndarray:
    if value is None:
        raise StrategyError(f"{where} is None")
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2:
        if 1 not in arr.shape:
            raise StrategyError(
                f"{where} is {arr.shape}; a single column or row is required"
            )
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise StrategyError(f"{where} must be 1-D, got {arr.ndim} dimensions")
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise StrategyError(f"{where} has fewer than 2 usable values")
    return arr


def _import_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"strategy_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise StrategyError(f"could not read {path.name} as a Python module")
    module = importlib.util.module_from_spec(spec)

    folder = str(path.parent)
    added = folder not in sys.path
    if added:
        sys.path.insert(0, folder)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise StrategyError(f"{type(exc).__name__} while running "
                            f"{path.name}: {exc}") from exc
    finally:
        if added:
            try:
                sys.path.remove(folder)
            except ValueError:
                pass
    return module


def load_strategy(path: str | Path) -> LoadedStrategy:
    path = Path(path)
    if not path.exists():
        raise StrategyError(f"{path} does not exist")

    if path.suffix.lower() in {".csv", ".parquet", ".pq"}:
        return LoadedStrategy(name=path.stem, returns=load_returns(path),
                              source_path=path, kind="file")
    if path.suffix.lower() != ".py":
        raise StrategyError(f"unsupported file type {path.suffix!r}; "
                            "drop a .py strategy or a .csv of returns")

    module = _import_module(path)

    for attr in RETURN_NAMES:
        if hasattr(module, attr):
            returns = _to_array(getattr(module, attr), f"{path.name}:{attr}")
            kind = attr
            break
    else:
        for attr in RETURN_FUNCS + EQUITY_NAMES + EQUITY_FUNCS:
            if not hasattr(module, attr):
                continue
            raw = getattr(module, attr)
            if attr in RETURN_FUNCS or attr in EQUITY_FUNCS:
                if not callable(raw):
                    raise StrategyError(f"{path.name}:{attr} is not callable")
                try:
                    raw = raw()
                except Exception as exc:
                    raise StrategyError(f"{type(exc).__name__} in "
                                        f"{path.name}:{attr}(): {exc}") from exc
            values = _to_array(raw, f"{path.name}:{attr}")
            returns = values if attr in RETURN_FUNCS else equity_to_returns(values)
            kind = attr
            break
        else:
            raise StrategyError(f"No returns found in {path.name}.\n{CONTRACT_HELP}")

    if kind in RETURN_NAMES + RETURN_FUNCS and np.nanmax(np.abs(returns)) > 2.0:
        raise StrategyError(
            f"Values in {path.name} look like whole percentages. "
            "Please divide by 100, or use EQUITY for an equity curve."
        )

    ppy = getattr(module, "PERIODS_PER_YEAR", None)
    label = getattr(module, "LABEL", None)
    return LoadedStrategy(
        name=str(label) if label else path.stem,
        returns=returns,
        source_path=path,
        periods_per_year=float(ppy) if ppy else None,
        kind=kind,
    )
