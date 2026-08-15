from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Method = Literal["iid", "stationary", "block", "normal", "t"]
RuinBasis = Literal["initial", "peak"]


@dataclass
class SimConfig:
    method: Method = "stationary"
    n_paths: int = 10_000
    horizon: int = 252
    block: float | None = None
    seed: int = 0

    initial_capital: float = 100_000.0
    leverage: float = 1.0
    compound: bool = True

    ruin_threshold: float = 0.5
    ruin_basis: RuinBasis = "initial"
    stop_at_ruin: bool = True

    periods_per_year: float = 252.0
    fan_paths: int = 5_000
    chunk_elements: int = 4_000_000

    t_df: float = 4.0

    label: str = "run"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not 0.0 <= self.ruin_threshold < 1.0:
            raise ValueError("ruin_threshold must be in [0, 1)")
        if self.leverage <= 0:
            raise ValueError("leverage must be > 0")
        if self.block is not None and self.block <= 0:
            raise ValueError("block must be > 0")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")

    @property
    def chunk_size(self) -> int:
        return max(1, min(self.n_paths, self.chunk_elements // self.horizon))

    def to_dict(self) -> dict:
        return asdict(self)
