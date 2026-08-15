from __future__ import annotations

import numpy as np
import pytest

from mc import SimConfig, simulate
from mc.data import equity_to_returns, load_returns, synthetic_returns
from mc.engine import (apply_ruin_barrier, build_equity, cagr, max_drawdown,
                       max_underwater, realised_returns, sweep_leverage,
                       time_underwater)
from mc.report import render_text, summarise
from mc.resample import auto_block_length, resample


@pytest.fixture
def sample() -> np.ndarray:
    return synthetic_returns(750, 0.10, 0.15, dist="t", autocorr=0.1, seed=42)


@pytest.mark.parametrize("method", ["iid", "stationary", "block", "normal", "t"])
def test_resample_shape(sample, method):
    out = resample(sample, 7, 33, method=method, rng=np.random.default_rng(0))
    assert out.shape == (7, 33)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("method", ["iid", "stationary", "block"])
def test_bootstrap_only_reuses_observed_values(sample, method):
    out = resample(sample, 20, 50, method=method, rng=np.random.default_rng(1))
    assert np.isin(out, sample).all()


def test_resample_is_deterministic_given_a_seed(sample):
    a = resample(sample, 10, 20, rng=np.random.default_rng(5))
    b = resample(sample, 10, 20, rng=np.random.default_rng(5))
    c = resample(sample, 10, 20, rng=np.random.default_rng(6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_block_methods_retain_autocorrelation_that_iid_destroys():
    rng = np.random.default_rng(3)
    ar = np.zeros(2_000)
    for i in range(1, ar.size):
        ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 0.01)

    def lag1(paths: np.ndarray) -> float:
        x = paths - paths.mean(axis=1, keepdims=True)
        num = (x[:, :-1] * x[:, 1:]).sum(axis=1)
        den = (x * x).sum(axis=1)
        return float(np.mean(num / den))

    iid = lag1(resample(ar, 400, 500, method="iid", rng=np.random.default_rng(7)))
    stat = lag1(resample(ar, 400, 500, method="stationary", block=20,
                         rng=np.random.default_rng(7)))
    assert abs(iid) < 0.05
    assert stat > 0.35


def test_auto_block_length_reacts_to_dependence():
    rng = np.random.default_rng(11)
    white = rng.normal(0, 0.01, 1_000)
    ar = np.zeros(1_000)
    for i in range(1, ar.size):
        ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 0.01)
    assert auto_block_length(white) < 2.0
    assert auto_block_length(ar) > 5.0


def test_parametric_methods_match_sample_moments(sample):
    out = resample(sample, 4_000, 250, method="normal", rng=np.random.default_rng(2))
    assert out.mean() == pytest.approx(sample.mean(), abs=2e-4)
    assert out.std() == pytest.approx(sample.std(ddof=1), rel=0.05)


def test_resample_rejects_bad_input():
    with pytest.raises(ValueError):
        resample(np.array([0.01]), 5, 5)
    with pytest.raises(ValueError):
        resample(np.array([0.01, np.nan]), 5, 5)
    with pytest.raises(ValueError):
        resample(np.array([0.01, 0.02]), 5, 5, method="nope")


def test_compounded_equity_math():
    r = np.array([[0.1, -0.5, 0.2]])
    eq = build_equity(r, 100.0, leverage=1.0, compound=True)
    assert eq[0] == pytest.approx([100, 110, 55, 66])


def test_fixed_notional_equity_math():
    r = np.array([[0.1, -0.5, 0.2]])
    eq = build_equity(r, 100.0, leverage=1.0, compound=False)
    assert eq[0] == pytest.approx([100, 110, 60, 80])


def test_leverage_scales_period_returns():
    r = np.array([[0.1, -0.1]])
    eq = build_equity(r, 100.0, leverage=2.0, compound=True)
    assert eq[0] == pytest.approx([100, 120, 96])


def test_leverage_can_wipe_the_account_out_and_it_stays_out():
    r = np.array([[-0.6, 0.5, 0.5]])
    eq = build_equity(r, 100.0, leverage=2.0, compound=True)
    assert eq[0, 1] == 0.0
    assert np.all(eq[0, 1:] == 0.0)


def test_equity_never_goes_negative_without_compounding():
    r = np.array([[-0.8, -0.8]])
    eq = build_equity(r, 100.0, leverage=1.0, compound=False)
    assert np.all(eq >= 0.0)


def test_barrier_flags_and_freezes_the_path():
    eq = build_equity(np.array([[0.1, -0.6, 0.5]]), 100.0, 1.0, True)
    frozen, ruin = apply_ruin_barrier(eq, 100.0, 0.5, "initial", stop_at_ruin=True)
    assert ruin[0] == 2
    assert frozen[0] == pytest.approx([100, 110, 44, 44])


def test_barrier_can_be_observed_without_stopping():
    eq = build_equity(np.array([[0.1, -0.6, 0.5]]), 100.0, 1.0, True)
    kept, ruin = apply_ruin_barrier(eq, 100.0, 0.5, "initial", stop_at_ruin=False)
    assert ruin[0] == 2
    assert kept[0, 3] == pytest.approx(66.0)


def test_peak_basis_triggers_on_drawdown_not_on_starting_capital():
    eq = build_equity(np.array([[1.0, -0.55]]), 100.0, 1.0, True)
    _, initial_basis = apply_ruin_barrier(eq, 100.0, 0.5, "initial", True)
    _, peak_basis = apply_ruin_barrier(eq, 100.0, 0.5, "peak", True)
    assert initial_basis[0] == -1
    assert peak_basis[0] == 2


def test_no_path_starts_ruined():
    eq = build_equity(np.array([[0.01]]), 100.0, 1.0, True)
    _, ruin = apply_ruin_barrier(eq, 100.0, 0.99, "initial", True)
    assert ruin[0] == -1


def test_realised_returns_survive_zero_equity():
    eq = np.array([[100.0, 50.0, 0.0, 0.0]])
    out = realised_returns(eq)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx([-0.5, -1.0, 0.0])


def test_max_drawdown_on_a_hand_built_curve():
    eq = np.array([[100.0, 110.0, 44.0, 44.0]])
    assert max_drawdown(eq)[0] == pytest.approx(0.6)


def test_max_underwater_counts_the_longest_stretch_not_the_total():
    eq = np.array([[100.0, 90.0, 100.0, 80.0, 70.0, 100.0]])
    assert max_underwater(eq)[0] == 2
    assert time_underwater(eq)[0] == pytest.approx(3 / 6)


def test_monotonic_curve_has_no_drawdown():
    eq = np.array([[100.0, 101.0, 102.0]])
    assert max_drawdown(eq)[0] == 0.0
    assert max_underwater(eq)[0] == 0.0


def test_cagr_and_wipeout():
    assert cagr(np.array([[100.0, 200.0]]), periods_per_year=1.0)[0] == pytest.approx(1.0)
    assert cagr(np.array([[100.0, 0.0]]), periods_per_year=1.0)[0] == -1.0


def test_simulate_shapes_and_ranges(sample):
    cfg = SimConfig(n_paths=500, horizon=60, seed=1, fan_paths=100)
    res = simulate(sample, cfg)
    assert res.n_paths == 500
    assert res.equity_sample.shape == (100, 61)
    for key in ("cagr", "max_drawdown", "ruined", "terminal_equity"):
        assert res.metrics[key].shape == (500,)
    assert np.all(res.metrics["max_drawdown"] >= 0)
    assert np.all(res.metrics["max_drawdown"] <= 1)
    assert np.all(res.metrics["terminal_equity"] >= 0)


def test_simulate_is_reproducible(sample):
    cfg = SimConfig(n_paths=200, horizon=40, seed=99)
    a = simulate(sample, cfg).metrics["terminal_equity"]
    b = simulate(sample, cfg).metrics["terminal_equity"]
    assert np.array_equal(a, b)


def test_chunking_does_not_change_the_distribution(sample):
    one = simulate(sample, SimConfig(n_paths=20_000, horizon=50, seed=4,
                                     chunk_elements=10_000_000))
    many = simulate(sample, SimConfig(n_paths=20_000, horizon=50, seed=4,
                                      chunk_elements=5_000))
    assert many.config.chunk_size < 20_000
    a, b = one.metrics["terminal_equity"], many.metrics["terminal_equity"]
    assert a.mean() == pytest.approx(b.mean(), rel=0.01)
    assert np.percentile(a, 5) == pytest.approx(np.percentile(b, 5), rel=0.02)
    assert np.percentile(a, 95) == pytest.approx(np.percentile(b, 95), rel=0.02)


def test_ruin_probability_rises_with_leverage(sample):
    def p_ruin(lev: float) -> float:
        cfg = SimConfig(n_paths=3_000, horizon=252, seed=8, leverage=lev,
                        ruin_threshold=0.5, fan_paths=0)
        return float(simulate(sample, cfg).metrics["ruined"].mean())
    assert p_ruin(1.0) <= p_ruin(3.0) <= p_ruin(6.0)
    assert p_ruin(6.0) > 0.0


def test_ruined_paths_never_recover_when_stopping(sample):
    cfg = SimConfig(n_paths=1_000, horizon=252, seed=2, leverage=8.0,
                    ruin_threshold=0.5, stop_at_ruin=True, fan_paths=1_000)
    res = simulate(sample, cfg)
    ruined = res.metrics["ruined"].astype(bool)
    assert ruined.any()
    final = res.equity_sample[ruined[: res.equity_sample.shape[0]], -1]
    assert np.all(final <= 0.5 * cfg.initial_capital + 1e-6)


def test_zero_return_series_is_flat_and_never_ruins():
    flat = np.zeros(300)
    res = simulate(flat, SimConfig(n_paths=100, horizon=50, seed=1))
    assert np.allclose(res.metrics["terminal_equity"], 100_000.0)
    assert res.metrics["ruined"].sum() == 0
    assert np.allclose(res.metrics["max_drawdown"], 0.0)


@pytest.mark.parametrize("kwargs", [
    {"n_paths": 0}, {"horizon": 0}, {"initial_capital": 0},
    {"ruin_threshold": 1.0}, {"ruin_threshold": -0.1},
    {"leverage": 0}, {"block": 0}, {"periods_per_year": 0},
])
def test_config_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        SimConfig(**kwargs)


def test_chunk_size_respects_the_element_budget():
    cfg = SimConfig(n_paths=10_000, horizon=1_000, chunk_elements=100_000)
    assert cfg.chunk_size == 100


def test_equity_to_returns_roundtrip():
    eq = np.array([100.0, 110.0, 99.0])
    r = equity_to_returns(eq)
    assert r == pytest.approx([0.1, -0.1])
    assert 100 * np.cumprod(1 + r)[-1] == pytest.approx(99.0)


def test_load_returns_reads_a_returns_column(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("date,ret\n2020-01-01,0.01\n2020-01-02,-0.02\n", encoding="utf-8")
    assert load_returns(path, column="ret") == pytest.approx([0.01, -0.02])


def test_load_returns_converts_an_equity_curve(tmp_path):
    path = tmp_path / "eq.csv"
    path.write_text("date,equity\n2020-01-01,100\n2020-01-02,110\n", encoding="utf-8")
    assert load_returns(path, column="equity", kind="equity") == pytest.approx([0.1])


def test_load_returns_catches_percent_quoted_returns(tmp_path):
    path = tmp_path / "pct.csv"
    path.write_text("date,ret\n2020-01-01,1.5\n2020-01-02,-2.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="too large"):
        load_returns(path, column="ret", kind="returns")


def test_load_returns_reports_a_missing_column(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("date,ret\n2020-01-01,0.01\n2020-01-02,0.01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not in"):
        load_returns(path, column="nope")


def test_synthetic_autocorr_shows_up_in_the_series():
    r = synthetic_returns(5_000, autocorr=0.5, seed=3)
    x = r - r.mean()
    assert float(x[:-1] @ x[1:] / (x @ x)) > 0.35


def test_summary_is_complete_and_in_range(sample):
    res = simulate(sample, SimConfig(n_paths=500, horizon=60, seed=1))
    s = summarise(res)
    for key in ("config", "input", "observed", "distribution", "probabilities",
                "time_to_ruin", "caveats"):
        assert key in s
    assert all(0.0 <= v <= 1.0 for v in s["probabilities"].values())
    assert s["distribution"]["max_drawdown"]["p05"] <= s["distribution"]["max_drawdown"]["p95"]
    assert len(s["caveats"]) >= 4


def test_report_renders_without_a_single_ruined_path():
    res = simulate(np.zeros(300), SimConfig(n_paths=100, horizon=50, seed=1))
    text = render_text(summarise(res))
    assert "Monte Carlo" in text
    assert "ruin" in text


def test_summary_is_json_serialisable(tmp_path, sample):
    import json
    from mc.report import save
    res = simulate(sample, SimConfig(n_paths=200, horizon=40, seed=1))
    path = save(summarise(res), tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["config"]["n_paths"] == 200


def test_sweep_covers_every_level(sample):
    cfg = SimConfig(n_paths=400, horizon=60, seed=1)
    out = sweep_leverage(sample, cfg, [1.0, 2.0, 4.0], n_paths=400)
    assert out["leverage"] == [1.0, 2.0, 4.0]
    assert all(len(v) == 3 for v in out.values())
    assert out["median_max_dd"][0] < out["median_max_dd"][2]
