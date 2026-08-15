from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from mc.strategy import StrategyError, load_strategy


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_returns_constant(tmp_path):
    p = write(tmp_path, "s.py", "RETURNS = [0.01, -0.02, 0.03]\n")
    out = load_strategy(p)
    assert out.returns == pytest.approx([0.01, -0.02, 0.03])
    assert out.kind == "RETURNS"
    assert out.name == "s"


def test_lowercase_returns_also_works(tmp_path):
    p = write(tmp_path, "s.py", "returns = [0.01, 0.02]\n")
    assert load_strategy(p).returns == pytest.approx([0.01, 0.02])


def test_get_returns_function(tmp_path):
    p = write(tmp_path, "s.py", "def get_returns():\n    return [0.01, 0.02, -0.01]\n")
    out = load_strategy(p)
    assert out.kind == "get_returns"
    assert out.returns.size == 3


def test_equity_is_converted_to_returns(tmp_path):
    p = write(tmp_path, "s.py", "EQUITY = [100, 110, 99]\n")
    out = load_strategy(p)
    assert out.returns == pytest.approx([0.1, -0.1])
    assert out.kind == "EQUITY"


def test_get_equity_function(tmp_path):
    p = write(tmp_path, "s.py", "def get_equity():\n    return [100, 200]\n")
    assert load_strategy(p).returns == pytest.approx([1.0])


def test_returns_take_priority_over_equity(tmp_path):
    p = write(tmp_path, "s.py", "RETURNS = [0.05, 0.05]\nEQUITY = [1, 2, 3]\n")
    assert load_strategy(p).kind == "RETURNS"


def test_optional_metadata_is_picked_up(tmp_path):
    p = write(tmp_path, "s.py",
              "RETURNS = [0.01, 0.01]\nLABEL = 'my alpha'\nPERIODS_PER_YEAR = 52\n")
    out = load_strategy(p)
    assert out.name == "my alpha"
    assert out.periods_per_year == 52.0


def test_numpy_and_pandas_objects_are_accepted(tmp_path):
    p = write(tmp_path, "np.py", "import numpy as np\nRETURNS = np.array([0.01, 0.02])\n")
    assert load_strategy(p).returns.size == 2
    p2 = write(tmp_path, "pd.py",
               "import pandas as pd\nRETURNS = pd.Series([0.01, 0.02, 0.03])\n")
    assert load_strategy(p2).returns.size == 3


def test_single_column_frame_is_flattened(tmp_path):
    p = write(tmp_path, "s.py",
              "import pandas as pd\nRETURNS = pd.DataFrame({'r': [0.01, 0.02]})\n")
    assert load_strategy(p).returns == pytest.approx([0.01, 0.02])


def test_non_finite_values_are_dropped(tmp_path):
    p = write(tmp_path, "s.py",
              "import numpy as np\nRETURNS = [0.01, np.nan, 0.02, np.inf]\n")
    assert load_strategy(p).returns == pytest.approx([0.01, 0.02])


def test_a_strategy_may_import_its_own_neighbours(tmp_path):
    write(tmp_path, "helper.py", "VALUES = [0.01, 0.02, 0.03]\n")
    p = write(tmp_path, "s.py", "from helper import VALUES\nRETURNS = VALUES\n")
    assert load_strategy(p).returns.size == 3


def test_missing_contract_explains_itself(tmp_path):
    p = write(tmp_path, "s.py", "X = 1\n")
    with pytest.raises(StrategyError, match="RETURNS"):
        load_strategy(p)


def test_strategy_exception_is_reported_with_its_message(tmp_path):
    p = write(tmp_path, "s.py", "raise ValueError('data file missing')\n")
    with pytest.raises(StrategyError, match="data file missing"):
        load_strategy(p)


def test_percent_quoted_returns_are_refused(tmp_path):
    p = write(tmp_path, "s.py", "RETURNS = [1.5, -2.5]\n")
    with pytest.raises(StrategyError, match="divide by 100"):
        load_strategy(p)


def test_too_few_values(tmp_path):
    p = write(tmp_path, "s.py", "RETURNS = [0.01]\n")
    with pytest.raises(StrategyError, match="fewer than 2"):
        load_strategy(p)


def test_two_dimensional_returns_are_refused(tmp_path):
    p = write(tmp_path, "s.py",
              "import numpy as np\nRETURNS = np.ones((3, 4)) * 0.01\n")
    with pytest.raises(StrategyError, match="single column"):
        load_strategy(p)


def test_unsupported_extension(tmp_path):
    p = write(tmp_path, "notes.txt", "hello\n")
    with pytest.raises(StrategyError, match="unsupported file type"):
        load_strategy(p)


def test_missing_file(tmp_path):
    with pytest.raises(StrategyError, match="does not exist"):
        load_strategy(tmp_path / "nope.py")


def test_csv_of_returns_is_accepted(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("date,ret\n2020-01-01,0.01\n2020-01-02,-0.02\n", encoding="utf-8")
    out = load_strategy(p)
    assert out.kind == "file"
    assert out.returns == pytest.approx([0.01, -0.02])


def test_runner_produces_a_full_result_set(tmp_path):
    strategy = write(tmp_path, "s.py",
                     "import numpy as np\n"
                     "RETURNS = np.random.default_rng(0).normal(0.0004, 0.01, 400)\n")
    out_dir = tmp_path / "out"
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "path": str(strategy),
        "config": {"n_paths": 200, "horizon": 40, "seed": 1},
        "sweep_levels": None,
        "out_dir": str(out_dir),
        "themes": ["light"],
    }), encoding="utf-8")

    proc = subprocess.run([sys.executable, str(ROOT / "mc_runner.py"), str(job)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr

    report = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "report.txt").exists()
    assert (out_dir / "dashboard.png").exists()
    assert "PROGRESS" in proc.stdout


def test_runner_reports_a_broken_strategy_without_crashing(tmp_path):
    strategy = write(tmp_path, "bad.py", "X = 1\n")
    out_dir = tmp_path / "out"
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "path": str(strategy),
        "config": {"n_paths": 50, "horizon": 10},
        "sweep_levels": None,
        "out_dir": str(out_dir),
        "themes": ["light"],
    }), encoding="utf-8")

    proc = subprocess.run([sys.executable, str(ROOT / "mc_runner.py"), str(job)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1
    report = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert "RETURNS" in report["error"]
