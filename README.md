-Monte Carlo strategy tester

I made this small Python app to get a better feel for the risk in a trading
strategy. It takes a history of returns, rearranges it into thousands of
possible paths, and shows how those paths might turn out.

It is useful for exploring drawdowns, losses, and recovery time. It is not a
prediction of future performance.
-Quick start

You need Python 3.10 or newer.

To run it manually:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python mc_ui.py
```

You can also try the command-line version with the sample data:

```bash
python run_mc.py --input examples/data/example_returns.csv --column ret
```

-Using your own data

This app accepts CSV, Parquet, and Python files. drag a file onto the window or
use the Browse button.

A CSV file can contain returns or an equity curve. For a Python strategy file,
provide one of these values:

```python
RETURNS = [0.01, -0.005, 0.012]
```

```python
EQUITY = [100_000, 101_000, 100_495, 101_701]
```

Functions named `get_returns()` and `get_equity()` work too. See
`examples/example_strategy.py` for an example.

Parquet support needs one extra package:

```bash
python -m pip install pyarrow
```

-Results

Each run is saved in `results/<name>/`. you get a json summary and charts for
equity paths, drawdowns, ending equity, and recovery time.

The default sampling method keeps some of the short-term patterns in the
original returns. You can change it in the app or with the `--method` command-line
option.

-Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Disclaimer: A quick warning

The results are only as useful as the data you put in. A backtest can contain
bias, and future markets may behave very differently. Trading costs and
slippage should already be included in your returns.
