import numpy as np

LABEL = "example: 60/40-ish with fat tails"
PERIODS_PER_YEAR = 252


def get_returns():
    rng = np.random.default_rng(20260815)
    mu = 0.09 / PERIODS_PER_YEAR
    sigma = 0.14 / np.sqrt(PERIODS_PER_YEAR)

    shocks = rng.standard_t(4, size=1260)
    shocks /= np.sqrt(4 / (4 - 2))

    smoothed = np.empty_like(shocks)
    smoothed[0] = shocks[0]
    for i in range(1, shocks.size):
        smoothed[i] = 0.08 * smoothed[i - 1] + np.sqrt(1 - 0.08**2) * shocks[i]

    return mu + sigma * smoothed
