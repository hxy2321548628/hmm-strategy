"""Map training states to next-session returns, then decide long/cash exposure."""

import numpy as np
import pandas as pd

from benke.config.enums import ThresholdMode
from benke.config.settings import SignalConfig


def cash_states(
    states: np.ndarray, returns: pd.Series, n_states: int, config: SignalConfig
) -> set[int]:
    next_returns = returns.shift(-1).to_numpy()
    threshold = config.cumulative_threshold
    if config.threshold_mode == ThresholdMode.BENCHMARK:
        threshold = float(np.prod(1 + next_returns[np.isfinite(next_returns)]))
    selected = set()
    for state in range(n_states):
        sample = next_returns[(states == state) & np.isfinite(next_returns)]
        # Unobserved states have no evidence supporting long exposure.
        if len(sample) == 0 or float(np.prod(1 + sample)) < threshold:
            selected.add(state)
    return selected
