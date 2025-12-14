# Monte_Carlo_option_americane_1.py

import numpy as np
from typing import Optional, Sequence, Tuple


def american_option_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_sims: int,
    is_call: bool = True,
    dividends: Optional[Sequence[Tuple[float, float]]] = None,
) -> float:
    """
    Longstaff–Schwartz Monte Carlo pricing for an American option
    with optional discrete dividends.

    Notes
    -----
    All parameter validation is done in the main script.
    This function assumes that inputs are consistent and valid.
    """
    dt = T / n_steps

    # Map dividend times to step indices
    dividend_steps = {}
    if dividends is not None:
        for t_div, amount in dividends:
            step = int(t_div / T * n_steps)
            dividend_steps[step] = amount
    # At each dividend step we subtract the cash amount from the simulated spot.

    # Simulate paths under the risk–neutral measure
    S = np.zeros((n_sims, n_steps + 1), dtype=float)
    S[:, 0] = S0

    for t in range(1, n_steps + 1):
        z = np.random.normal(0.0, 1.0, size=n_sims)
        S[:, t] = S[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )
        if t in dividend_steps:
            # Apply dividend as a cash drop on the spot
            S[:, t] -= dividend_steps[t]

    # Payoff matrix (paths x time)
    if is_call:
        payoff = np.maximum(S - K, 0.0)
    else:
        payoff = np.maximum(K - S, 0.0)

    # Backward induction: Longstaff–Schwartz
    cashflow = payoff[:, -1].copy()

    for t in range(n_steps - 1, 0, -1):
        in_the_money = payoff[:, t] > 0.0
        if not np.any(in_the_money):
            cashflow *= np.exp(-r * dt)
            continue

        X = S[in_the_money, t]
        Y = cashflow[in_the_money] * np.exp(-r * dt)
        # Y is an estimate of the continuation value.

        # Polynomial regression (quadratic) for continuation value
        coeffs = np.polyfit(X, Y, deg=2)
        continuation = np.polyval(coeffs, X)

        exercise = payoff[in_the_money, t] > continuation
        idx = np.where(in_the_money)[0]

        cashflow[idx[exercise]] = payoff[idx[exercise], t]
        cashflow[idx[~exercise]] = Y[~exercise]

        # Discount one step back for all paths
        cashflow *= np.exp(-r * dt)

    return float(np.mean(cashflow))
