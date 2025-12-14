# asian_option.py
#
# Single-function Monte Carlo pricer for Asian options.
# All input validation is performed in main.py.
# This module only contains numerical logic.

from typing import Literal, Optional
import math
import numpy as np

OptionTypeStr = Literal["CALL", "PUT"]
AverageType = Literal["ARITH", "GEOM"]
AverageOnType = Literal["PRICE", "STRIKE"]


def price_asian_option_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int,
    dt: float,
    option_type: OptionTypeStr,
    avg_type: AverageType = "ARITH",
    avg_on: AverageOnType = "PRICE",
    random_seed: Optional[int] = None,
) -> float:
    """
    Monte Carlo pricing of an Asian option in the Black–Scholes model.

    Features
    --------
    - Arithmetic or geometric average of the underlying.
    - Average on price ("PRICE"):   payoff depends on the average and fixed strike K.
    - Average on strike ("STRIKE"): payoff depends on S_T and the average as strike.

    Dynamics under the risk–neutral measure:
        dS_t = r S_t dt + sigma S_t dW_t

    Notes
    -----
    * All input validation (ranges, types, values) is performed in main.py.
      This function assumes arguments are already consistent.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Number of time steps (at least 1)
    n_steps = max(1, int(math.ceil(T / dt)))
    dt_eff = T / n_steps  # adjust dt so that n_steps * dt_eff = T

    payoffs = np.empty(n_sims, dtype=float)

    drift = (r - 0.5 * sigma * sigma) * dt_eff
    diffusion = sigma * math.sqrt(dt_eff)

    is_call = (option_type.upper() == "CALL")

    for sim in range(n_sims):
        s_t = S0

        sum_prices = 0.0
        sum_log_prices = 0.0

        for _ in range(n_steps):
            z = np.random.normal(0.0, 1.0)
            s_t = s_t * math.exp(drift + diffusion * z)

            if avg_type == "ARITH":
                sum_prices += s_t
            else:  # "GEOM"
                sum_log_prices += math.log(s_t)

        # Last simulated price
        s_T = s_t

        # Average level
        if avg_type == "ARITH":
            avg_value = sum_prices / n_steps
        else:  # "GEOM"
            avg_value = math.exp(sum_log_prices / n_steps)

        # Effective underlying and strike depending on avg_on
        if avg_on == "PRICE":
            # Average price option: average is the price, K is fixed strike
            underlying_effective = avg_value
            strike_effective = K
        else:
            # Average strike option: S_T is underlying, average is strike
            underlying_effective = s_T
            strike_effective = avg_value

        # Payoff according to CALL/PUT
        if is_call:
            payoff = max(underlying_effective - strike_effective, 0.0)
        else:
            payoff = max(strike_effective - underlying_effective, 0.0)

        payoffs[sim] = payoff

    mean_payoff = float(np.mean(payoffs))
    price = math.exp(-r * T) * mean_payoff
    return price

      
