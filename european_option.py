# pricing_european.py

import math
from typing import Literal

OptionKind = Literal["call", "put"]

def _norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function N(0, 1) using math.erf.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_european(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionKind = "call",
) -> float:
    """
    Price a European call or put option with the Black–Scholes model.

    Notes
    -----
    This function assumes that all inputs have already been validated
    in the main script. It does not perform any error checking.
    """
    # Black–Scholes d1 and d2 (no internal validation, main already checked inputs)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    opt = option_type.lower()
    if opt == "call":
        price = S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    elif opt == "put":
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)
    else:
        # We rely on main for type checking; this is just a defensive fallback.
        raise ValueError("option_type must be 'call' or 'put'.")

    return price
