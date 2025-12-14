# binary.py

import math
from typing import Literal

OptionKind = Literal["call", "put"]
BinaryKind = Literal["cash", "asset"]


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def price_binary_bs(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: OptionKind = "call",   # "call" or "put"
    binary_type: BinaryKind = "cash",   # "cash" or "asset"
    payout: float = 1.0,                # cash amount for cash-or-nothing
) -> float:
    """
    Black–Scholes pricing for a European binary option.

    Notes
    -----
    All parameter validation is handled in the main script.
    """
    # Black–Scholes d1 and d2 with dividend yield q
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    opt = option_type.lower()
    btype = binary_type.lower()

    if btype == "cash":
        # Cash-or-nothing: discounted probability of finishing in-the-money
        if opt == "call":
            price = math.exp(-r * T) * _norm_cdf(d2) * payout
        elif opt == "put":
            price = math.exp(-r * T) * (1.0 - _norm_cdf(d2)) * payout
        else:
            raise ValueError("option_type must be 'call' or 'put'.")
    elif btype == "asset":
        # Asset-or-nothing: present value of underlying times probability of ITM
        if opt == "call":
            price = S0 * math.exp(-q * T) * _norm_cdf(d1)
        elif opt == "put":
            price = S0 * math.exp(-q * T) * (1.0 - _norm_cdf(d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'.")
    else:
        raise ValueError("binary_type must be 'cash' or 'asset'.")

    return price
