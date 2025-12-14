# main.py
#
# This script:
#   - reads option data from an Excel file,
#   - validates inputs (only here: all raises are in this file),
#   - routes each row to the correct pricer:
#         EUROPEAN, ASIAN, AMERICAN, BINARY,
#   - computes Greeks via bump & revalue,
#   - collects results and saves them to an output Excel file.
#
# All prints and comments are in English.

import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from european_option import black_scholes_european
from american_option import american_option_mc
from asian_option import price_asian_option_mc
from binary_option import price_binary_bs


# =====================================================================
#  1) VALIDATION & PARSING HELPERS (COMMON TO ALL STYLES)
# =====================================================================

class RowValidationError(Exception):
    """Validation error for a specific Excel row."""
    pass


def _parse_float(row: pd.Series, col: str, row_id: Any) -> float:
    """
    Parse a float from the row and raise a clear error if missing or invalid.
    """
    if col not in row:
        raise RowValidationError(f"Missing required column '{col}'.")
    try:
        value = float(row[col])
    except Exception:
        raise RowValidationError(f"Column '{col}' must be a number (row id={row_id}).")
    return value


def _parse_optional_float(row: pd.Series, col: str, default: float) -> float:
    """
    Parse an optional float; if the column is missing or NaN, returns default.
    """
    if col not in row or pd.isna(row[col]):
        return default
    try:
        return float(row[col])
    except Exception:
        return default


def _parse_string(row: pd.Series, col: str, row_id: Any) -> str:
    """
    Parse a non-empty string from the row.
    """
    if col not in row:
        raise RowValidationError(f"Missing required column '{col}'.")
    value = str(row[col]).strip()
    if not value:
        raise RowValidationError(f"Column '{col}' cannot be empty (row id={row_id}).")
    return value


def validate_common_params(
    row_id: Any,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str,
    style: str,
) -> None:
    """
    Validate common parameters used by all option styles.
    All error checks are centralized here (not in pricer modules).
    """
    if S0 <= 0.0:
        raise RowValidationError(f"S0 must be > 0 (row id={row_id}).")
    if K <= 0.0:
        raise RowValidationError(f"K must be > 0 (row id={row_id}).")
    if sigma <= 0.0:
        raise RowValidationError(f"sigma must be > 0 (row id={row_id}).")
    if T <= 0.0:
        raise RowValidationError(f"T must be > 0 (row id={row_id}).")

    opt = option_type.upper()
    if opt not in ("CALL", "PUT"):
        raise RowValidationError(
            f"option_type must be CALL or PUT, got '{option_type}' (row id={row_id})."
        )

    st = style.upper()
    if st not in ("EUROPEAN", "ASIAN", "AMERICAN", "BINARY"):
        raise RowValidationError(
            f"style must be one of EUROPEAN / ASIAN / AMERICAN / BINARY, got '{style}' (row id={row_id})."
        )

    # Basic sanity checks (you can tighten these if you want)
    if T > 50.0:
        raise RowValidationError(
            f"T seems too large (>50 years). Check units (row id={row_id})."
        )
    if abs(r) > 1.0 or abs(q) > 1.0:
        raise RowValidationError(
            f"r or q seem too large in absolute value (>1). Check units (row id={row_id})."
        )


def parse_common_fields(row: pd.Series) -> Dict[str, Any]:
    """
    Read and validate all fields that are common to every style.
    Returns a dict with normalized values.
    No financial quantities (no d1, d2, etc.) are computed here.
    """
    row_id = row.get("id", "N/A")

    style = _parse_string(row, "style", row_id).upper()
    option_type = _parse_string(row, "option_type", row_id).upper()

    S0 = _parse_float(row, "S0", row_id)
    K = _parse_float(row, "K", row_id)
    r = _parse_float(row, "r", row_id)
    sigma = _parse_float(row, "sigma", row_id)
    T = _parse_float(row, "T", row_id)
    q = _parse_optional_float(row, "q", 0.0)  # dividend yield optional

    validate_common_params(row_id, S0, K, r, q, sigma, T, option_type, style)

    return {
        "row_id": row_id,
        "style": style,
        "option_type": option_type,
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "q": q,
    }


# =====================================================================
#  2) STYLE-SPECIFIC PRICERS (MAIN ONLY ROUTES, NO CALCULATION HERE)
# =====================================================================

def price_european(common: Dict[str, Any]) -> float:
    """
    Price a European option using Black–Scholes.
    Common parameters are already validated and in 'common'.
    All mathematical details (d1, d2, N, etc.) are inside the pricer.
    """
    price = black_scholes_european(
        S0=common["S0"],
        K=common["K"],
        T=common["T"],
        r=common["r"],
        sigma=common["sigma"],
        option_type=common["option_type"].lower(),  # pricer expects "call"/"put"
    )
    return float(price)


def price_asian(common: Dict[str, Any], row: pd.Series) -> float:
    """
    Price an Asian option via Monte Carlo using a single pricing function.
    All validation of inputs is done here in main.py.
    """
    row_id = common["row_id"]

    # Additional fields for Asian options
    raw_avg_type = row.get("avg_type", "ARITH")
    avg_type = str(raw_avg_type).strip().upper() or "ARITH"
    if avg_type not in ("ARITH", "GEOM"):
        raise RowValidationError(
            f"avg_type must be ARITH or GEOM, got '{raw_avg_type}' (row id={row_id})."
        )

    raw_avg_on = row.get("avg_on", "PRICE")
    avg_on = str(raw_avg_on).strip().upper() or "PRICE"
    if avg_on not in ("PRICE", "STRIKE"):
        raise RowValidationError(
            f"avg_on must be PRICE or STRIKE, got '{raw_avg_on}' (row id={row_id})."
        )

    n_sims = int(_parse_float(row, "n_sims", row_id))
    dt = _parse_float(row, "dt", row_id)
    if n_sims <= 0:
        raise RowValidationError(f"n_sims must be > 0 (row id={row_id}).")
    if dt <= 0.0:
        raise RowValidationError(f"dt must be > 0 (row id={row_id}).")

    # All parameters already validated; we just pass them to the pricer.
    price = price_asian_option_mc(
        S0=common["S0"],
        K=common["K"],
        r=common["r"],
        sigma=common["sigma"],
        T=common["T"],
        n_sims=n_sims,
        dt=dt,
        option_type=common["option_type"],  # "CALL" or "PUT"
        avg_type=avg_type,                  # "ARITH" or "GEOM"
        avg_on=avg_on,                      # "PRICE" or "STRIKE"
    )

    return float(price)


def _parse_dividends(row: pd.Series, row_id: Any, T: float) -> Optional[List[Tuple[float, float]]]:
    """
    Optional helper for American options.

    Expects two optional columns:
      - div_times:   e.g. "0.25, 0.5"
      - div_amounts: e.g. "1.0, 0.5"

    Both lists must have the same length if provided.
    All validations for dividend data are done here.
    """
    if "div_times" not in row or pd.isna(row["div_times"]):
        return None
    times_str = str(row["div_times"]).strip()
    if not times_str:
        return None

    amounts_str = str(row.get("div_amounts", "")).strip()
    if not amounts_str:
        raise RowValidationError(
            f"div_times is provided but div_amounts is missing/empty (row id={row_id})."
        )

    times_parts = [x.strip() for x in times_str.split(",") if x.strip()]
    amounts_parts = [x.strip() for x in amounts_str.split(",") if x.strip()]

    if len(times_parts) != len(amounts_parts):
        raise RowValidationError(
            f"div_times and div_amounts must have the same number of elements (row id={row_id})."
        )

    dividends: List[Tuple[float, float]] = []
    for t_str, a_str in zip(times_parts, amounts_parts):
        try:
            t_val = float(t_str)
            a_val = float(a_str)
        except Exception:
            raise RowValidationError(
                f"Invalid dividend time/amount '{t_str}', '{a_str}' (row id={row_id})."
            )
        if t_val <= 0.0 or t_val >= T:
            raise RowValidationError(
                f"Dividend time {t_val} must be in (0, T) (row id={row_id})."
            )
        dividends.append((t_val, a_val))

    return dividends


def price_american(common: Dict[str, Any], row: pd.Series) -> float:
    """
    Price an American option via Longstaff–Schwartz Monte Carlo.
    All error checking is done here; the pricer only does math.
    """
    row_id = common["row_id"]

    n_steps = int(_parse_float(row, "n_steps", row_id))
    n_sims = int(_parse_float(row, "n_sims", row_id))
    if n_steps <= 0:
        raise RowValidationError(f"n_steps must be > 0 (row id={row_id}).")
    if n_sims <= 0:
        raise RowValidationError(f"n_sims must be > 0 (row id={row_id}).")

    dividends = _parse_dividends(row, row_id, common["T"])

    is_call = (common["option_type"] == "CALL")

    price = american_option_mc(
        S0=common["S0"],
        K=common["K"],
        r=common["r"],
        sigma=common["sigma"],
        T=common["T"],
        n_steps=n_steps,
        n_sims=n_sims,
        is_call=is_call,
        dividends=dividends,
    )
    return float(price)


def price_binary(common: Dict[str, Any], row: pd.Series) -> float:
    """
    Price a European binary option (cash-or-nothing / asset-or-nothing).
    All checks are done in the main.
    """
    row_id = common["row_id"]

    raw_btype = row.get("binary_type", "cash")
    binary_type = str(raw_btype).strip().lower() or "cash"
    if binary_type not in ("cash", "asset"):
        raise RowValidationError(
            f"binary_type must be 'cash' or 'asset', got '{raw_btype}' (row id={row_id})."
        )

    payout = _parse_optional_float(row, "payout", 1.0)
    if payout <= 0.0:
        raise RowValidationError(f"payout must be > 0 (row id={row_id}).")

    price = price_binary_bs(
        S0=common["S0"],
        K=common["K"],
        r=common["r"],
        q=common["q"],
        sigma=common["sigma"],
        T=common["T"],
        option_type=common["option_type"].lower(),
        binary_type=binary_type,
        payout=payout,
    )
    return float(price)


# =====================================================================
#  3) GREEKS (BUMP & REVALUE USING THE EXISTING PRICERS)
# =====================================================================

def greeks_universal_bump_revalue(
    common: Dict[str, Any],
    row: pd.Series,
    bump_rel: float = 1e-4,
) -> Tuple[float, float, float, float, float]:
    """
    Compute Delta, Gamma, Vega, Theta, Rho via bump & revalue
    using the pricing functions defined above:
        - price_european
        - price_asian
        - price_american
        - price_binary

    All input validation is assumed already done in common/row.
    """

    S0 = common["S0"]
    K = common["K"]
    r = common["r"]
    q = common["q"]
    sigma = common["sigma"]
    T = common["T"]
    style = common["style"]  # "EUROPEAN" / "ASIAN" / "AMERICAN" / "BINARY"

    # Helper that calls the appropriate pricer for the bumped parameters
    def P(S0_: float, K_: float, r_: float, q_: float, sigma_: float, T_: float) -> float:
        bumped = dict(common)
        bumped["S0"] = S0_
        bumped["K"] = K_
        bumped["r"] = r_
        bumped["q"] = q_
        bumped["sigma"] = sigma_
        bumped["T"] = T_

        if style == "EUROPEAN":
            return price_european(bumped)
        elif style == "ASIAN":
            return price_asian(bumped, row)
        elif style == "AMERICAN":
            return price_american(bumped, row)
        elif style == "BINARY":
            return price_binary(bumped, row)
        else:
            raise ValueError(f"Unsupported style '{style}' in Greeks engine.")

    # ---- base price ----
    price0 = P(S0, K, r, q, sigma, T)

    # ---- Delta & Gamma (bump S0) ----
    h_S = bump_rel * S0 if S0 != 0 else bump_rel
    price_up_S = P(S0 + h_S, K, r, q, sigma, T)
    price_dn_S = P(max(S0 - h_S, 1e-12), K, r, q, sigma, T)

    delta = (price_up_S - price_dn_S) / (2.0 * h_S)
    gamma = (price_up_S - 2.0 * price0 + price_dn_S) / (h_S ** 2)

    # ---- Vega (bump sigma) ----
    h_sigma = bump_rel * sigma if sigma != 0 else bump_rel
    price_up_sigma = P(S0, K, r, q, sigma + h_sigma, T)
    price_dn_sigma = P(S0, K, r, q, max(sigma - h_sigma, 1e-12), T)

    vega = (price_up_sigma - price_dn_sigma) / (2.0 * h_sigma)

    # ---- Rho (bump r) ----
    h_r = bump_rel * max(abs(r), 1.0)
    price_up_r = P(S0, K, r + h_r, q, sigma, T)
    price_dn_r = P(S0, K, r - h_r, q, sigma, T)

    rho = (price_up_r - price_dn_r) / (2.0 * h_r)

    # ---- Theta (bump T) ----
    if T > 1e-6:
        h_T = bump_rel * T
        price_up_T = P(S0, K, r, q, sigma, T + h_T)
        price_dn_T = P(S0, K, r, q, sigma, max(T - h_T, 1e-12))
        dV_dT = (price_up_T - price_dn_T) / (2.0 * h_T)
        theta = -dV_dT
    else:
        theta = 0.0

    return float(delta), float(gamma), float(vega), float(theta), float(rho)


# =====================================================================
#  4) MAIN LOOP
# =====================================================================

def main() -> None:
    # ---- INPUT FILE NAME ----
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "input_options.xlsx"

    print(f"Reading input from '{input_path}'...")

    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"FATAL ERROR: unable to read Excel file '{input_path}': {e}")
        sys.exit(1)

    if df.empty:
        print("No data found in the input file.")
        sys.exit(0)

    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        try:
            # ---------------------------------------------------------
            # 1) Parse and validate all common parameters once
            # ---------------------------------------------------------
            common = parse_common_fields(row)

            # ---------------------------------------------------------
            # 2) Route to the correct pricer
            # ---------------------------------------------------------
            style = common["style"]
            if style == "EUROPEAN":
                price = price_european(common)
            elif style == "ASIAN":
                price = price_asian(common, row)
            elif style == "AMERICAN":
                price = price_american(common, row)
            elif style == "BINARY":
                price = price_binary(common, row)
            else:
                # This should not happen thanks to validation
                raise RowValidationError(f"Unsupported style '{style}'.")

            # ---------------------------------------------------------
            # 3) Greeks via bump & revalue (same function for all styles)
            # ---------------------------------------------------------
            delta, gamma, vega, theta, rho = greeks_universal_bump_revalue(common, row)

            # ---------------------------------------------------------
            # 4) Collect output (same structure for all styles)
            # ---------------------------------------------------------
            out_row: Dict[str, Any] = {
                "id": common["row_id"],
                "style": style,
                "option_type": common["option_type"],
                "S0": common["S0"],
                "K": common["K"],
                "r": common["r"],
                "q": common["q"],
                "sigma": common["sigma"],
                "T": common["T"],
                "price": price,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
                "rho": rho,
            }

            results.append(out_row)
            print(
                f"[OK] Row {idx} (id={common['row_id']}): "
                f"style={style}, price={price:.6f}"
            )

        except RowValidationError as ve:
            print(f"[VALIDATION ERROR] Row {idx} (id={row.get('id','N/A')}): {ve}")
        except Exception as e:
            print(f"[ERROR] Row {idx} (id={row.get('id','N/A')}): {e}")

    # =================================================================
    #  5) SAVE OUTPUT
    # =================================================================
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_excel("output_prices.xlsx", index=False)
        print("\nSaved results to 'output_prices.xlsx'.")
    else:
        print("\nNo valid rows to save.")


if __name__ == "__main__":
    main()
