"""Draft Surplus Valuation Framework for NFL Front Offices.

Models the non-linear relationship between draft capital investment and career production:
    E[Snaps | Pick] = A * exp(-lambda * Pick) + C
Quantifies whether an athlete generates surplus value over their draft pick baseline.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Empirical baseline parameters estimated across NFL draft classes
DEFAULT_EXP_A = 1853.6
DEFAULT_EXP_LAMBDA = 0.0094
DEFAULT_EXP_C = 17.7

# Approximate NFL salary cap market rate per regular season snap by position
POSITION_CAP_PER_SNAP = {
    "DE": 16000.0,
    "EDGE": 16000.0,
    "OLB": 14000.0,
    "DT": 14000.0,
    "WR": 13500.0,
    "T": 15000.0,
    "OT": 15000.0,
    "CB": 12000.0,
    "G": 10500.0,
    "OG": 10500.0,
    "C": 10000.0,
    "TE": 9500.0,
    "SS": 9000.0,
    "FS": 9000.0,
    "DEFAULT": 10000.0,
}


def decay_function(
    pick: Union[float, np.ndarray], A: float, lambda_val: float, C: float
) -> Union[float, np.ndarray]:
    """Parametric exponential decay function."""
    return A * np.exp(-lambda_val * np.asarray(pick)) + C


def expected_career_snaps(
    pick_number: Union[int, float, np.ndarray, pd.Series],
    A: float = DEFAULT_EXP_A,
    lambda_val: float = DEFAULT_EXP_LAMBDA,
    C: float = DEFAULT_EXP_C,
) -> Union[float, np.ndarray]:
    """Calculate the expected career snaps for a given overall draft pick.

    Args:
        pick_number: Overall draft pick (1 to 260)
        A: Amplitude parameter (default: 1853.6)
        lambda_val: Exponential decay rate (default: 0.0094)
        C: Asymptotic baseline constant (default: 17.7)

    Returns:
        Expected snaps as float or numpy array.
    """
    pick_arr = np.asarray(pick_number, dtype=np.float64)
    # Undrafted free agents or missing picks assigned nominal pick 260
    pick_clean = np.where(np.isnan(pick_arr) | (pick_arr <= 0), 260.0, pick_arr)
    expected = decay_function(pick_clean, A, lambda_val, C)
    if np.isscalar(pick_number):
        return float(expected.item())
    return expected


def calculate_surplus_snaps(
    actual_snaps: Union[int, float, np.ndarray, pd.Series],
    pick_number: Union[int, float, np.ndarray, pd.Series],
    A: float = DEFAULT_EXP_A,
    lambda_val: float = DEFAULT_EXP_LAMBDA,
    C: float = DEFAULT_EXP_C,
) -> Union[float, np.ndarray]:
    """Calculate career surplus snaps over expected draft capital baseline.

    Positive surplus indicates the prospect outperformed their draft position.

    Args:
        actual_snaps: Realized career snaps
        pick_number: Overall draft pick
        A, lambda_val, C: Exponential decay parameters

    Returns:
        Surplus snaps (actual - expected)
    """
    actual_arr = np.asarray(actual_snaps, dtype=np.float64)
    expected = expected_career_snaps(pick_number, A, lambda_val, C)
    surplus = actual_arr - expected
    if np.isscalar(actual_snaps) and np.isscalar(pick_number):
        return float(surplus.item())
    return surplus


def fit_draft_decay_curve(
    picks: np.ndarray, snaps: np.ndarray
) -> Tuple[float, float, float, Dict[str, float]]:
    """Fit the exponential decay curve to empirical draft cohort data.

    Args:
        picks: Array of draft overall picks
        snaps: Array of career snaps

    Returns:
        Tuple of (A, lambda_val, C, metrics_dict)
    """
    mask = (~np.isnan(picks)) & (~np.isnan(snaps)) & (picks > 0)
    p_valid = picks[mask].astype(float)
    s_valid = snaps[mask].astype(float)

    if len(p_valid) < 5:
        return DEFAULT_EXP_A, DEFAULT_EXP_LAMBDA, DEFAULT_EXP_C, {"r2": 0.0, "rmse": 0.0}

    # Initial parameter guesses: A ~ max snaps, lambda ~ 0.01, C ~ min snaps
    p0 = [1800.0, 0.01, 20.0]
    bounds = ([100.0, 1e-4, 0.0], [5000.0, 0.1, 500.0])

    try:
        popt, _ = curve_fit(decay_function, p_valid, s_valid, p0=p0, bounds=bounds, maxfev=5000)
        A_fit, lambda_fit, C_fit = popt
        pred = decay_function(p_valid, A_fit, lambda_fit, C_fit)
        ss_tot = np.sum((s_valid - np.mean(s_valid)) ** 2)
        ss_res = np.sum((s_valid - pred) ** 2)
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-6))
        rmse = np.sqrt(np.mean((s_valid - pred) ** 2))
        return float(A_fit), float(lambda_fit), float(C_fit), {"r2": float(r2), "rmse": float(rmse)}
    except Exception:
        return DEFAULT_EXP_A, DEFAULT_EXP_LAMBDA, DEFAULT_EXP_C, {"r2": 0.0, "rmse": 0.0}


def calculate_financial_surplus(
    surplus_snaps: Union[float, np.ndarray, pd.Series],
    position: Union[str, np.ndarray, pd.Series],
    custom_cap_rate: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """Convert career surplus snaps into estimated financial dollar value ($).

    Args:
        surplus_snaps: Career surplus snaps
        position: Player position string or array of strings
        custom_cap_rate: Optional fixed dollar rate per snap

    Returns:
        Estimated financial surplus in dollars.
    """
    if custom_cap_rate is not None:
        rate = custom_cap_rate
    elif isinstance(position, str):
        rate = POSITION_CAP_PER_SNAP.get(position.upper(), POSITION_CAP_PER_SNAP["DEFAULT"])
    else:
        pos_list = [str(p).upper() for p in position]
        rate = np.array(
            [POSITION_CAP_PER_SNAP.get(p, POSITION_CAP_PER_SNAP["DEFAULT"]) for p in pos_list]
        )

    return surplus_snaps * rate
