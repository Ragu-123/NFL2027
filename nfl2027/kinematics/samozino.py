"""Biomechanical Force-Velocity-Power Profiling (Samozino & Morin 2016).

Fits mono-exponential velocity curves v(t) = v0 * (1 - exp(-t / tau))
to horizontal sprint trajectories (40-yard dash) and derives continuous
mechanical properties:
- Theoretical maximal velocity v0 [yd/s]
- Acceleration time constant tau [s]
- Maximal horizontal acceleration a_max [yd/s^2]
- Theoretical maximal relative horizontal force f0 [N/kg]
- Maximal relative mechanical power Pmax [W/kg]
- Force-velocity slope S_fv [mechanical phenotype]
- Rate of force development index (RFD)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.optimize import curve_fit


def exponential_velocity(t: np.ndarray, v_max: float, tau: float) -> np.ndarray:
    """Mono-exponential sprint velocity model."""
    return v_max * (1.0 - np.exp(-np.maximum(t, 0.0) / np.maximum(tau, 0.01)))


def fit_samozino_fv_profile(
    speed_curve: np.ndarray,
    dt: float = 0.1,
    weight_lbs: Optional[float] = 220.0,
) -> Dict[str, float]:
    """Fit continuous Samozino Force-Velocity profile from sprint speed trajectory.

    Args:
        speed_curve: 1D array of speed points during 40-yard dash [yd/s]
        dt: Sampling time delta in seconds (default 0.1s)
        weight_lbs: Athlete body weight in pounds for total force/power calculation

    Returns:
        Dictionary of biomechanical profiling parameters.
    """
    speed_curve = np.asarray(speed_curve, dtype=np.float64)
    L = len(speed_curve)
    if L < 5:
        return {
            'fv_v0': 0.0, 'fv_tau': 1.0, 'fv_amax': 0.0,
            'fv_f0_rel': 0.0, 'fv_pmax_rel': 0.0, 'fv_slope': 0.0,
            'fv_rfd': 0.0, 'fv_F0_total': 0.0, 'fv_Pmax_total': 0.0,
        }

    t_curve = np.arange(L) * dt
    peak_idx = int(np.argmax(speed_curve))
    fit_end = min(L, max(peak_idx + 3, 15))
    t_fit = t_curve[:fit_end]
    s_fit = speed_curve[:fit_end]

    try:
        popt, _ = curve_fit(
            exponential_velocity, t_fit, s_fit,
            p0=[float(np.max(s_fit)), 1.0],
            bounds=([3.0, 0.1], [15.0, 4.0]),
            maxfev=600,
        )
        v_0 = float(popt[0])
        tau = float(popt[1])
    except Exception:
        v_0 = float(np.max(s_fit))
        tau = 1.0

    a_max = v_0 / tau
    f_0 = a_max
    p_max = (f_0 * v_0) / 4.0
    s_fv = - (f_0 / (v_0 + 1e-4))
    rfd_index = a_max / (tau + 1e-4)

    w = 220.0 if (weight_lbs is None or np.isnan(weight_lbs)) else float(weight_lbs)
    mass_kg = w * 0.45359237
    total_F0 = f_0 * mass_kg
    total_Pmax = p_max * mass_kg

    return {
        'fv_v0': float(v_0),
        'fv_tau': float(tau),
        'fv_amax': float(a_max),
        'fv_f0_rel': float(f_0),
        'fv_pmax_rel': float(p_max),
        'fv_slope': float(s_fv),
        'fv_rfd': float(rfd_index),
        'fv_F0_total': float(total_F0),
        'fv_Pmax_total': float(total_Pmax),
    }


def estimate_samozino_from_jump(
    vertical_inches: Optional[float] = None,
    broad_jump_inches: Optional[float] = None,
    weight_lbs: Optional[float] = 220.0,
) -> Dict[str, float]:
    """Estimate continuous Samozino F-V parameters from vertical/broad jump mechanics.

    Based on Morin & Samozino (2016) ballistic push-off impulse physics:
    h_v = v_takeoff^2 / (2 * g) -> v_takeoff = sqrt(2 * g * h_v).
    Push-off acceleration a_po = g * (h_v / h_push_off), where h_push_off ~ 0.40m.
    Horizontal force potential scales with explosive jump impulse.

    Args:
        vertical_inches: Standing vertical jump in inches
        broad_jump_inches: Standing broad jump in inches
        weight_lbs: Body weight in pounds

    Returns:
        Dictionary of biomechanical profiling parameters.
    """
    w = 220.0 if (weight_lbs is None or np.isnan(weight_lbs)) else float(weight_lbs)
    mass_kg = w * 0.45359237

    # Default to cohort average vertical (32.5 inches) if missing
    vert_in = 32.5 if (vertical_inches is None or np.isnan(vertical_inches)) else float(vertical_inches)
    vert_m = vert_in * 0.0254

    # Theoretical takeoff velocity and horizontal sprint projection
    g = 9.81
    v_to = np.sqrt(2.0 * g * max(vert_m, 0.2))  # m/s
    v_0_yds = (v_to * 1.09361) * 2.15  # Projection to maximum horizontal sprint speed (yd/s)
    v_0 = float(np.clip(v_0_yds, 8.5, 12.8))

    # Push-off acceleration and time constant tau
    h_po = 0.40  # effective push-off displacement (m)
    a_po_rel = g * (1.0 + vert_m / h_po) * 0.75  # relative horizontal conversion
    f_0 = float(np.clip(a_po_rel, 5.0, 11.5))
    tau = float(np.clip(v_0 / (f_0 + 1e-4), 0.8, 1.8))
    a_max = v_0 / tau
    p_max = (f_0 * v_0) / 4.0
    s_fv = - (f_0 / (v_0 + 1e-4))
    rfd_index = a_max / (tau + 1e-4)

    total_F0 = f_0 * mass_kg
    total_Pmax = p_max * mass_kg

    return {
        'fv_v0': float(v_0),
        'fv_tau': float(tau),
        'fv_amax': float(a_max),
        'fv_f0_rel': float(f_0),
        'fv_pmax_rel': float(p_max),
        'fv_slope': float(s_fv),
        'fv_rfd': float(rfd_index),
        'fv_F0_total': float(total_F0),
        'fv_Pmax_total': float(total_Pmax),
    }

