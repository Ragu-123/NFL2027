"""Continuous Differential Geometry & Frenet-Serret Invariants for Trajectories.

Implements continuous temporal and geometric derivatives:
- Instantaneous jerk (rate of force development)
- Circular wrap-around angular velocity
- Normal / centripetal acceleration (cutting loads)
- Path curvature kappa
- Specific mechanical power and cumulative work
- Centripetal kinetic flux
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch


def compute_differential_geometry_cpu(
    speed: np.ndarray,
    accel: np.ndarray,
    direction: np.ndarray,
    dt: float = 0.1,
    eps: float = 0.01,
    seq_lens: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute continuous Frenet-Serret kinematic derivatives using vectorized NumPy.

    Supports 1D arrays (T,) or 2D batch arrays (N, T).

    Args:
        speed: Tangential speed array [yd/s]
        accel: Tangential acceleration array [yd/s^2]
        direction: Motion heading angle in degrees [0, 360)
        dt: Sampling delta time in seconds (default: 0.1s for 10 Hz)
        eps: Small epsilon regularizer preventing division by zero

    Returns:
        Tuple of (jerk, curvature, normal_accel, power, flux):
        - jerk: da / dt [yd/s^3]
        - curvature: |omega| / (speed + eps) [rad/yd]
        - normal_accel: speed * |omega| [yd/s^2]
        - power: speed * accel [W/kg normalized]
        - flux: normal_accel * speed [yd^2/s^3]
    """
    is_1d = speed.ndim == 1
    if is_1d:
        speed = speed[np.newaxis, :]
        accel = accel[np.newaxis, :]
        direction = direction[np.newaxis, :]

    # 1. Specific Mechanical Power: p = s * a
    power = speed * accel

    # 2. Instantaneous Jerk: j = da / dt
    jerk = np.zeros_like(accel)
    jerk[:, 1:] = (accel[:, 1:] - accel[:, :-1]) / dt

    # 3. Continuous Angular Velocity omega with circular wrap-around [-180, 180]
    dir_diff = np.zeros_like(direction)
    dir_diff[:, 1:] = direction[:, 1:] - direction[:, :-1]
    wrapped_diff = dir_diff - 360.0 * np.floor((dir_diff + 180.0) / 360.0)
    omega = (wrapped_diff / dt) * (np.pi / 180.0)
    abs_omega = np.abs(omega)

    # 4. Centripetal / Normal Acceleration: a_n = s * |omega|
    an = speed * abs_omega

    # 5. Differential Curvature: kappa = |omega| / (s + eps)
    curv = abs_omega / (speed + eps)

    # 6. Centripetal Kinetic Flux: Flux = a_n * s
    flux = an * speed

    # Zero out padded elements beyond seq_lens
    if seq_lens is not None:
        lens_arr = np.asarray(seq_lens)
        if lens_arr.ndim == 0:
            lens_arr = np.array([lens_arr])
        t_idx = np.arange(speed.shape[1])[np.newaxis, :]
        pad_mask = t_idx >= lens_arr[:, np.newaxis]
        jerk[pad_mask] = 0.0
        curv[pad_mask] = 0.0
        an[pad_mask] = 0.0
        power[pad_mask] = 0.0
        flux[pad_mask] = 0.0

    if is_1d:
        return jerk[0], curv[0], an[0], power[0], flux[0]
    return jerk, curv, an, power, flux


def compute_differential_geometry_torch(
    speed: torch.Tensor,
    accel: torch.Tensor,
    direction: torch.Tensor,
    dt: float = 0.1,
    eps: float = 0.01,
    seq_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute continuous Frenet-Serret derivatives using PyTorch tensors.

    Supports arbitrary device (CPU / CUDA) and batch dimensions (..., T).

    Args:
        speed: Tangential speed tensor
        accel: Tangential acceleration tensor
        direction: Motion heading tensor in degrees
        dt: Sampling delta time in seconds
        eps: Epsilon regularizer for curvature
        seq_lens: Optional sequence lengths tensor to mask padded elements to 0.0

    Returns:
        Tuple of (jerk, curvature, normal_accel, power, flux) as torch.Tensors.
    """
    # 1. Specific Mechanical Power
    power = speed * accel

    # 2. Instantaneous Jerk
    jerk = torch.zeros_like(accel)
    jerk[..., 1:] = (accel[..., 1:] - accel[..., :-1]) / dt

    # 3. Circular wrap-around angular velocity
    dir_diff = torch.zeros_like(direction)
    dir_diff[..., 1:] = direction[..., 1:] - direction[..., :-1]
    wrapped_diff = dir_diff - 360.0 * torch.floor((dir_diff + 180.0) / 360.0)
    omega = (wrapped_diff / dt) * (np.pi / 180.0)
    abs_omega = torch.abs(omega)

    # 4. Centripetal / Normal Acceleration
    an = speed * abs_omega

    # 5. Differential Curvature
    curv = abs_omega / (speed + eps)

    # 6. Centripetal Kinetic Flux
    flux = an * speed

    # Zero out padded elements beyond seq_lens
    if seq_lens is not None:
        T = speed.shape[-1]
        t_idx = torch.arange(T, device=speed.device)
        if seq_lens.ndim == 1 and speed.ndim >= 2:
            pad_mask = t_idx.unsqueeze(0) >= seq_lens.unsqueeze(1)
            jerk = jerk.masked_fill(pad_mask, 0.0)
            curv = curv.masked_fill(pad_mask, 0.0)
            an = an.masked_fill(pad_mask, 0.0)
            power = power.masked_fill(pad_mask, 0.0)
            flux = flux.masked_fill(pad_mask, 0.0)

    return jerk, curv, an, power, flux


def extract_kinematic_invariants(
    speed_arr: np.ndarray,
    accel_arr: np.ndarray,
    jerk_arr: np.ndarray,
    curv_arr: np.ndarray,
    an_arr: np.ndarray,
    power_arr: np.ndarray,
    flux_arr: np.ndarray,
    seq_lens: np.ndarray,
    meta_keys: List[Tuple],
    dt: float = 0.1,
) -> pd.DataFrame:
    """Extract scalar physical invariants and phase markers per drill run.

    Args:
        speed_arr: Speed matrix (N, T)
        accel_arr: Accel matrix (N, T)
        jerk_arr: Jerk matrix (N, T)
        curv_arr: Curvature matrix (N, T)
        an_arr: Normal accel matrix (N, T)
        power_arr: Power matrix (N, T)
        flux_arr: Kinetic flux matrix (N, T)
        seq_lens: Valid sequence lengths (N,)
        meta_keys: List of (nfl_id, drill_type, drill_name, attempt)
        dt: Sampling delta time

    Returns:
        DataFrame of scalar kinematic features for each drill attempt.
    """
    feature_records = []

    for i, (nfl_id, drill_type, drill_name, attempt) in enumerate(meta_keys):
        L = int(seq_lens[i])
        if L < 5:
            continue

        s_seq = speed_arr[i, :L]
        a_seq = accel_arr[i, :L]
        j_seq = jerk_arr[i, :L]
        c_seq = curv_arr[i, :L]
        an_seq = an_arr[i, :L]
        p_seq = power_arr[i, :L]
        fl_seq = flux_arr[i, :L]

        L_burst_05 = min(5, L)
        L_burst_10 = min(10, L)

        feature_records.append(
            {
                "nfl_id": nfl_id,
                "drill_type": drill_type,
                "drill_name": drill_name,
                "attempt": attempt,
                "peak_speed": float(np.max(s_seq)),
                "peak_accel": float(np.max(a_seq)),
                "accel_burst_05": float(np.max(a_seq[:L_burst_05])),
                "jerk_burst_05": float(np.max(j_seq[:L_burst_05])),
                "power_burst_05": float(np.max(p_seq[:L_burst_05])),
                "accel_burst_10": float(np.max(a_seq[:L_burst_10])),
                "jerk_burst_10": float(np.max(j_seq[:L_burst_10])),
                "power_burst_10": float(np.max(p_seq[:L_burst_10])),
                "brake_accel_min": float(np.min(a_seq)),
                "curv_p95": float(np.percentile(c_seq, 95)),
                "an_max": float(np.max(an_seq)),
                "flux_p90": float(np.percentile(fl_seq, 90)),
                "cum_mech_work": float(np.sum(p_seq) * dt),
            }
        )

    return pd.DataFrame(feature_records)


def aggregate_player_features(
    df_feat: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_latent_emb: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate per-drill features into athlete-level master matrix with physical interaction terms.

    Args:
        df_feat: Per-drill feature DataFrame from extract_kinematic_invariants
        df_meta: Player demographic & combine metadata from merge_metadata
        df_latent_emb: Optional DataFrame with PyTorch latent embeddings per nfl_id

    Returns:
        Athlete-level master DataFrame ready for multi-task modeling.
    """
    # 1. Forty Yard Dash Kinematics
    forty_df = df_feat[df_feat["drill_type"] == "FORTY_YARD_DASH"]
    if len(forty_df) > 0:
        forty_feat = forty_df.groupby("nfl_id").agg(
            forty_peak_speed=("peak_speed", "max"),
            forty_peak_accel=("peak_accel", "max"),
            forty_accel_05=("accel_burst_05", "max"),
            forty_jerk_05=("jerk_burst_05", "max"),
            forty_power_05=("power_burst_05", "max"),
            forty_accel_10=("accel_burst_10", "max"),
            forty_jerk_10=("jerk_burst_10", "max"),
            forty_power_10=("power_burst_10", "max"),
            forty_work=("cum_mech_work", "max"),
        ).reset_index()
    else:
        forty_feat = pd.DataFrame({"nfl_id": df_meta["nfl_id"].unique()})

    # 2. Overall Athlete Differential Geometry across all drills
    overall_kin = df_feat.groupby("nfl_id").agg(
        all_peak_speed=("peak_speed", "max"),
        all_peak_accel=("peak_accel", "max"),
        all_peak_jerk=("jerk_burst_10", "max"),
        all_peak_power=("power_burst_10", "max"),
        all_peak_an=("an_max", "max"),
        all_peak_curv=("curv_p95", "max"),
        all_max_brake=("brake_accel_min", "min"),
    ).reset_index()

    # 3. Position-Specific Skill Drills
    dl_df = df_feat[df_feat["drill_type"] == "SKILL_DRILLS_DL"]
    if len(dl_df) > 0:
        dl_drill_feat = dl_df.groupby("nfl_id").agg(
            dl_peak_an=("an_max", "max"),
            dl_peak_curv=("curv_p95", "max"),
            dl_burst_jerk=("jerk_burst_05", "max"),
            dl_burst_power=("power_burst_05", "max"),
        ).reset_index()
    else:
        dl_drill_feat = pd.DataFrame(columns=["nfl_id", "dl_peak_an", "dl_peak_curv", "dl_burst_jerk", "dl_burst_power"])

    wr_df = df_feat[df_feat["drill_type"] == "SKILL_DRILLS_WR"]
    if len(wr_df) > 0:
        wr_drill_feat = wr_df.groupby("nfl_id").agg(
            wr_brake=("brake_accel_min", "min"),
            wr_peak_an=("an_max", "max"),
            wr_peak_curv=("curv_p95", "max"),
            wr_burst_jerk=("jerk_burst_05", "max"),
        ).reset_index()
    else:
        wr_drill_feat = pd.DataFrame(columns=["nfl_id", "wr_brake", "wr_peak_an", "wr_peak_curv", "wr_burst_jerk"])

    master_df = df_meta.merge(forty_feat, on="nfl_id", how="left")
    master_df = master_df.merge(overall_kin, on="nfl_id", how="left")
    master_df = master_df.merge(dl_drill_feat, on="nfl_id", how="left")
    master_df = master_df.merge(wr_drill_feat, on="nfl_id", how="left")

    if df_latent_emb is not None:
        master_df = master_df.merge(df_latent_emb, on="nfl_id", how="left")

    # Anthropometric mechanical interaction terms
    weight = master_df.get("combine_weight", 200.0).fillna(200.0)
    jerk_05 = master_df.get("forty_jerk_05", 0.0).fillna(0.0)
    power_05 = master_df.get("forty_power_05", 0.0).fillna(0.0)
    power_10 = master_df.get("forty_power_10", 0.0).fillna(0.0)
    speed_peak = master_df.get("forty_peak_speed", 0.0).fillna(0.0)

    master_df["mass_scaled_jerk"] = jerk_05 / (weight + 1e-4)
    master_df["mass_scaled_power"] = power_05 / (weight + 1e-4)
    master_df["momentum_peak"] = weight * speed_peak
    master_df["power_to_weight"] = power_10 / (weight + 1e-4)

    return master_df
