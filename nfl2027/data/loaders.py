"""Robust data loading, relational merging, and trajectory sequence extraction.

Handles schema normalization, 4-tuple sequence grouping, and synthetic data generation.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from nfl2027.config import (
    TRADITIONAL_COMBINE_COLS,
    SEQUENCE_GROUPING_KEYS,
    Config,
    detect_data_dir,
)


def load_raw_data(data_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Load competition CSV tables with proper numeric type coercion.

    Args:
        data_dir: Path to directory containing CSV files. If None, auto-detected.

    Returns:
        Dictionary containing DataFrames for:
        'players', 'combine_results', 'player_career_successes',
        'combine_tracking', and 'player_play'.
    """
    if data_dir is None:
        data_dir = detect_data_dir()
    if data_dir is None or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Please specify data_dir or set NFL2027_DATA_DIR environment variable."
        )

    filenames = {
        "players": "players.csv",
        "combine_results": "combine_results.csv",
        "player_career_successes": "player_career_successes.csv",
        "combine_tracking": "combine_tracking.csv",
        "player_play": "player_play.csv",
    }

    dfs = {}
    for key, fname in filenames.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required table {fname} not found in {data_dir}")
        dfs[key] = pd.read_csv(fpath)

    # Ensure numeric types on combine metrics
    for col in TRADITIONAL_COMBINE_COLS:
        if col in dfs["combine_results"].columns:
            dfs["combine_results"][col] = pd.to_numeric(
                dfs["combine_results"][col], errors="coerce"
            )

    return dfs


def merge_metadata(
    df_players: pd.DataFrame,
    df_combine_res: pd.DataFrame,
    df_success: pd.DataFrame,
) -> pd.DataFrame:
    """Merge player demographics, combine measurements, and career success metrics.

    Uses the composite shared key ['nfl_id', 'draft_year'] to avoid column collisions.

    Args:
        df_players: DataFrame from players.csv
        df_combine_res: DataFrame from combine_results.csv
        df_success: DataFrame from player_career_successes.csv

    Returns:
        Merged master metadata DataFrame with anthropometric ratios and snap totals.
    """
    # Merge using composite key on draft_year to avoid column name collisions (_x, _y)
    df_meta = df_players.merge(
        df_combine_res, on=["nfl_id", "draft_year"], how="inner"
    ).merge(df_success, on="nfl_id", how="inner")

    # Total career snaps
    off_snaps = df_meta.get("career_offensive_snaps", 0)
    def_snaps = df_meta.get("career_defensive_snaps", 0)
    df_meta["total_career_snaps"] = off_snaps + def_snaps

    # Anthropometric indices
    if "combine_weight" in df_meta.columns and "combine_height" in df_meta.columns:
        df_meta["bmi"] = (
            df_meta["combine_weight"] / (df_meta["combine_height"] ** 2)
        ) * 703.0
    if "arm_length" in df_meta.columns and "combine_height" in df_meta.columns:
        df_meta["arm_to_height_ratio"] = (
            df_meta["arm_length"] / df_meta["combine_height"]
        )
    if "wing_span" in df_meta.columns and "combine_height" in df_meta.columns:
        df_meta["wingspan_to_height_ratio"] = (
            df_meta["wing_span"] / df_meta["combine_height"]
        )

    return df_meta


def prepare_tracking_sequences(
    df_comb_trk: pd.DataFrame,
    max_len: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple], List[pd.DataFrame]]:
    """Assemble fixed-duration padded tracking sequence matrices.

    Groups strictly by 4-tuple key: ['nfl_id', 'drill_type', 'drill_name', 'attempt'].

    Args:
        df_comb_trk: Raw combine tracking DataFrame
        max_len: Fixed sequence length for tensor batching

    Returns:
        Tuple containing:
        - speed_arr: float32 array of shape (N, max_len)
        - accel_arr: float32 array of shape (N, max_len)
        - dir_arr: float32 array of shape (N, max_len)
        - seq_lens: int32 array of shape (N,)
        - meta_keys: List of (nfl_id, drill_type, drill_name, attempt)
        - seq_groups: List of original DataFrames per sequence
    """
    # Filter to player entities only
    if "entity_type" in df_comb_trk.columns:
        player_trk = df_comb_trk[df_comb_trk["entity_type"] == "PLAYER"].copy()
    else:
        player_trk = df_comb_trk.copy()

    # Time sorting
    if "time" in player_trk.columns:
        player_trk["time_dt"] = pd.to_datetime(player_trk["time"], format="mixed", errors="coerce")
        sort_cols = [c for c in SEQUENCE_GROUPING_KEYS + ["time_dt"] if c in player_trk.columns]
        player_trk = player_trk.sort_values(by=sort_cols).reset_index(drop=True)

    # Group by 4-tuple key
    valid_group_keys = [k for k in SEQUENCE_GROUPING_KEYS if k in player_trk.columns]
    seq_groups = [group for _, group in player_trk.groupby(valid_group_keys)]
    num_seqs = len(seq_groups)

    speed_arr = np.zeros((num_seqs, max_len), dtype=np.float32)
    accel_arr = np.zeros((num_seqs, max_len), dtype=np.float32)
    dir_arr = np.zeros((num_seqs, max_len), dtype=np.float32)
    seq_lens = np.zeros(num_seqs, dtype=np.int32)
    meta_keys = []

    for i, grp in enumerate(seq_groups):
        L = min(len(grp), max_len)
        speed_arr[i, :L] = grp["s"].values[:L]
        accel_arr[i, :L] = grp["a"].values[:L]
        dir_arr[i, :L] = grp["dir"].values[:L]
        seq_lens[i] = L
        meta_keys.append(
            (
                grp["nfl_id"].iloc[0],
                grp["drill_type"].iloc[0] if "drill_type" in grp.columns else "DRILL",
                grp["drill_name"].iloc[0] if "drill_name" in grp.columns else "NAME",
                grp["attempt"].iloc[0] if "attempt" in grp.columns else 1,
            )
        )

    return speed_arr, accel_arr, dir_arr, seq_lens, meta_keys, seq_groups


def prepare_play_targets(
    df_pp: pd.DataFrame,
    master_df: pd.DataFrame,
    min_dl_get_off_snaps: int = 15,
    min_dl_pressure_snaps: int = 25,
    min_wr_routes: int = 25,
) -> Dict[str, pd.DataFrame]:
    """Aggregate in-game performance target labels for cross-validation tasks.

    Args:
        df_pp: player_play.csv DataFrame
        master_df: Combined metadata and kinematic features DataFrame
        min_dl_get_off_snaps: Minimum snap threshold for DL get-off aggregation
        min_dl_pressure_snaps: Minimum snap threshold for DL pressure rate
        min_wr_routes: Minimum route threshold for WR cushion respect

    Returns:
        Dict mapping task names ('dl_get_off', 'dl_pressure_rate', 'wr_cushion')
        to merged DataFrames containing features and target labels.
    """
    targets = {}

    # Task 1: DL Get-Off
    if "player_get_off" in df_pp.columns:
        dl_target = (
            df_pp.dropna(subset=["player_get_off"])
            .groupby("nfl_id")
            .agg(target_get_off=("player_get_off", "mean"), snaps=("player_get_off", "count"))
            .reset_index()
        )
        dl_target = dl_target[dl_target["snaps"] >= min_dl_get_off_snaps]
        df_dl = master_df.merge(dl_target, on="nfl_id", how="inner")
        targets["dl_get_off"] = df_dl

    # Task 2: DL Pressure Rate
    req_cols = {"player_get_off", "sack", "time_to_pressure"}
    if req_cols.issubset(df_pp.columns):
        dl_pp_all = df_pp[
            df_pp["player_get_off"].notna()
            | (df_pp["sack"] > 0)
            | df_pp["time_to_pressure"].notna()
        ].copy()
        dl_impact = (
            dl_pp_all.groupby("nfl_id")
            .agg(
                snaps=("play_id", "count") if "play_id" in dl_pp_all.columns else ("sack", "count"),
                sacks=("sack", "sum"),
                pressures=("time_to_pressure", "count"),
            )
            .reset_index()
        )
        dl_impact["pressure_rate"] = dl_impact["pressures"] / dl_impact["snaps"].clip(lower=1)
        dl_impact = dl_impact[dl_impact["snaps"] >= min_dl_pressure_snaps]
        df_pr = master_df.merge(dl_impact, on="nfl_id", how="inner")
        targets["dl_pressure_rate"] = df_pr

    # Task 3: WR Cushion Respect
    if "cushion" in df_pp.columns and "combine_position" in master_df.columns:
        wr_ids = master_df[master_df["combine_position"] == "WR"]["nfl_id"]
        wr_pp_all = df_pp[df_pp["nfl_id"].isin(wr_ids)].copy()
        wr_cush = (
            wr_pp_all.groupby("nfl_id")
            .agg(
                routes=("play_id", "count") if "play_id" in wr_pp_all.columns else ("cushion", "count"),
                mean_cushion=("cushion", "mean"),
            )
            .reset_index()
        )
        wr_cush = wr_cush[wr_cush["routes"] >= min_wr_routes]
        df_wr = master_df.merge(wr_cush, on="nfl_id", how="inner")
        targets["wr_cushion"] = df_wr

    return targets


def generate_synthetic_dataset(
    num_players: int = 60,
    drills_per_player: int = 4,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generate realistic synthetic DataFrames matching NFL Big Data Bowl 2027 schema.

    Enables complete unit testing and local development without access to raw data.

    Args:
        num_players: Number of synthetic prospects
        drills_per_player: Number of drill attempts per player
        random_seed: Seed for reproducibility

    Returns:
        Dictionary of DataFrames: 'players', 'combine_results',
        'player_career_successes', 'combine_tracking', 'player_play'.
    """
    rng = np.random.RandomState(random_seed)

    player_ids = np.arange(50000, 50000 + num_players)
    positions = ["WR", "CB", "DE", "DT", "T", "G", "TE", "OLB", "SS"]
    assigned_pos = rng.choice(positions, size=num_players)
    draft_years = rng.choice([2023, 2024, 2025], size=num_players)
    draft_picks = rng.randint(1, 256, size=num_players)

    # 1. Players table
    df_players = pd.DataFrame(
        {
            "nfl_id": player_ids,
            "display_name": [f"Athlete_{pid}" for pid in player_ids],
            "draft_year": draft_years,
            "position": assigned_pos,
        }
    )

    # 2. Combine results table
    heights = np.where(np.isin(assigned_pos, ["T", "G", "DE", "DT"]), rng.normal(76, 1.5, num_players), rng.normal(72, 1.8, num_players))
    weights = np.where(np.isin(assigned_pos, ["T", "G", "DE", "DT"]), rng.normal(285, 20, num_players), rng.normal(205, 15, num_players))
    forties = np.where(np.isin(assigned_pos, ["WR", "CB"]), rng.normal(4.45, 0.08, num_players), rng.normal(4.90, 0.20, num_players))
    ten_splits = forties * 0.35 + rng.normal(0, 0.02, num_players)

    df_combine_res = pd.DataFrame(
        {
            "nfl_id": player_ids,
            "draft_year": draft_years,
            "combine_position": assigned_pos,
            "combine_height": np.round(heights, 1),
            "combine_weight": np.round(weights, 1),
            "hand_size": np.round(rng.normal(9.5, 0.5, num_players), 2),
            "arm_length": np.round(rng.normal(32.5, 1.2, num_players), 2),
            "wing_span": np.round(rng.normal(78.0, 3.0, num_players), 2),
            "ten_yd_split": np.round(ten_splits, 2),
            "forty": np.round(forties, 2),
            "vertical": np.round(rng.normal(34.0, 3.5, num_players), 1),
            "broad_jump": np.round(rng.normal(120.0, 8.0, num_players), 1),
            "three_cone": np.round(rng.normal(7.10, 0.30, num_players), 2),
            "short_shuttle": np.round(rng.normal(4.30, 0.20, num_players), 2),
            "bench_reps": rng.randint(12, 35, num_players),
        }
    )

    # 3. Player career successes table
    expected_snaps = 1853.6 * np.exp(-0.0094 * draft_picks) + 17.7
    career_snaps = np.maximum(0, rng.normal(expected_snaps, 250)).astype(int)
    df_success = pd.DataFrame(
        {
            "nfl_id": player_ids,
            "draft_overall_pick": draft_picks,
            "career_offensive_snaps": np.where(np.isin(assigned_pos, ["WR", "TE", "T", "G"]), career_snaps, 0),
            "career_defensive_snaps": np.where(np.isin(assigned_pos, ["DE", "DT", "CB", "OLB", "SS"]), career_snaps, 0),
            "games_started": np.clip(career_snaps // 60, 0, 50),
        }
    )

    # 4. Combine tracking table
    tracking_rows = []
    drill_choices = [
        ("FORTY_YARD_DASH", "FORTY_YARD_DASH"),
        ("SKILL_DRILLS_DL", "RUN_THE_HOOP_DRILL"),
        ("SKILL_DRILLS_WR", "GAUNTLET_DRILL"),
        ("SHORT_SHUTTLE", "SHORT_SHUTTLE"),
    ]

    base_time = pd.Timestamp("2026-03-01 10:00:00")
    for pid, pos in zip(player_ids, assigned_pos):
        for att in range(1, drills_per_player + 1):
            drill_type, drill_name = drill_choices[(att - 1) % len(drill_choices)]
            seq_len = rng.randint(40, 110)
            
            # Physics-like trajectory: burst acceleration then cruising
            t = np.arange(seq_len) * 0.1
            v_max = rng.uniform(7.0, 10.5)
            tau = rng.uniform(1.0, 2.0)
            speed = v_max * (1.0 - np.exp(-t / tau)) + rng.normal(0, 0.05, seq_len)
            speed = np.maximum(speed, 0.0)
            accel = np.gradient(speed, 0.1)
            # Simulated curved route or straight dash
            if "HOOP" in drill_name or "SHUTTLE" in drill_name:
                direction = (np.sin(t * 1.5) * 60.0 + 90.0) % 360.0
            else:
                direction = rng.normal(90.0, 2.0, seq_len) % 360.0

            x = np.cumsum(speed * np.cos(np.radians(direction))) * 0.1
            y = np.cumsum(speed * np.sin(np.radians(direction))) * 0.1

            for step in range(seq_len):
                tracking_rows.append(
                    {
                        "nfl_id": pid,
                        "entity_type": "PLAYER",
                        "drill_type": drill_type,
                        "drill_name": drill_name,
                        "attempt": att,
                        "time": str(base_time + pd.Timedelta(seconds=t[step])),
                        "x": float(x[step]),
                        "y": float(y[step]),
                        "s": float(speed[step]),
                        "a": float(accel[step]),
                        "dir": float(direction[step]),
                    }
                )

    df_comb_trk = pd.DataFrame(tracking_rows)

    # 5. Player play table (game plays)
    play_rows = []
    for pid, pos in zip(player_ids, assigned_pos):
        num_plays = rng.randint(20, 60)
        for p_idx in range(num_plays):
            get_off = float(rng.normal(0.72, 0.10)) if pos in ["DE", "DT", "OLB"] else np.nan
            cushion = float(rng.normal(6.5, 1.2)) if pos in ["WR", "CB"] else np.nan
            sack = 1 if (pos in ["DE", "DT"] and rng.rand() < 0.05) else 0
            ttp = float(rng.normal(2.4, 0.4)) if (pos in ["DE", "DT"] and rng.rand() < 0.15) else np.nan

            play_rows.append(
                {
                    "nfl_id": pid,
                    "play_id": 1000 + p_idx,
                    "game_id": 2026010100 + (p_idx // 10),
                    "player_get_off": get_off,
                    "cushion": cushion,
                    "sack": sack,
                    "time_to_pressure": ttp,
                }
            )

    df_pp = pd.DataFrame(play_rows)

    return {
        "players": df_players,
        "combine_results": df_combine_res,
        "player_career_successes": df_success,
        "combine_tracking": df_comb_trk,
        "player_play": df_pp,
    }
