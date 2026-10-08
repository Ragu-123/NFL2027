"""Global Configuration & Constants for NFL2027.

Handles environment detection (Kaggle cloud vs local workstation),
drill definitions, physical sampling constants, and model hyperparameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Candidate dataset directories checked in order
CANDIDATE_DATA_DIRS = [
    os.environ.get("NFL2027_DATA_DIR", ""),
    "/kaggle/input/competitions/nfl-big-data-bowl-2027/nfl-big-data-bowl-2027",
    "/kaggle/input/nfl-big-data-bowl-2027",
    str(Path.cwd() / "data"),
    str(Path.cwd().parent / "data"),
    str(Path(__file__).resolve().parent.parent / "data"),
]

# Candidate output directories for saving figures and artifacts
CANDIDATE_OUTPUT_DIRS = [
    os.environ.get("NFL2027_OUTPUT_DIR", ""),
    "/kaggle/working",
    str(Path.cwd() / "figures"),
    str(Path.cwd().parent / "figures"),
    str(Path(__file__).resolve().parent.parent / "knowledge" / "figures"),
]


def detect_data_dir() -> Optional[str]:
    """Auto-detect the path containing the NFL Big Data Bowl 2027 dataset."""
    for path in CANDIDATE_DATA_DIRS:
        if path and os.path.isdir(path):
            # Check for core file presence
            if os.path.exists(os.path.join(path, "players.csv")):
                return path
    return None


def detect_output_dir() -> str:
    """Auto-detect or create a suitable directory for saving outputs."""
    for path in CANDIDATE_OUTPUT_DIRS:
        if path and os.path.isdir(path):
            return path
    # Fallback to local 'figures' directory
    fallback = str(Path.cwd() / "figures")
    os.makedirs(fallback, exist_ok=True)
    return fallback


# Traditional combine physical metrics reported by scouts
TRADITIONAL_COMBINE_COLS = [
    "combine_height",
    "combine_weight",
    "hand_size",
    "arm_length",
    "wing_span",
    "ten_yd_split",
    "forty",
    "vertical",
    "broad_jump",
    "three_cone",
    "short_shuttle",
    "bench_reps",
]

# Traditional subset for baseline regression models (M0)
TRADITIONAL_BASELINE_COLS = [
    "combine_weight",
    "combine_height",
    "ten_yd_split",
    "forty",
    "vertical",
    "broad_jump",
]

# Standard grouping key for combine tracking sequences (4-tuple)
SEQUENCE_GROUPING_KEYS = ["nfl_id", "drill_type", "drill_name", "attempt"]

# Standard drill categories tracked by wearable sensors
DRILL_TYPES = [
    "FORTY_YARD_DASH",
    "SHORT_SHUTTLE",
    "THREE_CONE_DRILL",
    "SKILL_DRILLS_WR",
    "SKILL_DRILLS_DB",
    "SKILL_DRILLS_OL",
    "SKILL_DRILLS_DL",
    "SKILL_DRILLS_TE",
]

# 7-channel trajectory tensor channels
TRAJECTORY_CHANNELS = [
    "speed",
    "accel",
    "jerk",
    "dir",
    "a_n",
    "curvature",
    "power",
]


@dataclass
class Config:
    """Configuration container for NFL2027 pipelines."""

    # Paths
    data_dir: Optional[str] = field(default_factory=detect_data_dir)
    output_dir: str = field(default_factory=detect_output_dir)

    # Physical sampling constants
    dt: float = 0.1  # 10 Hz sampling rate (0.1s per frame)
    max_seq_len: int = 256  # Max sequence frames for padded tensor batches
    burst_05_frames: int = 5  # Frames corresponding to 0.5 seconds
    burst_10_frames: int = 10  # Frames corresponding to 1.0 seconds
    eps_curvature: float = 0.01  # Epsilon regularizer for curvature division

    # Neural encoder architecture
    in_channels: int = 7  # 7 kinematic channels
    hidden_dim: int = 32  # Latent embedding dimension
    num_heads: int = 4  # Multi-head attention heads
    dropout: float = 0.1  # Attention / conv dropout

    # Validation & Training
    random_seed: int = 42
    n_splits: int = 5  # 5-fold cross-validation
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 20

    # Draft Surplus Valuation constants: E[Snaps | Pick] = A * exp(-lambda * Pick) + C
    valuation_exp_a: float = 1853.6
    valuation_exp_lambda: float = 0.0094
    valuation_exp_c: float = 17.7

    # Device
    device: str = "cpu"

    def __post_init__(self):
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        elif self.device == "cpu" and torch.cuda.is_available():
            self.device = "cuda:0"


def get_config(**kwargs) -> Config:
    """Instantiate a Config with optional overrides."""
    return Config(**kwargs)
