"""NFL2027: Neuro-Kinematic Trajectory Transformer & Differential Geometry.

Production-grade modular library for NFL Big Data Bowl 2027.
Extracts continuous Frenet-Serret kinematic invariants from wearable sensors,
accelerates computation via Triton GPU kernels, embeds multi-scale dynamics via
temporal convolutional attention, and predicts regular-season NFL game performance.
"""

__version__ = "0.1.0"
__author__ = "Ragu-123"

from nfl2027.config import Config, get_config
from nfl2027.models.encoder import TrajectoryTemporalAttentionEncoder
from nfl2027.kinematics.differential_geometry import (
    compute_differential_geometry_cpu,
    compute_differential_geometry_torch,
    extract_kinematic_invariants,
)
from nfl2027.kinematics.triton_kernels import fused_differential_geometry

__all__ = [
    "Config",
    "get_config",
    "TrajectoryTemporalAttentionEncoder",
    "compute_differential_geometry_cpu",
    "compute_differential_geometry_torch",
    "extract_kinematic_invariants",
    "fused_differential_geometry",
]
