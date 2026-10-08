"""Kinematics and differential geometry computations for athlete trajectories."""

from nfl2027.kinematics.differential_geometry import (
    compute_differential_geometry_cpu,
    compute_differential_geometry_torch,
    extract_kinematic_invariants,
    aggregate_player_features,
)
from nfl2027.kinematics.triton_kernels import (
    fused_differential_geometry,
    HAS_TRITON,
    HAS_CUDA,
)

__all__ = [
    "compute_differential_geometry_cpu",
    "compute_differential_geometry_torch",
    "extract_kinematic_invariants",
    "aggregate_player_features",
    "fused_differential_geometry",
    "HAS_TRITON",
    "HAS_CUDA",
]
