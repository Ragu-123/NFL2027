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
from nfl2027.kinematics.samozino import (
    fit_samozino_fv_profile,
    estimate_samozino_from_jump,
    exponential_velocity,
)

__all__ = [
    "compute_differential_geometry_cpu",
    "compute_differential_geometry_torch",
    "extract_kinematic_invariants",
    "aggregate_player_features",
    "fused_differential_geometry",
    "fit_samozino_fv_profile",
    "estimate_samozino_from_jump",
    "exponential_velocity",
    "HAS_TRITON",
    "HAS_CUDA",
]
