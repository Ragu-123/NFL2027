"""Unit tests for Frenet-Serret differential geometry and kinematic invariants."""

import numpy as np
import pytest
import torch

from nfl2027.kinematics.differential_geometry import (
    compute_differential_geometry_cpu,
    compute_differential_geometry_torch,
    extract_kinematic_invariants,
)
from nfl2027.kinematics.triton_kernels import fused_differential_geometry


def test_constant_acceleration_zero_jerk():
    """Verify that linear acceleration produces zero jerk."""
    dt = 0.1
    t = np.arange(50) * dt
    # a(t) = 2.0 (constant) -> s(t) = 2.0 * t
    a = np.full(50, 2.0, dtype=np.float32)
    s = (2.0 * t).astype(np.float32)
    d = np.zeros(50, dtype=np.float32)

    jerk, curv, an, power, flux = compute_differential_geometry_cpu(s, a, d, dt=dt)

    # For t > 0, da/dt should be zero
    np.testing.assert_allclose(jerk[1:], 0.0, atol=1e-5)
    # Power p(t) = s(t) * a(t)
    np.testing.assert_allclose(power, s * a, atol=1e-5)
    # No directional change -> an = 0, curv = 0
    np.testing.assert_allclose(an, 0.0, atol=1e-5)
    np.testing.assert_allclose(curv, 0.0, atol=1e-5)


def test_circular_motion_frenet_serret():
    """Verify circular motion centripetal acceleration and curvature."""
    dt = 0.1
    num_frames = 100
    t = np.arange(num_frames) * dt
    R = 5.0  # radius: 5 yards
    s_val = 6.0  # speed: 6 yd/s
    omega_expected = s_val / R  # 1.2 rad/s
    an_expected = s_val * omega_expected  # 7.2 yd/s^2

    s = np.full(num_frames, s_val, dtype=np.float32)
    a = np.zeros(num_frames, dtype=np.float32)
    # Direction changes at omega_expected rad/s -> convert to degrees/s
    deg_rate = omega_expected * (180.0 / np.pi)
    d = ((deg_rate * t) % 360.0).astype(np.float32)

    jerk, curv, an, power, flux = compute_differential_geometry_cpu(s, a, d, dt=dt, eps=0.0)

    # From frame 1 onwards, centripetal acceleration should match s * omega
    np.testing.assert_allclose(an[1:], an_expected, rtol=1e-3)
    # Curvature should match 1/R = 0.2 rad/yd
    np.testing.assert_allclose(curv[1:], 1.0 / R, rtol=1e-3)
    # Jerk and power should be 0 since linear acceleration is 0
    np.testing.assert_allclose(jerk, 0.0, atol=1e-5)
    np.testing.assert_allclose(power, 0.0, atol=1e-5)


def test_direction_circular_wrap_around():
    """Verify that angular difference properly wraps around 0/360 degrees."""
    dt = 0.1
    # Turning right across north: 358 deg -> 2 deg (+4 deg step)
    d = np.array([358.0, 2.0, 6.0], dtype=np.float32)
    s = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    a = np.zeros(3, dtype=np.float32)

    jerk, curv, an, power, flux = compute_differential_geometry_cpu(s, a, d, dt=dt)

    # 4 degrees in 0.1s = 40 deg/s = 40 * pi / 180 = 0.69813 rad/s
    expected_omega = 40.0 * (np.pi / 180.0)
    expected_an = 5.0 * expected_omega

    np.testing.assert_allclose(an[1], expected_an, rtol=1e-3)
    np.testing.assert_allclose(an[2], expected_an, rtol=1e-3)


def test_numpy_vs_torch_differential_geometry_equivalence():
    """Verify complete numerical equivalence between CPU NumPy and PyTorch implementations."""
    rng = np.random.RandomState(42)
    N, T = 10, 80
    dt = 0.1
    s_np = rng.uniform(0.0, 10.0, size=(N, T)).astype(np.float32)
    a_np = rng.uniform(-3.0, 4.0, size=(N, T)).astype(np.float32)
    d_np = rng.uniform(0.0, 360.0, size=(N, T)).astype(np.float32)

    c_jerk, c_curv, c_an, c_power, c_flux = compute_differential_geometry_cpu(s_np, a_np, d_np, dt=dt)

    s_t = torch.tensor(s_np)
    a_t = torch.tensor(a_np)
    d_t = torch.tensor(d_np)
    t_jerk, t_curv, t_an, t_power, t_flux = compute_differential_geometry_torch(s_t, a_t, d_t, dt=dt)

    np.testing.assert_allclose(c_jerk, t_jerk.numpy(), atol=1e-5)
    np.testing.assert_allclose(c_curv, t_curv.numpy(), atol=1e-5)
    np.testing.assert_allclose(c_an, t_an.numpy(), atol=1e-5)
    np.testing.assert_allclose(c_power, t_power.numpy(), atol=1e-5)
    np.testing.assert_allclose(c_flux, t_flux.numpy(), atol=1e-5)


def test_fused_differential_geometry_fallback():
    """Verify that fused_differential_geometry produces valid tensors on any device."""
    N, T = 5, 60
    s = np.ones((N, T), dtype=np.float32) * 5.0
    a = np.ones((N, T), dtype=np.float32) * 2.0
    d = np.ones((N, T), dtype=np.float32) * 90.0
    lens = np.full(N, T, dtype=np.int32)

    jerk, curv, an, power, flux = fused_differential_geometry(s, a, d, lens, max_len=T, dt=0.1)

    assert isinstance(jerk, torch.Tensor)
    assert jerk.shape == (N, T)
    assert power.shape == (N, T)
    # Specific mechanical power = 5.0 * 2.0 = 10.0
    np.testing.assert_allclose(power.numpy(), 10.0, atol=1e-5)
