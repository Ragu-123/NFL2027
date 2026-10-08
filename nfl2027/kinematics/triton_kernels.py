"""Fused Triton GPU differential geometry kernel with seamless CPU/CUDA fallback.

Implements GPU-accelerated continuous trajectory invariants:
- Rate of force development (jerk)
- Centripetal cutting acceleration (a_n)
- Differential curvature (kappa)
- Specific mechanical power (p)
- Centripetal kinetic flux (Phi_n)
"""

import time
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch

from nfl2027.kinematics.differential_geometry import (
    compute_differential_geometry_torch,
    compute_differential_geometry_cpu,
)

# Detection of Triton and CUDA availability
HAS_CUDA = torch.cuda.is_available()

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False
    triton = None
    tl = None

# Define Triton JIT kernel if triton is present
if HAS_TRITON:
    @triton.jit
    def fused_trajectory_differential_geometry_kernel(
        speed_ptr,
        accel_ptr,
        dir_ptr,
        out_jerk_ptr,
        out_curv_ptr,
        out_an_ptr,
        out_power_ptr,
        out_flux_ptr,
        seq_len_ptr,
        max_len,
        dt,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Massively parallel fused GPU trajectory differential geometry kernel."""
        pid = tl.program_id(axis=0)
        seq_len = tl.load(seq_len_ptr + pid)
        offset = pid * max_len

        t_idx = tl.arange(0, BLOCK_SIZE)
        mask = t_idx < seq_len

        # Load raw trajectory channels
        s = tl.load(speed_ptr + offset + t_idx, mask=mask, other=0.0)
        a = tl.load(accel_ptr + offset + t_idx, mask=mask, other=0.0)
        d = tl.load(dir_ptr + offset + t_idx, mask=mask, other=0.0)

        # 1. Specific Mechanical Power: p = s * a
        power = s * a
        tl.store(out_power_ptr + offset + t_idx, power, mask=mask)

        # 2. Instantaneous Jerk: j = da / dt
        t_prev = tl.maximum(t_idx - 1, 0)
        a_prev = tl.load(accel_ptr + offset + t_prev, mask=mask, other=0.0)
        jerk = tl.where(t_idx > 0, (a - a_prev) / dt, 0.0)
        tl.store(out_jerk_ptr + offset + t_idx, jerk, mask=mask)

        # 3. Continuous Angular Velocity omega with circular wrap-around [-180, 180]
        d_prev = tl.load(dir_ptr + offset + t_prev, mask=mask, other=0.0)
        d_diff = d - d_prev
        wrapped_diff = d_diff - 360.0 * tl.floor((d_diff + 180.0) / 360.0)
        omega = tl.where(
            t_idx > 0, (wrapped_diff / dt) * (3.141592653589793 / 180.0), 0.0
        )
        abs_omega = tl.abs(omega)

        # 4. Centripetal / Normal Acceleration: a_n = s * |omega|
        a_n = s * abs_omega
        tl.store(out_an_ptr + offset + t_idx, a_n, mask=mask)

        # 5. Differential Curvature: kappa = |omega| / (s + 0.01)
        curv = abs_omega / (s + 0.01)
        tl.store(out_curv_ptr + offset + t_idx, curv, mask=mask)

        # 6. Centripetal Kinetic Flux: Flux = a_n * s
        flux = a_n * s
        tl.store(out_flux_ptr + offset + t_idx, flux, mask=mask)


def fused_differential_geometry(
    speed: Union[np.ndarray, torch.Tensor],
    accel: Union[np.ndarray, torch.Tensor],
    direction: Union[np.ndarray, torch.Tensor],
    seq_lens: Union[np.ndarray, torch.Tensor],
    max_len: int = 256,
    dt: float = 0.1,
    device: Optional[torch.device] = None,
    prefer_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute differential geometry computation with Triton GPU acceleration or CPU fallback.

    Args:
        speed: Speed array or tensor of shape (N, max_len)
        accel: Acceleration array or tensor of shape (N, max_len)
        direction: Direction array or tensor of shape (N, max_len)
        seq_lens: Sequence lengths array or tensor of shape (N,)
        max_len: Maximum sequence length (default: 256)
        dt: Delta time in seconds (default: 0.1)
        device: Torch device (defaults to cuda:0 if available, else cpu)
        prefer_triton: Whether to prefer Triton if installed and on GPU

    Returns:
        Tuple of (jerk, curvature, normal_accel, power, flux) as PyTorch tensors.
    """
    if device is None:
        device = torch.device("cuda:0" if HAS_CUDA else "cpu")

    # Convert inputs to torch tensors on target device
    if isinstance(speed, np.ndarray):
        s_t = torch.tensor(speed, device=device, dtype=torch.float32).contiguous()
    else:
        s_t = speed.to(device).contiguous().float()

    if isinstance(accel, np.ndarray):
        a_t = torch.tensor(accel, device=device, dtype=torch.float32).contiguous()
    else:
        a_t = accel.to(device).contiguous().float()

    if isinstance(direction, np.ndarray):
        d_t = torch.tensor(direction, device=device, dtype=torch.float32).contiguous()
    else:
        d_t = direction.to(device).contiguous().float()

    if isinstance(seq_lens, np.ndarray):
        lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32).contiguous()
    else:
        lens_t = seq_lens.to(device).contiguous().int()

    num_seqs = s_t.shape[0]

    # Attempt Triton GPU dispatch
    can_use_triton = prefer_triton and HAS_TRITON and HAS_CUDA and (s_t.device.type == "cuda")

    if can_use_triton:
        out_jerk = torch.empty_like(s_t)
        out_curv = torch.empty_like(s_t)
        out_an = torch.empty_like(s_t)
        out_power = torch.empty_like(s_t)
        out_flux = torch.empty_like(s_t)

        grid = (num_seqs,)
        fused_trajectory_differential_geometry_kernel[grid](
            s_t,
            a_t,
            d_t,
            out_jerk,
            out_curv,
            out_an,
            out_power,
            out_flux,
            lens_t,
            max_len,
            dt,
            BLOCK_SIZE=256,
        )
        return out_jerk, out_curv, out_an, out_power, out_flux

    # Fallback to PyTorch vectorized computation
    return compute_differential_geometry_torch(s_t, a_t, d_t, dt=dt, eps=0.01)


def benchmark_kernel(
    speed_arr: np.ndarray,
    accel_arr: np.ndarray,
    dir_arr: np.ndarray,
    seq_lens: np.ndarray,
    dt: float = 0.1,
    max_len: int = 256,
) -> Dict[str, float]:
    """Benchmark differential geometry execution comparing GPU kernel vs CPU baseline.

    Args:
        speed_arr: (N, max_len) numpy array
        accel_arr: (N, max_len) numpy array
        dir_arr: (N, max_len) numpy array
        seq_lens: (N,) sequence lengths
        dt: Delta time in seconds
        max_len: Maximum length

    Returns:
        Dict with execution times (ms), speedup factor, and frames per second.
    """
    num_seqs = len(speed_arr)
    total_frames = int(np.sum(seq_lens))

    # CPU benchmark on subset
    n_sample = min(200, num_seqs)
    t_cpu_start = time.perf_counter()
    for i in range(n_sample):
        L = seq_lens[i]
        s = speed_arr[i, :L]
        a = accel_arr[i, :L]
        d = dir_arr[i, :L]
        compute_differential_geometry_cpu(s, a, d, dt=dt)
    t_cpu_end = time.perf_counter()

    cpu_sample_time = t_cpu_end - t_cpu_start
    projected_cpu_time_ms = (cpu_sample_time / max(1, n_sample) * num_seqs) * 1000.0

    # Kernel execution (Triton GPU if available, else PyTorch CPU)
    device = torch.device("cuda:0" if (HAS_CUDA and HAS_TRITON) else "cpu")
    s_t = torch.tensor(speed_arr, device=device)
    a_t = torch.tensor(accel_arr, device=device)
    d_t = torch.tensor(dir_arr, device=device)
    lens_t = torch.tensor(seq_lens, device=device)

    # Warm-up run
    fused_differential_geometry(s_t, a_t, d_t, lens_t, max_len=max_len, dt=dt, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_kernel_start = time.perf_counter()
    fused_differential_geometry(s_t, a_t, d_t, lens_t, max_len=max_len, dt=dt, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_kernel_end = time.perf_counter()

    kernel_time_ms = (t_kernel_end - t_kernel_start) * 1000.0
    speedup = projected_cpu_time_ms / max(kernel_time_ms, 1e-6)
    throughput_fps = total_frames / max((t_kernel_end - t_kernel_start), 1e-6)

    return {
        "num_sequences": float(num_seqs),
        "total_frames": float(total_frames),
        "kernel_time_ms": kernel_time_ms,
        "cpu_projected_time_ms": projected_cpu_time_ms,
        "speedup_factor": speedup,
        "throughput_fps": throughput_fps,
        "used_triton": float(HAS_TRITON and device.type == "cuda"),
    }
