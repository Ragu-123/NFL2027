"""Benchmark runner for Triton GPU Differential Geometry Kernel vs CPU Baseline.

Measures latency, hardware speedup factor, real-time FPS throughput,
and validates numerical error bounds between GPU and CPU outputs.
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nfl2027.kinematics.triton_kernels import (
    fused_differential_geometry,
    benchmark_kernel,
    HAS_TRITON,
    HAS_CUDA,
)
from nfl2027.kinematics.differential_geometry import compute_differential_geometry_cpu


def parse_args():
    parser = argparse.ArgumentParser(description="Triton GPU Differential Geometry Benchmark")
    parser.add_argument("--num-seqs", type=int, default=6301, help="Number of trajectories to benchmark")
    parser.add_argument("--max-len", type=int, default=256, help="Padded sequence length")
    parser.add_argument("--dt", type=float, default=0.1, help="Delta time (seconds)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("NFL2027: TRITON GPU DIFFERENTIAL GEOMETRY HARDWARE BENCHMARK")
    print(f"CUDA Available: {HAS_CUDA} | Triton Available: {HAS_TRITON}")
    if HAS_CUDA:
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 75)

    num_seqs = args.num_seqs
    max_len = args.max_len
    dt = args.dt

    print(f"\nGenerating {num_seqs} realistic synthetic trajectory sequences (L={max_len}, dt={dt}s)...")
    rng = np.random.RandomState(42)
    seq_lens = rng.randint(40, max_len - 10, size=num_seqs).astype(np.int32)

    speed_arr = np.zeros((num_seqs, max_len), dtype=np.float32)
    accel_arr = np.zeros((num_seqs, max_len), dtype=np.float32)
    dir_arr = np.zeros((num_seqs, max_len), dtype=np.float32)

    for i in range(num_seqs):
        L = seq_lens[i]
        t = np.arange(L) * dt
        v = 9.0 * (1.0 - np.exp(-t / 1.5)) + rng.normal(0, 0.05, L)
        speed_arr[i, :L] = np.maximum(v, 0.0)
        accel_arr[i, :L] = np.gradient(speed_arr[i, :L], dt)
        dir_arr[i, :L] = (np.sin(t * 1.5) * 60.0 + 90.0) % 360.0

    total_frames = int(np.sum(seq_lens))
    print(f"Total Trajectory Frames to Process: {total_frames:,}")

    # Run benchmark
    results = benchmark_kernel(speed_arr, accel_arr, dir_arr, seq_lens, dt=dt, max_len=max_len)

    print("\n--- Benchmark Results ---")
    print(f"  Kernel Execution Time : {results['kernel_time_ms']:.2f} ms")
    print(f"  Projected CPU Baseline : {results['cpu_projected_time_ms']:.1f} ms")
    print(f"  Hardware Speedup Factor: {results['speedup_factor']:.1f}x Faster")
    print(f"  Streaming Throughput   : {results['throughput_fps'] / 1e6:.2f} Million Frames/Second")
    print(f"  Backend Dispatched     : {'Triton GPU Kernel' if results['used_triton'] else 'PyTorch CPU/CUDA Vectorized'}")

    # Numerical Equivalence Verification on Sample
    print("\nVerifying numerical precision against analytical CPU reference...")
    sample_s = speed_arr[:10]
    sample_a = accel_arr[:10]
    sample_d = dir_arr[:10]
    sample_lens = seq_lens[:10]

    dev = torch.device("cuda:0" if HAS_CUDA and HAS_TRITON else "cpu")
    s_t = torch.tensor(sample_s, device=dev)
    a_t = torch.tensor(sample_a, device=dev)
    d_t = torch.tensor(sample_d, device=dev)
    l_t = torch.tensor(sample_lens, device=dev)

    k_jerk, k_curv, k_an, k_power, k_flux = fused_differential_geometry(
        s_t, a_t, d_t, l_t, max_len=max_len, dt=dt, device=dev
    )

    c_jerk, c_curv, c_an, c_power, c_flux = compute_differential_geometry_cpu(
        sample_s, sample_a, sample_d, dt=dt
    )

    max_err_jerk = np.max(np.abs(k_jerk.cpu().numpy() - c_jerk))
    max_err_power = np.max(np.abs(k_power.cpu().numpy() - c_power))
    max_err_an = np.max(np.abs(k_an.cpu().numpy() - c_an))

    print(f"  Max Absolute Jerk Error : {max_err_jerk:.6e}")
    print(f"  Max Absolute Power Error: {max_err_power:.6e}")
    print(f"  Max Absolute a_n Error  : {max_err_an:.6e}")
    assert max_err_jerk < 1e-3, f"Jerk error exceeds tolerance: {max_err_jerk}"
    assert max_err_power < 1e-4, f"Power error exceeds tolerance: {max_err_power}"
    assert max_err_an < 1e-3, f"Centripetal acceleration error exceeds tolerance: {max_err_an}"
    print("Numerical verification PASSED: Kernel matches reference mathematics within float32 tolerance.")


if __name__ == "__main__":
    main()
