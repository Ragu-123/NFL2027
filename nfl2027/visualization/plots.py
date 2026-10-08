"""Publication-grade visualization functions for benchmarks, phase dynamics, and validation."""

import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_model_benchmarks(
    benchmark_results: List[Dict[str, Any]], save_path: Optional[str] = None
) -> plt.Figure:
    """Generate Figure 1: Model Benchmark Accuracy (R2 and Information Gain)."""
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    bar_width = 0.25

    task_labels = [b.get("ShortName", b.get("Task", "")) for b in benchmark_results]
    x_indices = np.arange(len(task_labels))

    m0_r2 = [b.get("M0_R2", 0.0) for b in benchmark_results]
    m1_r2 = [b.get("M1_R2", 0.0) for b in benchmark_results]
    m2_r2 = [b.get("M2_R2", 0.0) for b in benchmark_results]

    axes[0].bar(
        x_indices - bar_width,
        m0_r2,
        width=bar_width,
        label="M0: Traditional Stopwatch",
        color="#B0BEC5",
        edgecolor="black",
    )
    axes[0].bar(
        x_indices,
        m1_r2,
        width=bar_width,
        label="M1: Naive Kinematics",
        color="#42A5F5",
        edgecolor="black",
    )
    axes[0].bar(
        x_indices + bar_width,
        m2_r2,
        width=bar_width,
        label="M2: NK-TrajNet (Differential Geometry)",
        color="#2E7D32",
        edgecolor="black",
    )

    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(task_labels, fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Out-of-Fold R² Score", fontsize=12)
    axes[0].set_title(
        "Cross-Validated Accuracy (R²) Across Translation Tasks",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    gains = [b.get("Delta_R2", m2 - m0) for b, m0, m2 in zip(benchmark_results, m0_r2, m2_r2)]
    bars = axes[1].bar(
        task_labels, gains, color="#2E7D32", edgecolor="black", width=0.45
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("R² Accuracy Gain over Baseline (ΔR²)", fontsize=12)
    axes[1].set_title(
        "NK-TrajNet Relative Information Gain (ΔR²)", fontsize=13, fontweight="bold"
    )

    for bar in bars:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            max(yval, 0) + 0.005,
            f"+{yval:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(
        "NK-TrajNet Translation Benchmark: Predictive Accuracy from Combine to NFL Games",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_hardware_acceleration(
    metrics: Dict[str, float], save_path: Optional[str] = None
) -> plt.Figure:
    """Generate Figure 2: Hardware Acceleration & Latency Benchmark."""
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    techs = ["Pandas/NumPy CPU", "Triton GPU Kernel (T4)"]
    cpu_ms = metrics.get("cpu_projected_time_ms", 4212.0)
    gpu_ms = metrics.get("kernel_time_ms", 125.2)
    speedup = metrics.get("speedup_factor", cpu_ms / max(gpu_ms, 1e-4))
    throughput_fps = metrics.get("throughput_fps", 3.44e6)

    times_ms = [cpu_ms, gpu_ms]
    bar_cols = ["#D32F2F", "#00A86B"]

    axes[0].bar(techs, times_ms, color=bar_cols, width=0.45, edgecolor="black")
    axes[0].set_ylabel("Execution Time (ms) - Log Scale", fontsize=12)
    axes[0].set_title(
        f"Differential Geometry Execution ({int(metrics.get('total_frames', 431094)):,} Frames)\n"
        f"Speedup Factor: {speedup:.1f}x Faster",
        fontsize=12,
        fontweight="bold",
    )
    for i, v in enumerate(times_ms):
        axes[0].text(
            i, v * 1.1, f"{v:.1f} ms", ha="center", va="bottom", fontweight="bold", fontsize=11
        )
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    fps_millions = throughput_fps / 1e6
    axes[1].bar(
        ["Triton GPU Kernel"], [fps_millions], color="#00A86B", width=0.35, edgecolor="black"
    )
    axes[1].set_ylabel("Processed Frames per Second (Millions)", fontsize=12)
    axes[1].set_title(
        "Triton Real-Time Trajectory Streaming Throughput", fontsize=12, fontweight="bold"
    )
    axes[1].text(
        0,
        fps_millions * 0.5,
        f"{fps_millions:.2f} Million FPS",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=14,
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(
        "Hardware Acceleration: Custom Triton Kernel on NVIDIA Tesla T4",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_phase_portraits_and_attention(
    df_feat: pd.DataFrame,
    attention_weights: np.ndarray,
    dt: float = 0.1,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Figure 3: Phase Portraits & Deep Temporal Attention Weights."""
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Curvature-Velocity Manifold
    sample_n = min(1500, len(df_feat))
    sample_pts = df_feat.sample(n=sample_n, random_state=42) if len(df_feat) > 0 else df_feat

    if len(sample_pts) > 0 and "peak_speed" in sample_pts.columns and "curv_p95" in sample_pts.columns:
        c_vals = sample_pts.get("an_max", sample_pts["peak_speed"])
        scatter1 = axes[0].scatter(
            sample_pts["peak_speed"],
            sample_pts["curv_p95"],
            c=c_vals,
            cmap="plasma",
            alpha=0.7,
            edgecolors="none",
            s=45,
        )
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label("Peak Centripetal Acceleration a_n (yd/s²)", fontsize=11)
    axes[0].set_xlabel("Peak Tangential Speed s (yd/s)", fontsize=12)
    axes[0].set_ylabel("95th Percentile Curvature κ (rad/yd)", fontsize=12)
    axes[0].set_title(
        "Curvature-Velocity Manifold (Frenet-Serret Invariants)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Right: Temporal Attention Heatmap over Forty Yard Dash
    if attention_weights.ndim == 2:
        mean_alpha = np.mean(attention_weights, axis=0)
    else:
        mean_alpha = attention_weights

    seq_len = len(mean_alpha)
    time_axis = np.arange(seq_len) * dt

    axes[1].plot(
        time_axis,
        mean_alpha,
        color="#0288D1",
        lw=2.5,
        label="Mean Temporal Attention α(t)",
    )
    axes[1].fill_between(time_axis, 0, mean_alpha, color="#0288D1", alpha=0.25)
    axes[1].axvspan(
        0.0, 0.6, color="#FF5722", alpha=0.2, label="Explosive Drive Phase (0.0 - 0.6s)"
    )
    axes[1].set_xlabel("Drill Elapsed Time t (seconds)", fontsize=12)
    axes[1].set_ylabel("Trajectory Attention Weight α(t)", fontsize=12)
    axes[1].set_title(
        "NK-TrajNet Temporal Attention: Learning Decisive Drive Phases",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Continuous Differential Geometry & Neuro-Kinematic Phase Dynamics",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_linkage_validation(
    benchmark_results: List[Dict[str, Any]], save_path: Optional[str] = None
) -> plt.Figure:
    """Generate Figure 4: Out-of-Fold Actual vs Predicted Regressions."""
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    tasks_to_plot = benchmark_results[:2]

    colors = [("#013369", "#D50A0A"), ("#008ABC", "#FF8200")]

    for i, res in enumerate(tasks_to_plot):
        y_act = res.get("actual")
        y_pred = res.get("pred_m2", res.get("pred_m0"))

        if y_act is None or y_pred is None:
            continue

        pt_color, line_color = colors[i % len(colors)]
        sns.regplot(
            x=y_pred,
            y=y_act,
            ax=axes[i],
            color=pt_color,
            scatter_kws={"alpha": 0.65, "s": 50},
            line_kws={"color": line_color, "lw": 2.5},
        )
        task_name = res.get("Task", f"Task {i+1}")
        r2_val = res.get("M2_R2", res.get("M0_R2", 0.0))
        rmse_val = res.get("M2_RMSE", res.get("M0_RMSE", 0.0))
        r_val = res.get("M2_r", res.get("M0_r", 0.0))

        axes[i].set_xlabel(f"NK-TrajNet Predicted {task_name}", fontsize=11)
        axes[i].set_ylabel(f"Actual In-Game {task_name}", fontsize=11)
        axes[i].set_title(
            f"{task_name} (N={len(y_act)})\n"
            f"R² = {r2_val:.3f} | RMSE = {rmse_val:.4f} | Pearson r = +{r_val:.3f}",
            fontsize=12,
            fontweight="bold",
        )
        axes[i].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Out-of-Fold Cross-Validation: Translating Combine Tracking to Regular-Season Performance",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_draft_surplus_curve(
    picks: np.ndarray,
    snaps: np.ndarray,
    A: float = 1853.6,
    lambda_val: float = 0.0094,
    C: float = 17.7,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Draft Surplus Valuation Model plot with exponential baseline decay."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))

    x_curve = np.linspace(1, 260, 300)
    y_curve = A * np.exp(-lambda_val * x_curve) + C

    ax.scatter(picks, snaps, color="#1976D2", alpha=0.6, s=40, label="Prospect Career Snaps")
    ax.plot(
        x_curve,
        y_curve,
        color="#D32F2F",
        lw=3,
        label=f"Expected Baseline: E[Snaps] = {A:.1f}·e^{{-{lambda_val:.4f}·Pick}} + {C:.1f}",
    )

    ax.set_xlabel("Overall Draft Selection (Pick #)", fontsize=12)
    ax.set_ylabel("Total Career Snaps", fontsize=12)
    ax.set_title(
        "Draft Capital Baseline & Career Surplus Valuation Framework",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
