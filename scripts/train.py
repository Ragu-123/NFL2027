"""CLI Training & Evaluation Pipeline for NFL2027.

Executes end-to-end data ingestion, GPU/CPU differential geometry,
neural temporal attention encoding, multi-task 5-fold cross-validation,
and publication figure generation.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

# Ensure package is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nfl2027.config import Config, get_config, detect_data_dir, detect_output_dir
from nfl2027.data.loaders import (
    load_raw_data,
    merge_metadata,
    prepare_tracking_sequences,
    prepare_play_targets,
    generate_synthetic_dataset,
)
from nfl2027.kinematics.triton_kernels import (
    fused_differential_geometry,
    benchmark_kernel,
    HAS_TRITON,
    HAS_CUDA,
)
from nfl2027.kinematics.differential_geometry import (
    extract_kinematic_invariants,
    aggregate_player_features,
)
from nfl2027.models.encoder import TrajectoryTemporalAttentionEncoder
from nfl2027.models.regressors import (
    get_dl_get_off_models,
    get_pressure_rate_models,
    get_wr_cushion_models,
)
from nfl2027.evaluation.metrics import evaluate_pipeline
from nfl2027.visualization.plots import (
    plot_model_benchmarks,
    plot_hardware_acceleration,
    plot_phase_portraits_and_attention,
    plot_linkage_validation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="NFL2027 Model Training & Benchmark Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save figures and artifacts")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config(
        data_dir=args.data_dir or detect_data_dir(),
        output_dir=args.output_dir or detect_output_dir(),
        random_seed=args.seed,
    )

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("NFL2027: NEURO-KINEMATIC TRAJECTORY TRANSFORMER & DIFFERENTIAL GEOMETRY")
    print(f"Device: {device} | Triton Available: {HAS_TRITON} | CUDA Available: {HAS_CUDA}")
    print("=" * 80)

    # 1. Load Data
    use_synthetic = args.synthetic or (cfg.data_dir is None)
    if use_synthetic:
        print("\n[Step 1/5] Initializing synthetic dataset for demonstration...")
        raw_tables = generate_synthetic_dataset(num_players=70, drills_per_player=4, random_seed=cfg.random_seed)
    else:
        print(f"\n[Step 1/5] Ingesting competition datasets from {cfg.data_dir}...")
        raw_tables = load_raw_data(cfg.data_dir)

    df_meta = merge_metadata(
        raw_tables["players"],
        raw_tables["combine_results"],
        raw_tables["player_career_successes"],
    )
    print(f"Prospect cohort: {len(df_meta)} athletes merged across Combine and Career records.")

    # 2. Tracking Sequences & Differential Geometry
    print("\n[Step 2/5] Assembling trajectory sequences and running differential geometry...")
    speed_arr, accel_arr, dir_arr, seq_lens, meta_keys, seq_groups = prepare_tracking_sequences(
        raw_tables["combine_tracking"], max_len=cfg.max_seq_len
    )
    print(f"Assembled {len(seq_lens)} distinct drill runs.")

    # Execute fused differential geometry
    s_t = torch.tensor(speed_arr, device=device)
    a_t = torch.tensor(accel_arr, device=device)
    d_t = torch.tensor(dir_arr, device=device)
    lens_t = torch.tensor(seq_lens, device=device)

    out_jerk, out_curv, out_an, out_power, out_flux = fused_differential_geometry(
        s_t, a_t, d_t, lens_t, max_len=cfg.max_seq_len, dt=cfg.dt, device=device
    )

    bench_metrics = benchmark_kernel(speed_arr, accel_arr, dir_arr, seq_lens, dt=cfg.dt)
    print(f"Kernel execution time: {bench_metrics['kernel_time_ms']:.2f} ms")
    print(f"Projected CPU baseline: {bench_metrics['cpu_projected_time_ms']:.1f} ms")
    print(f"Speedup factor: {bench_metrics['speedup_factor']:.1f}x Faster")

    # 3. Neural Temporal Attention Encoding
    print("\n[Step 3/5] Extracting continuous trajectory embeddings via TrajectoryTemporalAttentionEncoder...")
    traj_tensor = torch.stack([s_t, a_t, out_jerk, d_t, out_an, out_curv, out_power], dim=1)

    forty_seq_indices = []
    forty_players = []
    for idx, (nfl_id, d_type, d_name, att) in enumerate(meta_keys):
        if d_type == "FORTY_YARD_DASH" and nfl_id not in forty_players:
            forty_seq_indices.append(idx)
            forty_players.append(nfl_id)

    encoder = TrajectoryTemporalAttentionEncoder(
        in_channels=cfg.in_channels,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
    ).to(device)

    if forty_seq_indices:
        forty_x = traj_tensor[forty_seq_indices, :, :100]
        forty_lens = lens_t[forty_seq_indices]
        pad_mask = torch.arange(100, device=device).unsqueeze(0) >= forty_lens.unsqueeze(1)
        encoder.eval()
        with torch.no_grad():
            forty_emb, forty_alpha = encoder(forty_x, mask=pad_mask)
        forty_emb_np = forty_emb.cpu().numpy()
        forty_alpha_np = forty_alpha.cpu().numpy()
        df_forty_emb = pd.DataFrame(forty_emb_np, columns=[f"emb_{k}" for k in range(cfg.hidden_dim)])
        df_forty_emb["nfl_id"] = forty_players
    else:
        df_forty_emb = None
        forty_alpha_np = np.ones((1, 100)) / 100.0

    # Extract scalar invariants and construct master feature matrix
    df_feat = extract_kinematic_invariants(
        speed_arr,
        accel_arr,
        out_jerk.cpu().numpy(),
        out_curv.cpu().numpy(),
        out_an.cpu().numpy(),
        out_power.cpu().numpy(),
        out_flux.cpu().numpy(),
        seq_lens,
        meta_keys,
        dt=cfg.dt,
    )
    master_df = aggregate_player_features(df_feat, df_meta, df_forty_emb)
    print(f"Master feature matrix constructed: {master_df.shape[0]} prospects x {master_df.shape[1]} physical metrics.")

    # 4. Multi-Task Translation Benchmark (5-Fold Cross-Validation)
    print("\n[Step 4/5] Evaluating competing models across NFL translation tasks...")
    targets = prepare_play_targets(raw_tables["player_play"], master_df)

    benchmark_tasks = [
        ("dl_get_off", "Pass Rusher Snap Get-Off (s)", "DL Get-Off", "target_get_off", get_dl_get_off_models()),
        ("dl_pressure_rate", "Pass Rush Pressure Rate", "Pressure Rate", "pressure_rate", get_pressure_rate_models()),
        ("wr_cushion", "WR Route Cushion Respect (yds)", "WR Cushion", "mean_cushion", get_wr_cushion_models()),
    ]

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_seed)
    benchmark_results = []

    for key, task_title, short_name, target_col, models in benchmark_tasks:
        if key not in targets or len(targets[key]) < 10:
            print(f"Skipping {task_title} (insufficient matching samples in dataset: {len(targets.get(key, []))})")
            continue

        df_task = targets[key]
        y = df_task[target_col].values.astype(np.float64)

        task_entry = {
            "Task": task_title,
            "ShortName": short_name,
            "N": len(y),
            "actual": y,
        }

        m_scores = {}
        for m_name in ["M0", "M1", "M2"]:
            cols, model = models[m_name]
            # Select available columns
            valid_cols = [c for c in cols if c in df_task.columns]
            if not valid_cols:
                continue
            X = df_task[valid_cols].fillna(df_task[valid_cols].median()).values
            eval_res = evaluate_pipeline(X, y, model, cv=kf)
            m_scores[m_name] = eval_res
            task_entry[f"{m_name}_R2"] = eval_res["r2"]
            task_entry[f"{m_name}_RMSE"] = eval_res["rmse"]
            task_entry[f"{m_name}_r"] = eval_res["pearson_r"]
            task_entry[f"pred_{m_name.lower()}"] = eval_res["predictions"]

        if "M0" in m_scores and "M2" in m_scores:
            task_entry["Delta_R2"] = m_scores["M2"]["r2"] - m_scores["M0"]["r2"]
            task_entry["RMSE_Reduction_Pct"] = (
                (m_scores["M0"]["rmse"] - m_scores["M2"]["rmse"]) / max(m_scores["M0"]["rmse"], 1e-6)
            ) * 100.0

        benchmark_results.append(task_entry)

        print(f"\n--- {task_title} (N={len(y)}) ---")
        if "M0" in m_scores:
            print(f"  M0 (Stopwatch Baseline) : R2 = {m_scores['M0']['r2']:.4f} | RMSE = {m_scores['M0']['rmse']:.4f} | r = {m_scores['M0']['pearson_r']:.3f}")
        if "M1" in m_scores:
            print(f"  M1 (Naive Tabular Kin.) : R2 = {m_scores['M1']['r2']:.4f} | RMSE = {m_scores['M1']['rmse']:.4f} | r = {m_scores['M1']['pearson_r']:.3f}")
        if "M2" in m_scores:
            print(f"  M2 (NK-TrajNet Proposed): R2 = {m_scores['M2']['r2']:.4f} | RMSE = {m_scores['M2']['rmse']:.4f} | r = {m_scores['M2']['pearson_r']:.3f}")
            if "M0" in m_scores:
                print(f"  Information Gain: Delta R2 = +{task_entry['Delta_R2']:.4f} | RMSE Reduction = +{task_entry['RMSE_Reduction_Pct']:.1f}%")

    # 5. Publication Figure Generation
    if not args.no_figures and benchmark_results:
        print(f"\n[Step 5/5] Generating publication figures in {cfg.output_dir}...")
        os.makedirs(cfg.output_dir, exist_ok=True)

        fig1_path = os.path.join(cfg.output_dir, "model_benchmark_accuracy.png")
        plot_model_benchmarks(benchmark_results, save_path=fig1_path)
        print(f"  Saved Figure 1: {fig1_path}")

        fig2_path = os.path.join(cfg.output_dir, "triton_kinematic_acceleration.png")
        plot_hardware_acceleration(bench_metrics, save_path=fig2_path)
        print(f"  Saved Figure 2: {fig2_path}")

        fig3_path = os.path.join(cfg.output_dir, "phase_portrait_frenet_serret.png")
        plot_phase_portraits_and_attention(df_feat, forty_alpha_np, dt=cfg.dt, save_path=fig3_path)
        print(f"  Saved Figure 3: {fig3_path}")

        fig4_path = os.path.join(cfg.output_dir, "in_game_linkage_validation.png")
        plot_linkage_validation(benchmark_results, save_path=fig4_path)
        print(f"  Saved Figure 4: {fig4_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
