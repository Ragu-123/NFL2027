"""Executable entry-point for NFL2027 (Kaggle Cloud & Local Workstations).

NK-TrajNet: Neuro-Kinematic Trajectory Transformer & Differential Geometry.
Authentic High-Accuracy Combine-to-Game Translation Engine (BDB 2027).

Can be run directly via:
    python code.py
"""

import sys
import os

# Guard against Python standard library 'code' module collision with ./code.py
if __name__ == "code":
    import importlib.util
    stdlib_dir = os.path.dirname(os.__file__)
    stdlib_code_path = os.path.join(stdlib_dir, "code.py")
    if os.path.exists(stdlib_code_path):
        spec = importlib.util.spec_from_file_location("_stdlib_code", stdlib_code_path)
        _stdlib_code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_stdlib_code)
        for _attr in dir(_stdlib_code):
            globals()[_attr] = getattr(_stdlib_code, _attr)
else:
    # Add current directory to path for nfl2027 package discovery
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import time
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.stats import pearsonr, spearmanr
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    warnings.filterwarnings("ignore")

    # Attempt modular import
    try:
        from nfl2027.config import Config, detect_data_dir, detect_output_dir
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
        MODULAR_AVAILABLE = True
    except ImportError:
        MODULAR_AVAILABLE = False


    def main():
        print("=" * 80)
        print("NK-TrajNet: NEURO-KINEMATIC TRAJECTORY TRANSFORMER & DIFFERENTIAL GEOMETRY")
        print("AUTHENTIC HIGH-ACCURACY COMBINE-TO-GAME TRANSLATION ENGINE (BDB 2027)")
        print("=" * 80)

        # 1. Environment & Path Detection
        detected_dir = detect_data_dir() if MODULAR_AVAILABLE else None
        data_dir = detected_dir or "/kaggle/input/competitions/nfl-big-data-bowl-2027/nfl-big-data-bowl-2027"
        fig_dir = detect_output_dir() if MODULAR_AVAILABLE else "/kaggle/working"
        os.makedirs(fig_dir, exist_ok=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Compute Device: {device} | Output Directory: {fig_dir}")

        # 2. Data Loading & Ingestion
        print("\n[Step 1/5] Ingesting competition datasets and player tracking...")
        if os.path.exists(os.path.join(data_dir, "players.csv")):
            raw_tables = load_raw_data(data_dir)
            df_players = raw_tables["players"]
            df_combine_res = raw_tables["combine_results"]
            df_success = raw_tables["player_career_successes"]
            df_comb_trk = raw_tables["combine_tracking"]
            df_pp = raw_tables["player_play"]
        else:
            print(f"Data directory '{data_dir}' not found on local filesystem.")
            print("Generating realistic synthetic dataset for local end-to-end execution...")
            raw_tables = generate_synthetic_dataset(num_players=70, drills_per_player=4, random_seed=42)
            df_players = raw_tables["players"]
            df_combine_res = raw_tables["combine_results"]
            df_success = raw_tables["player_career_successes"]
            df_comb_trk = raw_tables["combine_tracking"]
            df_pp = raw_tables["player_play"]

        df_meta = merge_metadata(df_players, df_combine_res, df_success)

        # 3. Assemble Tracking Sequences
        speed_arr, accel_arr, dir_arr, seq_lens, meta_keys, seq_groups = prepare_tracking_sequences(
            df_comb_trk, max_len=256
        )
        num_seqs = len(seq_lens)
        print(f"Tracking sequences assembled: {num_seqs} distinct drill runs across {df_meta['nfl_id'].nunique()} prospects.")

        # 4. Triton GPU Differential Geometry / Fallback
        print("\n[Step 2/5] Executing Differential Geometry Kernel...")
        s_t = torch.tensor(speed_arr, device=device)
        a_t = torch.tensor(accel_arr, device=device)
        d_t = torch.tensor(dir_arr, device=device)
        lens_t = torch.tensor(seq_lens, device=device)

        out_jerk, out_curv, out_an, out_power, out_flux = fused_differential_geometry(
            s_t, a_t, d_t, lens_t, max_len=256, dt=0.1, device=device
        )

        bench_metrics = benchmark_kernel(speed_arr, accel_arr, dir_arr, seq_lens, dt=0.1)
        print(f"Kernel executed across {num_seqs} sequences in {bench_metrics['kernel_time_ms']:.2f} ms!")
        print(f"Projected CPU Execution: {bench_metrics['cpu_projected_time_ms']:.1f} ms | Hardware Speedup: {bench_metrics['speedup_factor']:.1f}x Faster!")
        print(f"Real-Time Streaming Throughput: {bench_metrics['throughput_fps'] / 1e6:.2f} Million Frames/Second!")

        # 5. Neuro-Kinematic Temporal Attention Encoder
        print("\n[Step 3/5] Extracting continuous trajectory embeddings via Temporal Attention Network...")
        traj_tensor = torch.stack([s_t, a_t, out_jerk, d_t, out_an, out_curv, out_power], dim=1)

        forty_seq_indices = []
        forty_players = []
        for idx, (nfl_id, d_type, d_name, att) in enumerate(meta_keys):
            if d_type == "FORTY_YARD_DASH" and nfl_id not in forty_players:
                forty_seq_indices.append(idx)
                forty_players.append(nfl_id)

        encoder = TrajectoryTemporalAttentionEncoder().to(device)
        if forty_seq_indices:
            forty_x = traj_tensor[forty_seq_indices, :, :100]
            forty_lens = lens_t[forty_seq_indices]
            pad_mask = torch.arange(100, device=device).unsqueeze(0) >= forty_lens.unsqueeze(1)
            encoder.eval()
            with torch.no_grad():
                forty_latent_emb, forty_alpha = encoder(forty_x, mask=pad_mask)
            forty_latent_cpu = forty_latent_emb.cpu().numpy()
            forty_alpha_cpu = forty_alpha.cpu().numpy()
            df_forty_emb = pd.DataFrame(forty_latent_cpu, columns=[f"emb_{k}" for k in range(32)])
            df_forty_emb["nfl_id"] = forty_players
        else:
            df_forty_emb = None
            forty_alpha_cpu = np.ones((1, 100)) / 100.0

        # Extract scalar invariants & build master matrix
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
            dt=0.1,
        )
        master_df = aggregate_player_features(df_feat, df_meta, df_forty_emb)
        print(f"Master feature matrix constructed: {master_df.shape[0]} prospects x {master_df.shape[1]} physical metrics.")

        # 6. Multi-Task 5-Fold Cross-Validation Benchmark
        print("\n[Step 4/5] Evaluating competing models across NFL translation tasks...")
        targets = prepare_play_targets(df_pp, master_df)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        benchmark_results = []

        tasks = [
            ("dl_get_off", "Pass Rusher Snap Get-Off (s)", "DL Get-Off", "target_get_off", get_dl_get_off_models()),
            ("dl_pressure_rate", "Pass Rush Pressure Rate", "Pressure Rate", "pressure_rate", get_pressure_rate_models()),
            ("wr_cushion", "WR Cornerback Cushion Respect (yds)", "WR Cushion", "mean_cushion", get_wr_cushion_models()),
        ]

        for key, task_name, short_name, target_col, models in tasks:
            if key not in targets or len(targets[key]) < 10:
                continue

            df_task = targets[key]
            y = df_task[target_col].values.astype(np.float64)

            entry = {
                "Task": task_name,
                "ShortName": short_name,
                "N": len(y),
                "actual": y,
            }

            m_eval = {}
            for m_id in ["M0", "M1", "M2"]:
                cols, model = models[m_id]
                valid_cols = [c for c in cols if c in df_task.columns]
                if not valid_cols:
                    continue
                X = df_task[valid_cols].fillna(df_task[valid_cols].median()).values
                eval_res = evaluate_pipeline(X, y, model, cv=kf)
                m_eval[m_id] = eval_res
                entry[f"{m_id}_R2"] = eval_res["r2"]
                entry[f"{m_id}_RMSE"] = eval_res["rmse"]
                entry[f"{m_id}_r"] = eval_res["pearson_r"]
                entry[f"pred_{m_id.lower()}"] = eval_res["predictions"]

            if "M0" in m_eval and "M2" in m_eval:
                entry["Delta_R2"] = m_eval["M2"]["r2"] - m_eval["M0"]["r2"]
                entry["RMSE_Reduction_Pct"] = (
                    (m_eval["M0"]["rmse"] - m_eval["M2"]["rmse"]) / max(m_eval["M0"]["rmse"], 1e-6)
                ) * 100.0

            benchmark_results.append(entry)

            print(f"\n--- {task_name} (N={len(y)}) ---")
            if "M0" in m_eval:
                print(f"  M0 (Stopwatch Baseline) : R2 = {m_eval['M0']['r2']:.4f} | RMSE = {m_eval['M0']['rmse']:.4f} | Pearson r = {m_eval['M0']['pearson_r']:.3f}")
            if "M1" in m_eval:
                print(f"  M1 (Naive Tabular Kin.) : R2 = {m_eval['M1']['r2']:.4f} | RMSE = {m_eval['M1']['rmse']:.4f} | Pearson r = {m_eval['M1']['pearson_r']:.3f}")
            if "M2" in m_eval:
                print(f"  M2 (NK-TrajNet Proposed): R2 = {m_eval['M2']['r2']:.4f} | RMSE = {m_eval['M2']['rmse']:.4f} | Pearson r = {m_eval['M2']['pearson_r']:.3f}")
                if "M0" in m_eval:
                    pct_str = f" (+{entry['Delta_R2']/max(abs(m_eval['M0']['r2']), 1e-4)*100:.1f}%)" if m_eval['M0']['r2'] > 0 else ""
                    print(f"  Accuracy Gain: Delta R2 = +{entry['Delta_R2']:.4f}{pct_str} | RMSE Reduction = +{entry['RMSE_Reduction_Pct']:.1f}%")

        # 7. Figure Generation
        if benchmark_results:
            print("\n[Step 5/5] Generating publication-grade figures...")
            fig1_path = os.path.join(fig_dir, "model_benchmark_accuracy.png")
            plot_model_benchmarks(benchmark_results, save_path=fig1_path)

            fig2_path = os.path.join(fig_dir, "triton_kinematic_acceleration.png")
            plot_hardware_acceleration(bench_metrics, save_path=fig2_path)

            fig3_path = os.path.join(fig_dir, "phase_portrait_frenet_serret.png")
            plot_phase_portraits_and_attention(df_feat, forty_alpha_cpu, dt=0.1, save_path=fig3_path)

            fig4_path = os.path.join(fig_dir, "in_game_linkage_validation.png")
            plot_linkage_validation(benchmark_results, save_path=fig4_path)

        print("\n" + "=" * 80)
        print("EXPERIMENT SUCCESSFULLY COMPLETED: ALL BENCHMARKS & FIGURES GENERATED")
        print("=" * 80)


    if __name__ == "__main__":
        main()
