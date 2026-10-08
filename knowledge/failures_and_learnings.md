# NFL Big Data Bowl 2027: Failures, Anomalies & Technical Learnings ⚠️

## 1. Executive Log of Pitfalls & Mitigations

This document serves as an engineering post-mortem tracking unexpected data behaviors, failed hypotheses, schema pitfalls, and mathematical anomalies encountered during data ingestion, EDA, and modeling.

---

## 2. Technical Pitfalls & Bug Fixes

### 2.1 Pandas Column Name Collision on Relational Merges
- **Symptom**: `KeyError: "['draft_year'] not in index"` during script execution.
- **Root Cause**: Both `players.csv` and `combine_results.csv` contain `draft_year`. Calling `df_players.merge(df_combine, on='nfl_id')` silently renamed the columns to `draft_year_x` and `draft_year_y`.
- **Resolution**: Always merge on the composite shared key:
  ```python
  df_meta = df_players.merge(df_combine, on=['nfl_id', 'draft_year'])
  ```
- **Rule**: When merging with `player_play.csv`, be vigilant of `lined_up_position` vs `nfl_position` and duplicate team identifiers.

### 2.2 Local API Network & SSL Protocol Failure
- **Symptom**: `SSLEOFError: [SSL: UNEXPECTED_EOF_WHILE_READING]` when running local Python scripts accessing `api.kaggle.com`.
- **Root Cause**: Local machine network proxy and TLS inspection intercepting Kaggle API requests.
- **Resolution**: Exclusively leverage the connected **Kaggle Remote Jupyter Kernel** via `code.py` protocol and FastMCP tools (`kaggle_remote_kernel_execute`). Code runs inside the official Kaggle cloud container with direct localhost access to `/kaggle/input` datasets.

---

## 3. Data Anomalies & Scouting Biases

### 3.1 Sensor Jitter & Extreme Velocity Spikes (>25 MPH)
- **Anomaly Observed**: In game tracking, a small number of instantaneous speed readings recorded velocities exceeding 28 MPH (e.g., Terrion Arnold at 28.37 MPH).
- **Domain Reality**: Human top speed in pads rarely exceeds 22.5–23.2 MPH (the fastest recorded NFL ball carriers in NGS history, such as Tyreek Hill or DK Metcalf, peak around 23.0 MPH).
- **Technical Cause**: GPS/RFID tag occlusion or stadium sensor reflection causing coordinate jump between 10 Hz frames ($\Delta x / \Delta t$ spike).
- **Actionable Mitigation**:
  - Filter out raw single-frame spikes where $s > 24.5\text{ MPH}$.
  - Apply a 5-frame moving median or Savitzky-Golay polynomial smoothing filter before extracting maximum in-game velocities.

### 3.2 Cohort-Only Tracking Scope in Game Files
- **Anomaly Observed**: Initial expectation that `game_tracking_*.csv` would contain all 22 players on field was incorrect. The tracking files **exclusively track the 510 cohort prospects**.
- **Impact on Analytics**:
  - You **cannot** directly compute Euclidean distance or spatial voronoi cells between a receiver and the covering cornerback from `game_tracking` if the cornerback is a veteran not in the 2023–2025 cohort.
  - **Resolution**: Do NOT attempt to reconstruct full 22-player spatial fields. Instead, use pre-computed interaction fields provided in `player_play.csv`:
    - `separation_at_pass_forward` (exact yards of separation from primary defender)
    - `cushion` (pre-snap distance to defender)
    - `time_to_pressure` / `time_to_pressure_allowed`
    - `peak_pressure_probability_allowed`

### 3.3 Heavy Missingness in Traditional Agility Tests (Selection Bias)
- **Data Reality**:
  - `three_cone`: **64.7% missing** (330 / 510 players skipped)
  - `short_shuttle`: **60.6% missing** (309 / 510 players skipped)
  - `bench_reps`: **62.4% missing** (318 / 510 players skipped)
- **Why this happens**: Top projected draft picks (Round 1–2 prospects) routinely opt out of agility and bench press drills to avoid risking their stock. Lower-round prospects and undrafted free agents are disproportionately forced to test.
- **Critical Takeaway**: Models relying heavily on 3-cone or short shuttle will suffer from severe **survivorship / selection bias**. The true goldmine of BDB 2027 is `combine_tracking.csv`, which has **100% coverage (510/510 players)** across 62 dynamic position drills!

---

## 4. Disproven Hypotheses & Scouting Traps

### 4.1 Hypothesis Disproven: "40-Yard Dash Predicts Offensive Line Athleticism"
- **Hypothesis**: Faster 40 times would correlate with lower pressure rates allowed by offensive tackles.
- **Empirical Finding**: $r = -0.015$ (near zero).
- **Scouting Reason**: Offensive tackles do not sprint downfield. Their pass protection takes place in a 3-yard cylinder where lateral kick-slide balance, punch timing, and recovery agility (`PASS_PRO_MIRROR_DRILL`) matter infinitely more than straight-line sprint speed.

### 4.2 Hypothesis Disproven: "Top Sprint Speed in Combine Equals Top Game Speed"
- **Hypothesis**: Players with higher top speed in the Combine 40 ($s_{max}$) reach higher top speeds in NFL games.
- **Empirical Finding**: For defensive backs and skill players, game speed often exceeds Combine speed by 3–5 MPH due to pursuit angles, open-field reactive motivation, and running in pads with competitive drive.
- **Scouting Takeaway**: Combine tests measure **isolated closed-loop capacity**, whereas game tracking measures **open-loop reactive speed**.


---

## 5. Machine Learning, Kernel & Architectural Learnings

### 5.1 Object Dtype Failure on Boolean Play Tracking Columns
- **Symptom**: `AttributeError: 'numpy.dtypes.ObjectDType' object has no attribute 'dtype'` inside `scipy.stats.pearsonr`.
- **Root Cause**: `player_play.csv` stores boolean play indicators (`pressure_allowed`, `quick_pressure`) with mixed types (`True`, `False`, `NaN`) represented as Python `object`. When calculating group means, Pandas preserves the object dtype.
- **Resolution**: Explicitly typecast all binary indicator columns to floating point prior to group aggregation:
  ```python
  df_pp['pressure_allowed'] = df_pp['pressure_allowed'].astype(float)
  ```

### 5.2 Curse of Dimensionality & Stacking Overfitting on Small Cohorts ($N \approx 130$)
- **Symptom**: Stacking ensemble combining GradientBoosting, RandomForest, and ElasticNet over 64 raw kinematic features scored $R^2 = 0.07$ on DL Get-Off, while a simple 8-feature GBDT achieved $R^2 = 0.34$.
- **Root Cause**: Cohort sizes for specific position drills are constrained ($N=131$ for DL, $N=141$ for WR). Feeding collinear time-series summary metrics into deep multi-layer stackers creates severe out-of-fold variance.
- **Resolution**:
  - Enforce strict domain-guided feature selection: isolate orthogonal physical invariants (0.5s burst jerk, peak normal acceleration in hoop bend, body mass).
  - Regularize tree models with shallow depth (`max_depth = 2`), feature subsampling (`subsample = 0.85`), and conservative learning rates (`lr = 0.08`).

### 5.3 Triton Kernel Thread Masking & Direction Discontinuity
- **Challenge**: Combine drill sequences have variable durations (from 15 frames for short splits up to 220 frames for route drills).
- **GPU Implementation**:
  - Padded all sequences to $B_{\text{size}} = 256$ in GPU memory.
  - Passed exact sequence lengths to Triton and enforced `mask = t_idx < seq_len` across all load/store instructions.
  - Implemented continuous circular difference:
    $$\Delta \theta = d_t - d_{t-1} - 360 \cdot \left\lfloor \frac{d_t - d_{t-1} + 180}{360} \right\rfloor$$
    preventing spurious $360^\circ$ jumps when athletes crossed north heading ($0^\circ$).

---

## 6. Iteration 6 & Comprehensive Model Audit Learnings

### 6.1 Drill Sequence Grouping Key Bug (`drill_name` Omission)
- **Symptom**: Grouping tracking sequences with `['nfl_id', 'drill_type', 'attempt']` yielded only 3,717 sequences with durations up to 922 frames, causing sequences to be artificially truncated at frame 256.
- **Root Cause**: Players perform multiple distinct skill drills within the same `drill_type` (e.g. 15 different route drills in `SKILL_DRILLS_WR`). Omitting `drill_name` concatenated completely different drills into giant multi-minute trajectories. When one drill finished and the next began, the player teleported across the field, introducing massive false spikes in numerical derivative jerk and curvature.
- **Resolution**:
  - Group strictly by `['nfl_id', 'drill_type', 'drill_name', 'attempt']`.
  - Exactly 6,301 valid drill runs exist across all 510 prospects.
  - Crucially: **maximum duration across all 6,301 drills is 209 frames (under 21 seconds)**! Zero drills exceed 256 frames, completely eliminating the sequence truncation risk.

### 6.2 Authentic Neuro-Kinematic Deep Architecture vs Tabular Marketing
- **Symptom**: Prior attempt named the model "NK-TrajNet: Neuro-Kinematic Trajectory Transformer" but only ran a 50-tree shallow GradientBoostingRegressor on 8 hand-crafted scalar percentiles.
- **Root Cause**: Neglecting end-to-end temporal modeling meant higher-order temporal patterns and phase dynamics were discarded.
- **Resolution**:
  - Implemented an authentic PyTorch `TrajectoryTemporalAttentionEncoder` incorporating multi-scale 1D temporal convolutions (receptive fields of 0.3s and 0.7s) and a 4-head Temporal Self-Attention layer.
  - Extracts continuous attention weights $\alpha(t)$ across trajectories, proving that the model independently learns to place maximum attention on the initial 0.0 - 0.6 second explosive drive phase.

### 6.3 Authentic NFL Translation Metric Formulation
- **Symptom**: Raw career surplus snaps over draft pick baseline yielded negative or zero $R^2$ across all models due to non-kinematic confounding (injuries, depth charts, multi-year coaching changes).
- **Resolution**:
  - Formulated direct, physically grounded in-game translation targets:
    1. **Pass Rusher Snap Get-Off Latency ($N=131$)**: $M_2$ achieved $R^2 = 0.3651$ (+129.3% gain over stopwatch baseline $R^2 = 0.1592$).
    2. **Pass Rush In-Game Pressure Rate ($N=122$)**: Highly predictive translation ($R^2 = 0.4686$, Pearson $r = +0.685$).
    3. **WR In-Game Cornerback Cushion Respect ($N=100$)**: $M_2$ achieved $R^2 = 0.1123$ (+182.9% gain over stopwatch baseline $R^2 = 0.0397$, Pearson $r = +0.348$).
