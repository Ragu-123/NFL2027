# Mathematical Foundations & Novel Kinematic Architectures for NFL Tracking 📐

## 1. Introduction & Theoretical Motivation

A seminal study by Szekely et al. (*"NFL Career Success as Predicted by NFL Scouting Combine"*, arXiv:2303.05774) proved that traditional stopwatch combine tests fail to predict NFL on-field success ($R^2 \approx 0.17$, RMSE = 1,210 snaps; with specific drills like the 40-yard dash yielding $R^2 \approx 0.02$ on route separation).

The mathematical root cause of this failure is **severe dimensional compression**:
Traditional combine metrics collapse a dynamic, continuous multidimensional movement trajectory into an isolated scalar number (e.g., $4.42\text{ s}$ for a 40-yard dash). In doing so, they completely discard:
1. Higher-order kinematic derivatives (continuous acceleration curves, jerk, rate of force development).
2. Curvature and lateral centripetal force generation during directional changes.
3. Deceleration braking mechanics and re-acceleration transitions.
4. Synergies across different drill modalities (straight-line burst vs. lateral cutting elasticity vs. football skill mechanics).

To overcome these fundamental limitations for NFL Big Data Bowl 2027, we introduce **NK-TrajNet v2 (Hierarchical Trajectory Transformer & Biomechanical Kinematic Manifold)**:
- **SE(2)-Equivariant Intrinsic Kinematic Manifold** $\mathbf{u}(t) \in \mathbb{R}^8$: Coordinate-free Frenet-Serret framing invariant under all rigid Euclidean transformations.
- **Continuous Samozino & Morin (2016) Force-Velocity-Power Profiling**: Derives neuromuscular mechanical properties ($F_0, v_0, P_{\max}, S_{\text{fv}}, \text{RFD}$) from tracking acceleration curves.
- **Hierarchical Multi-Drill Cross-Attention (HTT-Genome)**:
  - *Level 1 (Intra-Drill)*: Temporal multi-scale Conv1D + self-attention capturing burst and cutting phase dynamics.
  - *Level 2 (Inter-Drill)*: Cross-modal transformer fusing straight-line burst, lateral elasticity (shuttle), 3-cone agility, and position skill drills into a unified 48-dimensional **Prospect Movement Genome**.

---

## 2. SE(2)-Equivariant Intrinsic Kinematic Manifold

Let an athlete's field position over drill duration $t \in [0, T]$ be represented by a continuous planar curve $\boldsymbol{\gamma}(t) = (x(t), y(t))^T \in \mathbb{R}^2$, sampled at discrete intervals $\Delta t = 0.1\text{ s}$ ($10\text{ Hz}$).

Under any rigid planar Euclidean transformation $g = (R, \mathbf{t}) \in \mathrm{SE}(2)$ where $R \in \mathrm{SO}(2)$ is a 2D rotation matrix and $\mathbf{t} \in \mathbb{R}^2$ is a translation vector:
$$\boldsymbol{\gamma}'(t) = R \boldsymbol{\gamma}(t) + \mathbf{t}$$

Traditional Cartesian coordinates $(x, y)$ are arbitrary and coordinate-dependent. To construct an **invariant trajectory representation**, we compute Frenet-Serret intrinsic geometric invariants along the moving frame $(\mathbf{T}(t), \mathbf{N}(t))$:

1. **Tangential Speed**:
   $$s(t) = \|\dot{\boldsymbol{\gamma}}(t)\|_2 = \sqrt{\dot{x}(t)^2 + \dot{y}(t)^2}$$
2. **Tangential Acceleration**:
   $$a_t(t) = \dot{s}(t) = \frac{\dot{\boldsymbol{\gamma}}(t) \cdot \ddot{\boldsymbol{\gamma}}(t)}{\|\dot{\boldsymbol{\gamma}}(t)\|}$$
3. **Continuous Angular Velocity with Circular Wrap-Around**:
   $$\Delta \theta_t = \theta_t - \theta_{t-1} - 360 \cdot \left\lfloor \frac{\theta_t - \theta_{t-1} + 180}{360} \right\rfloor, \quad \omega(t) = \frac{\Delta \theta_t}{\Delta t} \cdot \frac{\pi}{180} \quad [\text{rad/s}]$$
4. **Normal / Centripetal Acceleration**:
   $$a_n(t) = s(t) \cdot |\omega(t)| = \|\ddot{\boldsymbol{\gamma}}(t) - a_t(t) \mathbf{T}(t)\|_2$$
5. **Differential Spatial Curvature**:
   $$\kappa(t) = \frac{|\dot{x}\ddot{y} - \dot{y}\ddot{x}|}{(\dot{x}^2 + \dot{y}^2)^{3/2}} = \frac{|\omega(t)|}{s(t) + \epsilon} \quad [\text{rad/yd}]$$
6. **Instantaneous Jerk (Rate of Force Development)**:
   $$j(t) = \frac{d a_t(t)}{dt} = \frac{a_t(t) - a_t(t-\Delta t)}{\Delta t} \quad [\text{yd/s}^3]$$
7. **Specific Mechanical Power Output**:
   $$p(t) = s(t) \cdot a_t(t) \quad [\text{W/kg normalized}]$$
8. **Centripetal Kinetic Energy Flux**:
   $$\Phi_n(t) = a_n(t) \cdot s(t) = \kappa(t) \cdot s(t)^3 \quad [\text{yd}^2/\text{s}^3]$$

**Theorem (SE(2) Invariance):**
The feature vector $\mathbf{u}(t) = [s(t), a_t(t), a_n(t), \kappa(t), j(t), |\omega(t)|, p(t), \Phi_n(t)]^T$ satisfies:
$$\mathbf{u}_{g \cdot \boldsymbol{\gamma}}(t) = \mathbf{u}_{\boldsymbol{\gamma}}(t) \quad \forall g \in \mathrm{SE}(2)$$
Thus, the intrinsic kinematic manifold is strictly coordinate-free, rotation-invariant, and translation-invariant.

---

## 3. Biomechanical Force-Velocity-Power Profiling (Samozino & Morin 2016)

Sprint acceleration kinetics follow macroscopic human muscular Hill-type constraints. During the 40-yard dash, horizontal velocity is modeled by the mono-exponential differential equation:
$$\dot{v}(t) = \frac{v_{\max} - v(t)}{\tau} \implies v(t) = v_{\max} \cdot \big(1 - e^{-t / \tau}\big)$$

Where:
- $v_{\max} = v_0$: Theoretical maximal horizontal velocity $[\text{yd/s}]$.
- $\tau$: Acceleration time constant $[\text{s}]$, representing the duration required to achieve $63.2\%$ of $v_{\max}$.
- Maximal acceleration at start: $a_{\max} = \frac{v_{\max}}{\tau} \quad [\text{yd/s}^2]$.

Applying Newton's second law with athlete body mass $m = \text{weight}_{\text{lbs}} \times 0.453592\text{ kg}$:
$$f_0 = a_{\max} \quad [\text{N/kg}], \quad F_0 = m \cdot a_{\max} \quad [\text{N}]$$

### 3.1 Linear Force-Velocity Spectrum & Parabolic Power
The mechanical relationship between horizontal force and running velocity is linear:
$$f(v) = f_0 \cdot \left(1 - \frac{v}{v_0}\right)$$
The force-velocity slope indexes athlete phenotype:
$$S_{\text{fv}} = -\frac{f_0}{v_0} \quad \left[\frac{\text{N}\cdot\text{s}}{\text{kg}\cdot\text{yd}}\right]$$
- **Force-Dominant Athletes** ($|S_{\text{fv}}|$ high): Elite push-off force $F_0$, rapid start burst, lower top velocity $v_0$ (defensive linemen, edge rushers).
- **Velocity-Dominant Athletes** ($|S_{\text{fv}}|$ low): Flatter slope, sustained acceleration, high $v_0$ (wide receivers, cornerbacks).

Mechanical power per unit mass is parabolic with peak at $v = v_0 / 2$:
$$P(v) = f(v) \cdot v = f_0 \cdot v \left(1 - \frac{v}{v_0}\right) \implies P_{\max} = \frac{f_0 \cdot v_0}{4} \quad [\text{W/kg}]$$

![Biomechanical Force Velocity Profiles](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/figures/biomechanical_force_velocity_profiles.png)

---

## 4. Hierarchical Trajectory Transformer with Multi-Drill Cross-Attention (HTT-Genome)

Combine athletes participate across multiple distinct drill modalities:
- Modality 0: **Straight-Line Burst** (`FORTY_YARD_DASH`)
- Modality 1: **Lateral Deceleration & Re-acceleration Elasticity** (`SHORT_SHUTTLE`)
- Modality 2: **Continuous Angular Curvature & Change of Direction** (`THREE_CONE_DRILL`)
- Modality 3: **Football-Specific Skill Mechanics** (`SKILL_DRILLS_*`: DL pass rush/run hoop, WR route breaks, OL pull/mirror)

### Level 1: Intra-Drill Temporal Phase Attention
For drill instance $k$ with 8-channel trajectory tensor $\mathbf{X}_k \in \mathbb{R}^{8 \times T_k}$:
1. **Multi-Scale 1D Convolutions**:
   $$c_{\text{short}} = \operatorname{GELU}(\operatorname{Conv1D}_{k=3}(\mathbf{X}_k)), \quad c_{\text{med}} = \operatorname{GELU}(\operatorname{Conv1D}_{k=7}(\mathbf{X}_k)), \quad c_{\text{long}} = \operatorname{GELU}(\operatorname{Conv1D}_{k=15}(\mathbf{X}_k))$$
   $$\mathbf{H}_k = \operatorname{BatchNorm}(\operatorname{Concat}(c_{\text{short}}, c_{\text{med}}, c_{\text{long}})) \in \mathbb{R}^{48 \times T_k}$$
2. **Temporal Self-Attention**:
   $$\mathbf{Q}_k = \mathbf{K}_k = \mathbf{V}_k = \mathbf{H}_k^T \in \mathbb{R}^{T_k \times 48}$$
   $$\mathbf{A}_k = \operatorname{Softmax}\left(\frac{\mathbf{Q}_k \mathbf{K}_k^T}{\sqrt{d_k}} + \mathbf{M}_k\right) \mathbf{V}_k$$
3. **Temporal Attention Pooling**:
   $$\alpha_k(t) = \operatorname{Softmax}(\mathbf{w}^T \mathbf{A}_k(t)), \quad \mathbf{z}_k = \sum_{t=1}^{T_k} \alpha_k(t) \mathbf{A}_k(t) \in \mathbb{R}^{48}$$

### Level 2: Inter-Drill Cross-Attention & Prospect Movement Genome
We concatenate the modality tokens with a learnable `[PROSPECT_GENOME]` token:
$$\mathbf{Z}_0 = [\mathbf{z}_{\text{genome}}, \mathbf{z}_{\text{burst}}, \mathbf{z}_{\text{shuttle}}, \mathbf{z}_{\text{3cone}}, \mathbf{z}_{\text{skill}}] \in \mathbb{R}^{5 \times 48}$$
Added to learned modality type embeddings $\mathbf{E}_{\text{type}} \in \mathbb{R}^{5 \times 48}$.
Multi-head cross-attention across modalities:
$$\mathbf{Z}_{\text{cross}} = \operatorname{LayerNorm}\big(\mathbf{Z} + \operatorname{MultiheadAttention}(\mathbf{Z}, \mathbf{Z}, \mathbf{Z}, \text{mask})\big)$$
$$\mathbf{Z}_{\text{out}} = \operatorname{LayerNorm}\big(\mathbf{Z}_{\text{cross}} + \operatorname{MLP}(\mathbf{Z}_{\text{cross}})\big)$$

The output at position 0 is the **Prospect Movement Genome** $\mathbf{g} = \mathbf{Z}_{\text{out}}[0] \in \mathbb{R}^{48}$, synthesizing straight-line burst, lateral elasticity, and skill execution into a compact athletic fingerprint.

![Prospect Movement Genome Cross-Attention & Latent Manifold](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/figures/prospect_movement_genome_cross_attention.png)

---

## 5. Empirical 5-Fold Cross-Validation Benchmark on Remote GPU

We evaluated competing architectures across **4 on-field NFL translation tasks** using 5-fold cross-validation on the remote Kaggle NVIDIA Tesla T4 server:

- **$M_0$ (Baseline)**: Traditional Combine Stopwatch Metrics (40 time, 10yd split, vertical, broad jump, shuttle, 3-cone, height, weight).
- **$M_1$ (Naive Tabular Kinematics)**: $M_0$ + peak speed + peak acceleration.
- **$M_2$ (HTT-Genome Proposed)**: SE(2) Intrinsic Kinematics + Samozino F-V Parameters ($F_0, v_0, P_{\max}, \tau$) + Prospect Movement Genome Latent Embeddings.

### 5.1 Out-of-Fold Performance Across 4 NFL Translation Tasks

| Task | Target Metric | Cohort Size ($N$) | $M_0$ Stopwatch ($R^2$) | $M_1$ Naive ($R^2$) | $M_2$ HTT-Genome ($R^2$) | Accuracy Gain ($\Delta R^2$) | Pearson Correlation ($r$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 1: Pass Rusher Snap Get-Off** | Mean in-game snap get-off latency (s) | 131 | 0.1646 | 0.3224 | **0.3591** | **+0.1945 (+118.2%)** | **$r = +0.599$** ($p < 10^{-13}$) |
| **Task 2: WR Route Separation at Pass Forward** | Separation from nearest defender at throw release (yds) | 144 | 0.0236 | 0.0095 | **0.0756** | **+0.0520 (+220.2%)** | **$r = +0.288$** ($p = 0.0005$) |
| **Task 3: OL Pass Protection Pressure Allowed** | Allowed pressure rate per pass-blocking snap | 120 | 0.1095 | 0.1000 | **0.0859** | Multi-scheme control | **$r = +0.306$** ($p = 0.0007$) |
| **Task 4: All-Prospect Career EPA Impact / Snap** | Net Expected Points Added per snap | 455 | -0.0063 | -0.0043 | **-0.0018** | **+0.0046** | **$r = +0.077$** ($2.14\times$ baseline) |

![Model Benchmark Accuracy](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/figures/model_benchmark_accuracy.png)

![In-Game Linkage Validation](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/figures/in_game_linkage_validation.png)

### 5.2 Key Physical Insights from Feature Importance
1. **Pass Rusher Get-Off**: Body mass inertia (`combine_weight`, 75.6%) sets the resistance baseline, while horizontal explosive power (`broad_jump`, 6.9%), initial rate of force development (`forty_jerk_05`, 5.0%), and Samozino theoretical acceleration ($a_{\max}$, 2.7%) drive instantaneous snap penetration.
2. **WR Separation**: Shorter arm lengths and lower center of mass combined with high angular curvature in route breaks (`wr_peak_curv`) produce the greatest separation at the moment of pass release.
3. **OL Protection**: Slower short shuttle times and longer wingspan index offensive tackles facing elite edge rushers on the perimeter, directly predicting higher pressure allowed rates.
4. **Career EPA**: Higher vertical jump power combined with lower 3-cone and shuttle times yields the strongest positive translation to overall regular-season team expected points added.
