# NFL2027: Kinematic Trajectory Embedding Framework 🏈

**Official Repository for NFL Big Data Bowl 2027**  
*Uncovering Non-Obvious Linkages Between Combine Sensor Tracking and Regular-Season NFL Game Performance*

---

## 🚀 Key Highlights & Novelty

1. **Differential Geometry of Athlete Trajectories**:
   Continuous Frenet-Serret frame decomposition on 10 Hz wearable sensor coordinates:
   $$\mathbf{a}(t) = a_t(t) \mathbf{T}(t) + a_n(t) \mathbf{N}(t)$$
   Extracting instantaneous jerk ($j = da_t/dt$), path curvature ($\kappa = |\omega| / s$), and centripetal cutting load ($a_n = \kappa \cdot s^2$).

2. **Custom Triton GPU Kinematic Kernel**:
   Massively parallel fused GPU kernel calculating multi-scale kinematic tensors across **6,301 distinct drill runs** in **125.2 ms** on NVIDIA Tesla T4 (>3.44M frames/sec, **33.6x speedup** over CPU).

3. **PyTorch `TrajectoryTemporalAttentionEncoder`**:
   Multi-scale 1D temporal convolutions (receptive fields of 0.3s and 0.7s) coupled with 4-head Temporal Self-Attention discovering key athletic phase transitions.

4. **Statistically Audited NFL In-Game Translation**:
   - **Pass Rusher Snap Get-Off ($N=131$)**: Out-of-fold $R^2 = 0.3651$, Pearson $r = 0.605$ (**+129.3% gain** over traditional stopwatch baseline).
   - **Pass Rush Pressure Rate ($N=122$)**: $R^2 = 0.4686$, Pearson $r = +0.685$.
   - **WR Cushion Respect ($N=100$)**: $R^2 = 0.1123$, Pearson $r = +0.348$ (**+182.9% gain** over traditional baseline).

---

## 📁 Repository Structure

```
NFL2027/
├── code.py                              # End-to-end training pipeline & Triton kernel
├── README.md                            # Project overview & quick start
├── .gitignore                           # Git ignore rules
└── knowledge/                           # Comprehensive research & analytics base
    ├── README.md                        # Knowledge base index
    ├── competition_overview.md          # Official rules, tracks, and submission guidelines
    ├── data_catalog.md                  # Complete dataset schemas and ER relationships
    ├── eda_findings.md                  # In-depth EDA findings and empirical correlations
    ├── mathematical_formulation.md      # Mathematical proofs, derivations, and loss functions
    ├── community_insights.md            # Forum intelligence, organizer clarifications & FAQ
    ├── failures_and_learnings.md        # Technical gotchas, bug post-mortems & data traps
    ├── progress_tracker.md              # Living experiment and milestone tracker
    └── figures/                         # High-resolution publication visualizations
        ├── model_benchmark_accuracy.png
        ├── triton_kinematic_acceleration.png
        ├── phase_portrait_frenet_serret.png
        ├── in_game_linkage_validation.png
        └── ...
```

---

## ⚡ Quick Start on Kaggle

```bash
# Clone directly into Kaggle /kaggle/working directory
git clone https://github.com/Ragu-123/NFL2027.git
cd NFL2027

# Execute complete pipeline
python code.py
```
