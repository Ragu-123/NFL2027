# NFL Big Data Bowl 2027 - Knowledge Base 🏈

Welcome to the centralized **NFL Big Data Bowl 2027 Knowledge Base**. This repository documents the complete end-to-end research, exploratory data analysis (EDA), data catalogs, community discussions, experimental tracking, failures, and technical insights for the competition.

---

## 🗂️ Knowledge Base Architecture

The knowledge repository is organized into modular documents designed to preserve institutional memory and guide modeling strategy:

| Document | Purpose |
| :--- | :--- |
| [**`competition_overview.md`**](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/competition_overview.md) | Official problem statement, evaluation criteria, submission guidelines (2,000 words, 10 figures), tracks (Open & University), timeline, and prize pool. |
| [**`data_catalog.md`**](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/data_catalog.md) | Comprehensive catalog of all 9 dataset files (2.3+ GB, 22.6M+ tracking rows), schemas, relationships, foreign keys, and cohort definition. |
| [**`eda_findings.md`**](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/eda_findings.md) | In-depth Exploratory Data Analysis report detailing physical metrics, 62 combine drill types, sensor-to-field linkages, and speed differentials. |
| [**`mathematical_formulation.md`**](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/mathematical_formulation.md) | Rigorous mathematical foundations: differential geometry of trajectories, Frenet-Serret frames, jerk, curvature, mechanical power, and formal supervised loss functions. |
| [**`community_insights.md`**](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/community_insights.md) | Synthesized insights from Kaggle discussions, official Discord announcements, organizer clarifications, and domain rules. |
| [**`failures_and_learnings.md`**](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/failures_and_learnings.md) | Post-mortem of technical pitfalls, tracking sensor anomalies, schema collisions, and rejected analytical hypotheses. |
| [**`progress_tracker.md`**](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/progress_tracker.md) | Milestone tracker monitoring completed stages, in-progress investigations, and upcoming modeling roadmap. |
| [**`figures/`**](file:///c:/Users/SEC/Downloads/kaggle/NFL/knowledge/figures) | Generated high-resolution visualization charts and analytical plots supporting the research. |

---

## 🚀 Key Competition Takeaways at a Glance

1. **First-of-its-Kind Combine Tracking**: Unlike prior Big Data Bowls that focused strictly on in-game tracking, the 2027 edition introduces **wearable sensor tracking from the NFL Scouting Combine** (`combine_tracking.csv`).
2. **Cohort Scope**: The competition centers around a closed cohort of **510 draft prospects** across the 2023, 2024, and 2025 NFL Draft classes.
3. **Tracking Specificity**: Game tracking files (`game_tracking_2023.csv`, `2024`, `2025`) contain over **22.65 million frames**, tracking exclusively these cohort prospects rather than all 22 players on field.
4. **Core Research Challenge**: *"Uncover non-obvious linkages between Combine sensor tracking and regular-season NFL game performance."*
