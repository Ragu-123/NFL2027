# NFL Big Data Bowl 2027: Competition Overview 🏆

## 1. Executive Summary

- **Competition Title**: NFL Big Data Bowl 2027
- **Host**: The National Football League (NFL) Football Data & Analytics Team
- **Competition ID**: `162982`
- **Competition Slug / Ref**: `nfl-big-data-bowl-2027`
- **Competition URL**: [https://www.kaggle.com/competitions/nfl-big-data-bowl-2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)
- **Competition Category**: Featured / Analytics (Report & Notebook Track, No Automated Leaderboard)
- **Total Prize Pool**: $100,000 USD
- **Launch Date**: October 6, 2026
- **Submission Deadline**: January 6, 2027 (23:59 UTC)
- **Maximum Team Size**: 5 participants

---

## 1.1 In Plain Terms: What is This Competition?

Every spring, hundreds of college football prospects gather at Lucas Oil Stadium in Indianapolis for the **NFL Scouting Combine**. They run the 40-yard dash, jump, and perform drills in front of NFL head coaches, scouts, and General Managers who invest millions of dollars based on those results.

However, a famous problem in football is the **"Workout Warrior"**: an athlete who runs an elite 4.3-second 40-yard dash in shorts and t-shirt, but struggles when playing real 11-on-11 football in pads against elite NFL opponents. Conversely, other prospects test with mediocre combine times but play at an elite speed on game day (**"Game Speed"**).

**The Big Data Bowl 2027 Challenge:**
For the first time in history, the NFL tracked 510 college draft prospects with **wearable electronic RFID sensors** during their Combine drills (measuring exact $(x, y)$ coordinates, speed, and acceleration 10 times a second), and tracked these same athletes throughout their subsequent 3 seasons of NFL games.

The challenge is to **connect the dots**: Use the sensor tracking from the Combine to discover non-obvious mathematical signals that actually forecast whether a college athlete will succeed in real NFL games.

---

## 1.2 How Do We Compete Without a Traditional Leaderboard?

Traditional Kaggle competitions are "Prediction" competitions: you predict a target column (like 0 or 1), upload a `submission.csv`, and an algorithm scores you against a mathematical metric (like Log Loss or RMSE) on a live public/private leaderboard.

**NFL Big Data Bowl is an Analytics & Research Competition:**
- **No Automated Leaderboard**: There is no live scoreboard or CSV submission. You do not compete against an automated score.
- **The Submission Asset**: You submit a **Public Kaggle Notebook** containing:
  1. An executive research paper / writeup (maximum 2,000 words).
  2. Up to 10 high-quality analytical figures / visualizations.
  3. Clean, reproducible Python code demonstrating your feature engineering, data pipelines, and modeling.
- **How It is Judged**: A panel of **official NFL judges** (NFL team analytics directors, data scientists, coaches, and front-office executives) reads and grades every submission after the January 6, 2027 deadline.
- **The Scoring Rubric**:
  1. **Football Relevance & Actionability (30%)**: Can an NFL GM, scout, or coordinator immediately use this metric in the draft war room?
  2. **Data Science & Methodological Rigor (30%)**: Is the statistical modeling, tracking math, and feature engineering rigorous and reproducible?
  3. **Visual Storytelling & Narrative (20%)**: Are your charts compelling, clean, and intuitive to understand?
  4. **Novelty & Innovation (20%)**: Did you find a non-obvious insight that nobody else noticed?
- **The Prize**:
  - Top submissions are chosen as **Finalists** ($10,000 per finalist).
  - Finalists are flown to Indianapolis to present their findings on stage at the **2027 NFL Scouting Combine** in front of all 32 NFL teams. Winning or placing as a finalist frequently leads directly to full-time analytics jobs in the NFL!

---

## 2. Objective & Mission

The official challenge prompt:
> **"Uncover non-obvious linkages between Combine sensor tracking and regular-season NFL game performance."**

For decades, NFL talent evaluators, general managers, and scouts have relied on traditional stopwatch times and drill measurements from the annual NFL Scouting Combine in Indianapolis (such as 40-yard dash, vertical jump, 20-yard shuttle, and bench press). However:
- Traditional drills are isolated, pre-planned, and lack contact or football context.
- High-performing athletes in drills ("workout warriors") do not always translate into productive NFL starters.
- With the advent of Next Gen Stats (NGS) wearable RFID sensors during the Combine, teams now capture sub-second kinematics ($x, y$ coordinates, instantaneous speed $s$, acceleration $a$, orientation $o$, direction $dir$) during both traditional tests and football-specific skill drills.

The mission of BDB 2027 is to bridge the gap: **Can sensor-derived biomechanical and kinematic signatures at the Combine accurately forecast or contextualize NFL game execution, role adaptability, durability, and on-field efficiency?**

---

## 3. Competition Tracks

1. **Open Track**:
   - Open to all eligible data scientists, sports researchers, analysts, and engineers globally.
   - Cash prizes awarded to top submissions.
   - Finalists invited to present findings at the 2027 NFL Scouting Combine in Indianapolis.

2. **Undergraduate Track**:
   - Dedicated track reserved exclusively for undergraduate students currently enrolled in an accredited degree program.
   - Separate judging pool and prize allocation.

---

## 4. Submission Format & Strict Limits

Analytics competitions on Kaggle follow a notebook and report format:
- **Writeup Document / Kaggle Notebook**:
  - Maximum **2,000 words** of narrative text.
  - Maximum **10 static or interactive figures / visualizations**.
  - Must include problem framing, methodology, actionable insights, and limitations.
- **Accompanying Code**:
  - Publicly viewable Kaggle Notebook containing clean, reproducible preprocessing, feature engineering, and modeling pipelines.
  - Appendix tables, raw code cells, and auxiliary computations outside the 2,000-word narrative limit.
- **Evaluation Criteria**:
  - **Football Relevance & Actionability (30%)**: Are the insights practical for an NFL coaching staff, front office, or scouting department?
  - **Methodological Rigor & Data Science (30%)**: Statistical soundness, feature engineering clarity, control of confounders, validation strategy.
  - **Clarity of Communication & Storytelling (20%)**: Visual appeal, concise writing, logical structure.
  - **Innovation & Novelty (20%)**: Uncovering genuinely new, non-obvious linkages beyond basic correlations.

---

## 5. Timeline & Milestones

| Event | Date |
| :--- | :--- |
| **Competition Launch** | October 6, 2026 |
| **New Entrant & Merger Deadline** | January 6, 2027 (23:59 UTC) |
| **Final Notebook Submission Deadline** | January 6, 2027 (23:59 UTC) |
| **Finalist Announcements** | Late January 2027 |
| **NFL Scouting Combine Presentation** | Late February / Early March 2027 (Indianapolis, IN) |
