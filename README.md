# College Basketball Lineup Optimizer

A Bayesian machine learning system for optimizing college basketball lineups. Trained on 2022–26 D1 team-season data, the model predicts **adjusted efficiency margin (adjEM)** for any 5-man lineup using minute-weighted player attributes and role-synergy features.

Originally built around Illinois basketball, the Streamlit app now supports any D1 team in the dataset.

**[Live App](https://cbblineupoptimizer.streamlit.app)**

---

## How It Works

**Training** ([lineup_model.ipynb](lineup_model.ipynb))

For each team-season, rotation players (≥15% of team minutes) are aggregated into a feature vector capturing:

- **Minute-weighted individual stats** — ORtg, BPM, eFG%, TS%, USG%, AST%, TOV%, ORB%, DRB%, BLK%, STL%
- **Synergy features** — usage variance, frontcourt spacing (3PT%), guard playmaking depth, role entropy, frontcourt DBPM
- **Role-pair indicators** — binary flags for high-value playmaker/spacer combinations (e.g. Pure PG + Stretch 4) inspired by Sloan Analytics research

A **Bayesian Ridge** regression (inside a `StandardScaler → LassoCV selector → BayesianRidge` pipeline) is fit on team adjEM. The Bayesian posterior yields both a point estimate and a credible interval for every prediction.

**Cross-validation results (5-fold):** R² = 0.645 ± 0.017 | MAE = 6.09 ± 0.57 adjEM points

**Inference**

At prediction time, the same feature engineering runs on any group of 5 players drawn from the 2025–26 roster data. Minute weights are re-normalized within the selected lineup. The model returns a predicted adjEM and posterior standard deviation, used to compute 95% credible intervals.

---

## App Features

The Streamlit app ([app/streamlit_app.py](app/streamlit_app.py)) has four tabs:

| Tab | Description |
|---|---|
| **Lineup Rankings** | Enumerates all C(n, 5) lineups for the selected team and ranks by projected adjEM. Includes a top-10 bar chart with optional credible intervals and a downloadable CSV. |
| **Lineup Simulator** | Pick any 5 players to instantly see their projected adjEM, 95% CI, rank among all possible combos, synergy warnings (usage congestion, spacing), and a gauge chart contextualizing the projection against D1 norms. |
| **Synergy Explorer** | Heatmap of adjEM delta for every role-pair combination. Shows which role pairings add or subtract value for the current roster. Also displays per-player impact delta (avg adjEM in lineups with vs. without each player). |
| **Portal Simulator** | Search any D1 player and build a hypothetical lineup with 4 current roster players + the portal target. Includes role fit assessment (role crowding, usage compatibility, spacing value) and projected adjEM. |

---

## Project Structure

```
CBB_Lineup_Optimization/
├── app/
│   └── streamlit_app.py       # Streamlit application
├── data/
│   ├── player_data_20XX.csv   # Per-player season stats (2022–2026)
│   └── team_data_20XX.csv     # Per-team season stats with adjEM
├── images/
│   ├── feature_importance.png
│   └── predicted_vs_actual.png
├── models/
│   ├── lineup_model.pkl       # Trained pipeline artifact
│   └── model_metadata.json    # CV results and selected features
├── notebooks/
│   └── lineup_model.ipynb     # EDA and model development notebook
├── writeup/
│   └── illinois_lineup_optimizer_writeup.docx
├── lineup_model.py            # Training script
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Retrain the model** (optional — a pretrained model is already in `models/`):

```bash
python lineup_model.ipynb
```

**Run the app:**

```bash
streamlit run app/streamlit_app.py
```

---

## Data

Player and team data sourced from [Barttorvik](https://barttorvik.com). Each player row includes BartTorvik role classifications (Pure PG, Scoring PG, Combo G, Wing G, Wing F, Stretch 4, PF/C, C) which drive the synergy feature engineering.

Roster filtering applies a 15% minimum minute share threshold to focus on genuine rotation contributors.

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Bayesian Ridge Regression |
| Feature selection | LassoCV (threshold = median) |
| Selected features | `wtd_ortg`, `wtd_ast_tov`, `wtd_orb`, `wtd_drb`, `wtd_blk`, `wtd_stl`, `usg_var`, `guard_ast`, `role_entropy`, `n_pure_pg`, `n_c` |
| Training data | 2022–2026 D1 team-seasons |
| Target | adjEM = adjOE − adjDE |
| CV R² | 0.645 ± 0.017 |
| CV MAE | 6.09 ± 0.57 adjEM points |

Projections reflect **relative lineup quality** — use rankings for comparison across lineups rather than as absolute adjEM predictions.
