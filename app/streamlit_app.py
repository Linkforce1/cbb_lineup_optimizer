"""
Illinois Basketball Lineup Synergy Optimizer
Streamlit App
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from itertools import combinations
import json
import math
import os

# PAGE CONFIG

st.set_page_config(
    page_title="Illinois Lineup Optimizer",
    page_icon="🔶",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CONSTANTS 
MIN_PCT_THRESHOLD = 15.0
FRONTCOURT_ROLES = {"Stretch 4", "PF/C", "C"}
GUARD_ROLES      = {"Pure PG", "Scoring PG", "Combo G"}
WING_ROLES       = {"Wing G", "Wing F"}
ALL_ROLES = ["Pure PG", "Scoring PG", "Combo G", "Wing G", "Wing F",
             "Stretch 4", "PF/C", "C"]
SYNERGY_PAIRS = [
    ("Pure PG",    "Stretch 4"),
    ("Pure PG",    "PF/C"),
    ("Scoring PG", "Stretch 4"),
    ("Combo G",    "Stretch 4"),
    ("Combo G",    "PF/C"),
    ("Wing F",     "Pure PG"),
]

# LOAD ARTIFACTS 
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "lineup_model.pkl")
    artifact = joblib.load(model_path)
    return artifact["pipeline"], artifact["feat_cols"]

@st.cache_data
def load_all_players():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "player_data_2026.csv")
    df = pd.read_csv(data_path)
    df = df.rename(columns={" bpm": "bpm", " obpm": "obpm", " dbpm": "dbpm"})
    df['TPAR'] = df['TPA'] / (df['TPA'] + df['twoPA'])
    return df

def get_roster(all_players: pd.DataFrame, team: str) -> pd.DataFrame:
    return all_players[
        (all_players["team"] == team) &
        (all_players["Min_per"] >= MIN_PCT_THRESHOLD)
    ].reset_index(drop=True)
    

@st.cache_data
def load_metadata():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(current_dir, "..", "models", "model_metadata.json")
    with open(metadata_path) as f:
        return json.load(f)

pipeline, feat_cols = load_model()
all_players         = load_all_players()
metadata            = load_metadata()
all_teams           = sorted(all_players["team"].unique().tolist())
    
def build_lineup_features(lineup_players: pd.DataFrame) -> dict:
    """
    Given a DataFrame of 5 players (subset of the Illinois roster),
    compute the same features used in training.

    lineup_players must have the same columns as the filtered player data.
    Minutes are re-weighted to sum to 100% within the lineup.
    """
    grp = lineup_players.copy()

    denom = grp['TPA'] + grp['twoPA']
    grp['TPAR'] = np.where(denom > 0, grp['TPA'] / denom, 0.0)

    if grp["Min_per"].sum() == 0:
        grp["Min_per"] = 1.0 
    w = grp["Min_per"] / grp["Min_per"].sum()

    wtd_obpm  = (grp["obpm"]    * w).sum()
    wtd_dbpm  = (grp["dbpm"]    * w).sum()
    wtd_ortg  = (grp["ORtg"]    * w).sum()
    wtd_efg   = (grp["eFG"]     * w).sum()
    wtd_ts    = (grp["TS_per"]  * w).sum()
    wtd_usg   = (grp["usg"]     * w).sum()
    wtd_ast   = (grp["AST_per"] * w).sum()
    wtd_tov   = (grp["TO_per"]  * w).sum()
    wtd_orb   = (grp["ORB_per"] * w).sum()
    wtd_drb   = (grp["DRB_per"] * w).sum()
    wtd_blk   = (grp["blk_per"] * w).sum()
    wtd_stl   = (grp["stl_per"] * w).sum()
    wtd_ast_tov = wtd_ast/(wtd_tov + 1e-9)

    fc = grp[grp["role"].isin(FRONTCOURT_ROLES)]
    guards = grp[grp["role"].isin(GUARD_ROLES)]
    fc_3p_rate = fc["TPAR"].mean() if len(fc) > 0 else 0.0
    guard_ast  = (guards["AST_per"] * (guards["Min_per"] / max(guards["Min_per"].sum(), 1))).sum() \
                 if len(guards) > 0 else 0.0
    fc_dbpm    = (fc["dbpm"] * (fc["Min_per"] / max(fc["Min_per"].sum(), 1))).sum() \
                 if len(fc) > 0 else 0.0

    usg_var    = grp["usg"].var()
    role_counts = grp["role"].value_counts()
    roles_present = set(grp["role"].unique())
    role_probs = grp["role"].value_counts(normalize=True)
    role_entropy = -(role_probs * np.log(role_probs + 1e-9)).sum()

    pair_feats = {}
    for r1, r2 in SYNERGY_PAIRS:
        key = f"pair_{r1.replace(' ','_')}_{r2.replace(' ','_')}"
        pair_feats[key] = int(r1 in roles_present and r2 in roles_present)

    return {
        "wtd_obpm": wtd_obpm, "wtd_dbpm": wtd_dbpm,
        "wtd_ortg": wtd_ortg, "wtd_efg": wtd_efg, "wtd_ts": wtd_ts,
        "wtd_usg": wtd_usg, "wtd_ast": wtd_ast, "wtd_tov": wtd_tov,
        "wtd_orb": wtd_orb, "wtd_drb": wtd_drb,
        "wtd_blk": wtd_blk, "wtd_stl": wtd_stl,
        "usg_var": usg_var, "fc_3p_rate": fc_3p_rate,
        "guard_ast": guard_ast, "fc_dbpm": fc_dbpm,
        "role_entropy": role_entropy,
        "n_pure_pg":  role_counts.get("Pure PG",   0),
        "n_stretch4": role_counts.get("Stretch 4", 0),
        "n_pfc":      role_counts.get("PF/C",      0),
        "n_c":        role_counts.get("C",         0),
        "wtd_ast_tov":wtd_ast_tov,
        **pair_feats,
    }


def predict_lineup(pipeline, feat_cols: list, lineup_players: pd.DataFrame):
    """
    Predict adjEM for a 5-man lineup.
    Returns (point_estimate, std_estimate) — std from Bayesian posterior.
    """
    feats = build_lineup_features(lineup_players)
    X = np.array([[feats.get(c, 0.0) for c in feat_cols]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) 
    scaler = pipeline.named_steps["scaler"]
    selector = pipeline.named_steps["feature_selection"]
    model  = pipeline.named_steps["model"]
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)
    mean, std = model.predict(X_selected, return_std=True)
    return float(mean[0]), float(std[0])


def rank_all_lineups(pipeline, feat_cols: list, roster: pd.DataFrame,
                     n: int = 5) -> pd.DataFrame:
    """
    Enumerate all C(roster_size, n) lineups and rank by predicted adjEM.
    Returns a DataFrame sorted by predicted adjEM descending.
    """
    results = []
    for combo in combinations(range(len(roster)), n):
        lineup = roster.iloc[list(combo)]
        mean, std = predict_lineup(pipeline, feat_cols, lineup)
        # 95% credible interval
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        results.append({
            "players": " | ".join(lineup["player_name"].values),
            "roles":   " | ".join(lineup["role"].values),
            "pred_adjEM":  round(mean, 2),
            "ci_lower":    round(lower, 2),
            "ci_upper":    round(upper, 2),
            "uncertainty": round(std, 2),
        })

    return pd.DataFrame(results).sort_values("pred_adjEM", ascending=False).reset_index(drop=True)

def make_label(players_str):
    names = players_str.split(" | ")
    short = []
    for n in names:
        parts = n.split()
        if len(parts) >= 2:
            last = parts[1] if len(parts) > 2 and parts[-1] in ("II","Jr","III","IV") \
                   else parts[-1]
            short.append(f"{parts[0]} {last}")
        else:
            short.append(n)
    return " / ".join(short)

# THEME 

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

:root {
    --illini-orange: #E84A27;
    --illini-blue:   #13294B;
    --illini-light:  #F8F5F0;
    --illini-gray:   #6B7280;
    --card-bg:       #FFFFFF;
    --border:        #E5E7EB;
}

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: var(--illini-light);
    color: var(--illini-blue);
}

/* Header */
.app-header {
    background: var(--illini-blue);
    color: white;
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 5px solid var(--illini-orange);
}
.app-header h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: 0.02em;
    margin: 0;
    color: white;
}
.app-header p {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.7);
    margin: 0.4rem 0 0;
    font-weight: 300;
}

/* Section headers */
.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--illini-blue);
    border-left: 4px solid var(--illini-orange);
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    border-top: 3px solid var(--illini-orange);
}
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--illini-gray);
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--illini-blue);
}
.metric-card .sub {
    font-size: 0.75rem;
    color: var(--illini-gray);
    margin-top: 0.2rem;
}

/* Player chip */
.player-chip {
    display: inline-block;
    background: var(--illini-blue);
    color: white;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 0.05em;
}
.role-badge {
    display: inline-block;
    background: var(--illini-orange);
    color: white;
    border-radius: 3px;
    padding: 0.1rem 0.45rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-left: 0.3rem;
}

/* Lineup rank card */
.lineup-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
}
.lineup-card.top {
    border-left: 4px solid var(--illini-orange);
}
.lineup-rank {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--illini-orange);
    line-height: 1;
}

/* Confidence band */
.ci-band {
    background: #EFF6FF;
    border-radius: 4px;
    padding: 0.3rem 0.6rem;
    font-size: 0.78rem;
    color: #1D4ED8;
    font-weight: 500;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--illini-blue) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-baseweb]),
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSelectbox label {
    color: rgba(255,255,255,0.8) !important;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Hide streamlit branding */
#MainMenu, footer, header {visibility: expanded;}
</style>
""", unsafe_allow_html=True)

# SIDEBAR

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    selected_team = st.selectbox(
    "Select Team",
    options=all_teams,
    index=all_teams.index("Illinois") if "Illinois" in all_teams else 0,
    )
    roster = get_roster(all_players, selected_team)
    n_combos = math.comb(len(roster), 5)

    lineup_filter = st.selectbox(
    "Rankings Table",
    options=["Top 10", "Top 20", "Show all"],
    index=0,
)

    show_ci = st.checkbox("Show 95% Credible Intervals", value=True)
    show_synergy = st.checkbox("Show Synergy Breakdown", value=True)

    st.markdown("---")
    st.markdown("### 📋 Roster")
    for _, row in roster.iterrows():
        st.markdown(
            f"**{row['player_name']}** · {row['role']}<br>"
            f"<span style='font-size:0.78rem;color:rgba(255,255,255,0.6)'>"
            f"MIN% {row['Min_per']:.0f} · BPM {row['bpm']:.1f}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:rgba(255,255,255,0.4)'>"
        f"Roster only contains players that played at least {MIN_PCT_THRESHOLD}% of the team's minutes.<br>"
        "Model trained on 2022–26 D1 team-seasons.<br>"
        "Predicts adjEM from minute-weighted player attributes + role synergy features.<br>"
        "Credible intervals from Bayesian posterior.</div>",
        unsafe_allow_html=True
    )


# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="app-header">
    <h1>🏀 {selected_team} Lineup Optimizer</h1>
    <p>Bayesian Synergy Model · 2025–26 Season · {len(roster)} rotation players · {n_combos} possible lineups</p>
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Lineup Rankings",
    "🔬 Lineup Simulator",
    "🧬 Synergy Explorer",
    "🏀 Portal Simulator",
])

# TAB 1 — LINEUP RANKINGS

with tab1:
    st.markdown('<div class="section-title">All Possible Lineups — Ranked by Projected adjEM</div>',
                unsafe_allow_html=True)

    with st.spinner(f"Ranking all {n_combos} lineups..."):
        rankings = rank_all_lineups(pipeline, feat_cols, roster)

    if lineup_filter == "Top 10":
        display_df = rankings.head(10)
    elif lineup_filter == "Top 20":
        display_df = rankings.head(20)
    else:
        display_df = rankings

    # Summary metrics
    best  = rankings.iloc[0]
    avg = rankings['pred_adjEM'].mean()
    cutoff = rankings['pred_adjEM'].quantile(0.9)
    top_10_avg = rankings[rankings['pred_adjEM']>cutoff]['pred_adjEM'].mean()
    #spread = best["pred_adjEM"] - worst["pred_adjEM"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Lineup adjEM</div>
            <div class="value">+{best['pred_adjEM']:.1f}</div>
            <div class="sub">Projected efficiency margin</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Average Lineup adjEM</div>
            <div class="value">+{avg:.1f}</div>
            <div class="sub">Projected efficiency margin</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Top 10% Lineup Average</div>
            <div class="value">{top_10_avg:.1f}</div>
            <div class="sub">Projected efficiency margin</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Total Lineups</div>
            <div class="value">{n_combos}</div>
            <div class="sub"> Calculated based on {len(roster)} rotation players</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption(
    "Projections reflect relative lineup quality — use rankings for comparison "
    "across lineups rather than as absolute adjEM predictions.")

    # Plotly bar chart — top 10
    top10 = rankings.head(10).copy()
    top10["short_label"] = top10["players"].apply(make_label)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top10["pred_adjEM"],
        y=top10["short_label"],
        orientation="h",
        marker_color="#E84A27",
        marker_line_width=0,
        error_x=dict(
            type="data",
            symmetric=False,
            array=top10["ci_upper"] - top10["pred_adjEM"],
            arrayminus=top10["pred_adjEM"] - top10["ci_lower"],
            color="#13294B",
            thickness=2,
            width=6,
        ) if show_ci else None,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Projected adjEM: %{x:.2f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis_title="Projected adjEM",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Barlow, sans-serif", color="#13294B"),
        xaxis=dict(gridcolor="#F0F0F0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.markdown('<div class="section-title">Lineup Rankings Table</div>', unsafe_allow_html=True)

    display_table = display_df.copy()
    display_table.index = display_table.index + 1
    display_table.index.name = "Rank"

    # Rename for display
    display_table = display_table.rename(columns={
        "players":    "Players",
        "roles":      "Roles",
        "pred_adjEM": "Proj. adjEM",
        "ci_lower":   "CI Lower",
        "ci_upper":   "CI Upper",
        "uncertainty":"Uncertainty (σ)",
    })

    st.dataframe(
        display_table[["Players", "Roles", "Proj. adjEM", "CI Lower", "CI Upper", "Uncertainty (σ)"]],
        use_container_width=True,
        height=400,
    )

    csv = display_table.to_csv()
    st.download_button(
        "⬇ Download Rankings CSV",
        data=csv,
        file_name="illinois_lineup_rankings.csv",
        mime="text/csv",
    )


# TAB 2 — WHAT-IF SIMULATOR
 
with tab2:
    st.markdown('<div class="section-title">Build Any 5-Man Lineup</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Select exactly 5 players to see the projected adjEM and credible interval. "
        "Test lineups that have never played together.",
    )

    player_options = roster["player_name"].tolist()
    selected_players = st.multiselect(
        "Select 5 players",
        options=player_options,
        default=player_options[:5],
        max_selections=5,
    )

    if len(selected_players) == 5:
        lineup_df = roster[roster["player_name"].isin(selected_players)]
        lineup_df['TP_per'] = lineup_df['TP_per']*100
        mean, std = predict_lineup(pipeline, feat_cols, lineup_df)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std

        st.markdown("<br>", unsafe_allow_html=True)

        # Big result display
        col_result, col_detail = st.columns([1, 2])

        with col_result:
            color = "#E84A27" if mean >= 30 else "#13294B" if mean >= 20 else "#6B7280"
            st.markdown(f"""
            <div class="metric-card" style="padding:2rem;">
                <div class="label">Projected adjEM</div>
                <div class="value" style="font-size:3.5rem;color:{color}">
                    +{mean:.1f}
                </div>
                <div class="sub">95% CI: [{lower:.1f}, {upper:.1f}]</div>
                <div class="sub" style="margin-top:0.5rem">
                    σ = {std:.2f} &nbsp;·&nbsp; Uncertainty: {"Low" if std < 1.5 else "Medium" if std < 2.5 else "High"}
                </div>
            </div>""", unsafe_allow_html=True)

            # Rank among all 56
            rank_pos = (rankings["pred_adjEM"] > mean).sum() + 1
            st.markdown(f"""
            <div class="metric-card" style="margin-top:1rem">
                <div class="label">Lineup Rank</div>
                <div class="value">#{rank_pos} <span style="font-size:1rem;color:#6B7280">of {n_combos}</span></div>
                <div class="sub">Among all possible 5-man combos</div>
            </div>""", unsafe_allow_html=True)

        with col_detail:
            # Player breakdown table
            st.markdown('<div class="section-title">Player Breakdown</div>',
                        unsafe_allow_html=True)
            detail = lineup_df[["player_name", "role", "Min_per", "obpm", "dbpm", "ORtg",'TP_per' ,"usg", "eFG"]].copy()
            detail = detail.rename(columns={'TP_per':'3P%'})
            detail.columns = ["Player", "Role", "MIN%", "OBPM", "DBPM", "ORtg", '3P%',"USG%", "eFG%"]
            detail = detail.round(2)
            st.dataframe(detail, use_container_width=True, hide_index=True)

            # Synergy warnings
            if show_synergy:
                st.markdown('<div class="section-title" style="margin-top:1rem">Synergy Notes</div>',
                            unsafe_allow_html=True)
                roles_in_lineup = set(lineup_df["role"].tolist())

                # Usage congestion warning
                high_usg = lineup_df[lineup_df["usg"] > 25]["player_name"].tolist()
                if len(high_usg) >= 2:
                    st.warning(
                        f"⚠️ **Usage congestion:** {', '.join(high_usg)} both have USG% > 25. "
                        "Ball-handler redundancy may reduce offensive efficiency."
                    )

                # Spacing check
                fc_players = lineup_df[lineup_df["role"].isin(FRONTCOURT_ROLES)]
                if len(fc_players) > 0:
                    avg_fc_3p = fc_players["TP_per"].mean()
                    if avg_fc_3p < 25:
                        st.warning(
                            f"⚠️ **Limited spacing:** Frontcourt avg 3PT% is {avg_fc_3p:.0f}%. "
                            "Consider adding a stretch big to open driving lanes."
                        )
                    else:
                        st.success(
                            f"✅ **Good spacing:** Frontcourt avg 3PT% is {avg_fc_3p:.0f}%."
                        )
                st.info(
                    "💡 See the **Synergy Explorer** tab for data-driven role pair analysis "
                    "for this roster."
                    )
        # Gauge chart for context
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Context: Where Does This Lineup Project?</div>',
                    unsafe_allow_html=True)

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=mean,
            delta={"reference": 0, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [-30, 40], "tickwidth": 1},
                "bar": {"color": "#E84A27"},
                "steps": [
                    {"range": [-30, 0],  "color": "#FEE2E2"},
                    {"range": [0,  15],  "color": "#FEF9C3"},
                    {"range": [15, 25],  "color": "#D1FAE5"},
                    {"range": [25, 40],  "color": "#BFDBFE"},
                ],
                "threshold": {
                    "line": {"color": "#13294B", "width": 3},
                    "thickness": 0.8,
                    "value": mean,
                },
            },
            title={"text": "Projected adjEM vs D1 Context<br>"
                           "<span style='font-size:0.75em;color:gray'>"
                           "Elite > 25 | Good 15–25 | Average 0–15 | Below avg < 0</span>"},
            number={"suffix": " adjEM", "valueformat": ".1f"},
        ))
        fig2.update_layout(
            height=300,
            margin=dict(l=30, r=30, t=60, b=10),
            paper_bgcolor="white",
            font=dict(family="Barlow, sans-serif"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif len(selected_players) > 0:
        st.info(f"Select {5 - len(selected_players)} more player(s) to see projection.")
    else:
        st.info("Select 5 players to get started.")


# TAB 3 — SYNERGY EXPLORER

with tab3:
    st.markdown('<div class="section-title">Role Synergy Heatmap</div>',
                unsafe_allow_html=True)
    st.markdown(
        "For each role pair, this shows how the projected adjEM changes when both roles "
        "are present in a lineup vs. when only one is. Positive = complementary. Negative = redundant."
    )

    # Compute synergy delta for each role pair by averaging over all lineups
    # that contain vs. don't contain each pair
    role_list = sorted(roster["role"].unique().tolist())
    n_roles = len(role_list)
    synergy_matrix = pd.DataFrame(
        np.nan, index=role_list, columns=role_list
    )

    for r1, r2 in combinations(role_list, 2):
        both, one_only = [], []
        for _, row in rankings.iterrows():
            roles_in = row["roles"].split(" | ")
            has_r1 = r1 in roles_in
            has_r2 = r2 in roles_in
            if has_r1 and has_r2:
                both.append(row["pred_adjEM"])
            elif has_r1 or has_r2:
                one_only.append(row["pred_adjEM"])
        if both and one_only:
            delta = np.mean(both) - np.mean(one_only)
            synergy_matrix.loc[r1, r2] = round(delta, 2)
            synergy_matrix.loc[r2, r1] = round(delta, 2)

    # Fill diagonal with 0
    for r in role_list:
        synergy_matrix.loc[r, r] = 0.0

    fig3 = go.Figure(data=go.Heatmap(
        z=synergy_matrix.values.astype(float),
        x=synergy_matrix.columns.tolist(),
        y=synergy_matrix.index.tolist(),
        colorscale=[
            [0.0,  "#DC2626"],
            [0.5,  "#F9FAFB"],
            [1.0,  "#13294B"],
        ],
        zmid=0,
        text=synergy_matrix.round(1).values.astype(str),
        texttemplate="%{text}",
        textfont={"size": 12, "family": "Barlow Condensed"},
        hoverongaps=False,
        hovertemplate="%{y} + %{x}<br>Synergy delta: %{z:.2f} adjEM<extra></extra>",
        colorbar=dict(title="adjEM delta", tickfont=dict(family="Barlow")),
    ))
    fig3.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        font=dict(family="Barlow, sans-serif", color="#13294B"),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "Note: synergy deltas are computed from projected adjEM across the possible lineups. "
        "Small sample sizes per role pair mean these estimates carry uncertainty — "
        "treat as directional, not definitive."
    )

    # Individual player contribution table
    st.markdown('<div class="section-title" style="margin-top:1.5rem">Player Impact Summary</div>',
                unsafe_allow_html=True)
    st.markdown("Average projected adjEM of lineups containing each player.")

    player_impact = []
    for _, player in roster.iterrows():
        name = player["player_name"]
        lineups_with = rankings[rankings["players"].str.contains(name, regex=False)]
        lineups_without = rankings[~rankings["players"].str.contains(name, regex=False)]
        avg_with    = lineups_with["pred_adjEM"].mean()
        avg_without = lineups_without["pred_adjEM"].mean()
        player_impact.append({
            "Player":         name,
            "Role":           player["role"],
            "Avg adjEM (in)": round(avg_with, 2),
            "Avg adjEM (out)":round(avg_without, 2),
            "Impact delta":   round(avg_with - avg_without, 2),
        })

    impact_df = pd.DataFrame(player_impact).sort_values("Impact delta", ascending=False)

    fig4 = go.Figure(go.Bar(
        x=impact_df["Player"],
        y=impact_df["Impact delta"],
        marker_color=[
            "#E84A27" if v > 0 else "#13294B"
            for v in impact_df["Impact delta"]
        ],
        hovertemplate="<b>%{x}</b><br>Impact delta: %{y:.2f} adjEM<extra></extra>",
    ))
    fig4.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="adjEM delta (in vs out)",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Barlow, sans-serif", color="#13294B"),
        xaxis=dict(tickangle=-25),
        yaxis=dict(gridcolor="#F0F0F0"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(impact_df, use_container_width=True, hide_index=True)

# TAB 4 — PORTAL SIMULATOR

with tab4:
    st.markdown('<div class="section-title">Transfer Portal Fit Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Assess how a transfer portal target fits your system. "
        "Select a portal player to see their profile, role fit, and "
        "how they'd project in hypothetical lineups with your current roster."
    )

    # ── Portal player search (unchanged) ──
    portal_pool = all_players[
        (all_players["team"] != selected_team) &
        (all_players["Min_per"] >= MIN_PCT_THRESHOLD)
    ].copy()

    search = st.text_input("Search portal player by name", 
                           placeholder="e.g. Cameron Boozer")
    if search:
        portal_pool = portal_pool[
            portal_pool["player_name"].str.contains(search, case=False, na=False)
        ]

    if len(portal_pool) == 0:
        st.info("No players found. Try a different search.")
    else:
        portal_options = (
            portal_pool["player_name"] + " — " +
            portal_pool["team"] + " (" +
            portal_pool["role"] + ", ORtg " +
            portal_pool["ORtg"].round(1).astype(str) + ", BPM " +
            portal_pool["bpm"].round(1).astype(str) + ")"
        ).tolist()

        selected_portal_label = st.selectbox("Select portal target", 
                                             options=portal_options)
        portal_idx = portal_options.index(selected_portal_label)
        portal_player = portal_pool.iloc[[portal_idx]].copy()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 1: Player Profile ──
        st.markdown('<div class="section-title">Player Profile</div>',
                    unsafe_allow_html=True)

        profile_cols = ["player_name", "team", "role", "Min_per", 
                        "bpm", "obpm", "dbpm", "ORtg","TP_per","usg", "eFG",
                        "blk_per", "DRB_per", "AST_per"]
        st.dataframe(
            portal_player[profile_cols].rename(columns={
                "player_name": "Player", "team": "Current Team",
                "role": "Role", "Min_per": "MIN%", "bpm": "BPM",
                "obpm": "OBPM", "dbpm": "DBPM", "ORtg": "ORtg",
                "usg": "USG%", "eFG": "eFG%", "TP_per":"3P%",
                "blk_per": "BLK%", "DRB_per": "DRB%", "AST_per": "AST%"
            }).round(2),
            use_container_width=True,
            hide_index=True,
        )

        # ── Section 2: Role Fit Assessment ──
        st.markdown('<div class="section-title">Role Fit Assessment</div>',
                    unsafe_allow_html=True)

        portal_role = portal_player["role"].values[0]
        current_roles = roster["role"].value_counts()
        role_already_on_roster = current_roles.get(portal_role, 0)

        col1, col2, col3 = st.columns(3)

        with col1:
            # How many of this role already on roster
            color = "#E84A27" if role_already_on_roster >= 2 else "#2A9D8F"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Same Role on Roster</div>
                <div class="value" style="color:{color}">
                    {role_already_on_roster}
                </div>
                <div class="sub">{portal_role} already rostered</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            # USG compatibility — would they crowd existing ball handlers?
            high_usg_roster = (roster["usg"] > 22).sum()
            portal_usg = portal_player["usg"].values[0]
            usg_flag = portal_usg > 22 and high_usg_roster >= 2
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Usage Compatibility</div>
                <div class="value" style="color:{'#E84A27' if usg_flag else '#2A9D8F'}">
                    {'⚠️ High' if usg_flag else '✅ Good'}
                </div>
                <div class="sub">Portal USG% {portal_usg:.0f} · 
                {high_usg_roster} high-usage players on roster</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            # Spacing value — does frontcourt need shooting?
            portal_tpar = portal_player["TPAR"].values[0] if "TPAR" in portal_player else \
                          portal_player["TPA"].values[0] / max(portal_player["TPA"].values[0] + portal_player["twoPA"].values[0], 1)
            spacing_value = portal_role in FRONTCOURT_ROLES and portal_tpar > 0.35
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Spacing Value</div>
                <div class="value" style="color:{'#2A9D8F' if spacing_value else '#6B7280'}">
                    {'✅ Yes' if spacing_value else '—'}
                </div>
                <div class="sub">Stretch big who shoots threes</div>
            </div>""", unsafe_allow_html=True)

        # ── Section 3: Coach's Lineup Builder ──
        st.markdown('<div class="section-title" style="margin-top:1.5rem">'
                    'Build a Lineup With This Player</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"Select 4 players from your current roster to pair with "
            f"**{portal_player['player_name'].values[0]}** and see the projected adjEM."
        )

        current_player_options = roster["player_name"].tolist()
        selected_four = st.multiselect(
            f"Select 4 roster players to pair with {portal_player['player_name'].values[0]}",
            options=current_player_options,
            max_selections=4,
        )

        if len(selected_four) == 4:
            four_df = roster[roster["player_name"].isin(selected_four)]
            hypothetical_lineup = pd.concat([four_df, portal_player], ignore_index=True)

            mean, std = predict_lineup(pipeline, feat_cols, hypothetical_lineup)
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            mean = max(mean, 0)
            lower = max(lower, 0)

            # Rank against existing lineups for context
            rank_pos = (rankings["pred_adjEM"] > mean).sum() + 1
            n_lineups = len(rankings)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card" style="padding:2rem">
                    <div class="label">Projected adjEM</div>
                    <div class="value" style="font-size:3rem">+{mean:.1f}</div>
                    <div class="sub">95% CI: [{lower:.1f}, {upper:.1f}]</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card" style="padding:2rem">
                    <div class="label">How It Compares</div>
                    <div class="value">#{rank_pos}</div>
                    <div class="sub">vs your top {n_lineups} current lineups</div>
                </div>""", unsafe_allow_html=True)

            # Synergy notes reused from What-If tab
            if show_synergy:
                roles_in = set(hypothetical_lineup["role"].tolist())
                high_usg = hypothetical_lineup[hypothetical_lineup["usg"] > 25]["player_name"].tolist()
                if len(high_usg) >= 2:
                    st.warning(f"⚠️ **Usage congestion:** {', '.join(high_usg)} both have USG% > 25.")
                active_pairs = [f"{r1} + {r2}" for r1, r2 in SYNERGY_PAIRS
                                if r1 in roles_in and r2 in roles_in]
                if active_pairs:
                    st.success(f"✅ **High-value pairings present:** {', '.join(active_pairs)}")

        elif len(selected_four) > 0:
            st.info(f"Select {4 - len(selected_four)} more player(s) to see projection.")
        else:
            st.info("Select 4 roster players to build a hypothetical lineup.")
