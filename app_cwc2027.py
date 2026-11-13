import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="ICC Cricket World Cup 2027 — Predictor & Simulator",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 ICC Men's Cricket World Cup 2027 — Schedule, Venue & Monte Carlo Predictor")
st.caption(
    "Hosts: South Africa, Zimbabwe & Namibia | Dates: Oct 2 – Nov 21, 2027 | 14 Teams, 57 Matches, 12 Venues"
)

# -----------------------------------------------------------------------------
# MEMBER 1: DATA LOADING & WIN PROBABILITY ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_cwc27_data():
    sched = pd.read_csv("cwc_2027_schedule_venues.csv")
    teams = pd.read_csv("cwc_2027_team_ratings.csv")
    return sched, teams


sched_df, teams_df = load_cwc27_data()
rating_map = dict(zip(teams_df["team"], teams_df["odi_rating"]))
host_map = dict(zip(teams_df["team"], teams_df["is_host"]))


def win_probability(team_a: str, team_b: str, venue_country: str = "South Africa", host_boost: float = 6.0) -> float:
    """Logistic Elo-style win probability from ICC ODI Ratings + Home Advantage."""
    r_a = rating_map.get(team_a, 60)
    r_b = rating_map.get(team_b, 60)
    if (team_a == "South Africa" and venue_country == "South Africa") or (
        team_a == "Zimbabwe" and venue_country == "Zimbabwe"
    ):
        r_a += host_boost
    if (team_b == "South Africa" and venue_country == "South Africa") or (
        team_b == "Zimbabwe" and venue_country == "Zimbabwe"
    ):
        r_b += host_boost
    diff = r_a - r_b
    return float(1.0 / (1.0 + np.exp(-0.065 * diff)))


# -----------------------------------------------------------------------------
# SIDEBAR: SIMULATION & LIVE MATCHUP CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Simulator & Matchup Controls")
n_sims = st.sidebar.slider("Monte Carlo Simulations", 200, 3000, 1000, step=200)
host_boost = st.sidebar.slider("Home-Ground Rating Boost", 0.0, 12.0, 6.0, step=1.0)

team_list = teams_df["team"].tolist()
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Fixture Predictor")
sim_a = st.sidebar.selectbox("Team A", team_list, index=0)
sim_b = st.sidebar.selectbox("Team B", team_list, index=4)
venue_list = sorted(sched_df["venue"].unique().tolist())
sim_venue = st.sidebar.selectbox("Venue", venue_list, index=venue_list.index("Newlands Cricket Ground"))

venue_country = sched_df[sched_df["venue"] == sim_venue]["host_country"].iloc[0]
p_a = win_probability(sim_a, sim_b, venue_country, host_boost)
if sim_a != sim_b:
    st.sidebar.metric(f"{sim_a} Win Prob", f"{p_a*100:.1f}%")
    st.sidebar.metric(f"{sim_b} Win Prob", f"{(1-p_a)*100:.1f}%")
else:
    st.sidebar.warning("Choose two different teams.")

# -----------------------------------------------------------------------------
# 4-TAB ANALYTICS SUITE
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📅 1. 2027 Schedule & 12 Venues",
        "⚡ 2. Group Stage Match Predictor",
        "🎲 3. Monte Carlo Cup Simulator",
        "🧬 4. Venue & Group Clustering",
    ]
)

# --- TAB 1: SCHEDULE & VENUES (MEMBER 2) ---
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tournament Window", "Oct 2 – Nov 21, 2027", "51 Days")
    k2.metric("Total Matches", "57 Matches", "Super Series + Group + Super 7 + KO")
    k3.metric("Host Venues", "12 Stadiums", "8 SA • 3 ZIM • 1 NAM")
    k4.metric("Final Venue", "Wanderers, JHB", "Nov 21, 2027 (34,000 cap)")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        country_filter = st.multiselect(
            "Filter by Host Country",
            ["South Africa", "Zimbabwe", "Namibia"],
            default=["South Africa", "Zimbabwe", "Namibia"],
        )
    with col_f2:
        stage_filter = st.multiselect(
            "Filter by Stage",
            sched_df["stage"].unique().tolist(),
            default=sched_df["stage"].unique().tolist(),
        )

    filtered = sched_df[
        sched_df["host_country"].isin(country_filter) & sched_df["stage"].isin(stage_filter)
    ]
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    venue_summary = (
        sched_df.groupby(["venue", "city", "host_country", "stadium_capacity", "pitch_profile"])
        .size()
        .reset_index(name="published_fixtures")
    )
    fig_v = px.bar(
        venue_summary.sort_values("stadium_capacity", ascending=True),
        x="stadium_capacity",
        y="venue",
        color="host_country",
        orientation="h",
        title="2027 World Cup Host Stadiums by Seating Capacity",
    )
    st.plotly_chart(fig_v, use_container_width=True)

# --- TAB 2: GROUP STAGE MATCH PREDICTOR (MEMBER 3) ---
with tab2:
    st.subheader("Pre-Tournament Win Probability Across All 30 Group Stage Fixtures")
    gs = sched_df[sched_df["stage"] == "Group Stage"].copy()
    gs["team_1_win_%"] = gs.apply(
        lambda r: round(win_probability(r["team_1"], r["team_2"], r["host_country"], host_boost) * 100, 1),
        axis=1,
    )
    gs["team_2_win_%"] = (100.0 - gs["team_1_win_%"]).round(1)
    gs["predicted_winner"] = np.where(gs["team_1_win_%"] >= 50, gs["team_1"], gs["team_2"])
    gs["upset_watch"] = np.where(
        np.minimum(gs["team_1_win_%"], gs["team_2_win_%"]) >= 35.0, "🔥 Close / Upset Risk", "✅ Clear Favorite"
    )

    st.dataframe(
        gs[
            [
                "date",
                "group_name",
                "team_1",
                "team_2",
                "venue",
                "team_1_win_%",
                "team_2_win_%",
                "predicted_winner",
                "upset_watch",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_a * 100,
            title={"text": f"Sidebar Matchup: {sim_a} vs {sim_b} at {sim_venue}"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 35], "color": "#fde2e4"},
                    {"range": [35, 65], "color": "#fff1cc"},
                    {"range": [65, 100], "color": "#d8f3dc"},
                ],
            },
        )
    )
    gauge.update_layout(height=300)
    st.plotly_chart(gauge, use_container_width=True)

# --- TAB 3: MONTE CARLO TOURNAMENT SIMULATOR (MEMBER 3 & 4) ---
with tab3:
    st.subheader(f"Monte Carlo Tournament Simulation ({n_sims:,} Runs: Group Stage → Super 7 → Knockouts)")

    @st.cache_data
    def run_monte_carlo(n_runs: int, h_boost: float):
        rng = np.random.default_rng(42)
        group_a = teams_df[teams_df["group_2027"] == "Group A"]["team"].tolist()
        group_b = teams_df[teams_df["group_2027"] == "Group B"]["team"].tolist()
        all_t = group_a + group_b
        super7_cnt = {t: 0 for t in all_t}
        semi_cnt = {t: 0 for t in all_t}
        champ_cnt = {t: 0 for t in all_t}

        gs_matches = sched_df[sched_df["stage"] == "Group Stage"]

        for _ in range(n_runs):
            pts = {t: 0.0 for t in all_t}
            # Group Stage
            for _, m in gs_matches.iterrows():
                t1, t2, hc = m["team_1"], m["team_2"], m["host_country"]
                p1 = win_probability(t1, t2, hc, h_boost)
                if rng.random() < p1:
                    pts[t1] += 2.0 + rng.uniform(0.0, 0.05)  # NRR tiebreaker jitter
                else:
                    pts[t2] += 2.0 + rng.uniform(0.0, 0.05)

            ga_sorted = sorted(group_a, key=lambda x: pts[x], reverse=True)
            gb_sorted = sorted(group_b, key=lambda x: pts[x], reverse=True)
            best_4th = ga_sorted[3] if pts[ga_sorted[3]] >= pts[gb_sorted[3]] else gb_sorted[3]
            s7_teams = ga_sorted[:3] + gb_sorted[:3] + [best_4th]
            for t in s7_teams:
                super7_cnt[t] += 1

            # Super 7 single round-robin (21 matches)
            s7_pts = {t: 0.0 for t in s7_teams}
            for i in range(len(s7_teams)):
                for j in range(i + 1, len(s7_teams)):
                    t1, t2 = s7_teams[i], s7_teams[j]
                    p1 = win_probability(t1, t2, "South Africa", h_boost)
                    if rng.random() < p1:
                        s7_pts[t1] += 2.0 + rng.uniform(0.0, 0.05)
                    else:
                        s7_pts[t2] += 2.0 + rng.uniform(0.0, 0.05)

            sf_teams = sorted(s7_teams, key=lambda x: s7_pts[x], reverse=True)[:4]
            for t in sf_teams:
                semi_cnt[t] += 1

            # Semi-finals: 1 vs 4, 2 vs 3
            sf1_win = sf_teams[0] if rng.random() < win_probability(sf_teams[0], sf_teams[3], "South Africa", h_boost) else sf_teams[3]
            sf2_win = sf_teams[1] if rng.random() < win_probability(sf_teams[1], sf_teams[2], "South Africa", h_boost) else sf_teams[2]
            # Final at Johannesburg
            champ = sf1_win if rng.random() < win_probability(sf1_win, sf2_win, "South Africa", h_boost) else sf2_win
            champ_cnt[champ] += 1

        res = pd.DataFrame(
            [
                {
                    "Team": t,
                    "Group": teams_df[teams_df["team"] == t]["group_2027"].iloc[0],
                    "ODI Rating": rating_map[t],
                    "Super 7 %": round(super7_cnt[t] / n_runs * 100, 1),
                    "Semi-Final %": round(semi_cnt[t] / n_runs * 100, 1),
                    "Title Win %": round(champ_cnt[t] / n_runs * 100, 1),
                }
                for t in all_t
            ]
        ).sort_values("Title Win %", ascending=False)
        return res

    sim_df = run_monte_carlo(n_sims, host_boost)
    fig_mc = px.bar(
        sim_df,
        x="Title Win %",
        y="Team",
        color="Group",
        orientation="h",
        text="Title Win %",
        title="2027 Cricket World Cup Championship Odds (Monte Carlo)",
    )
    fig_mc.update_layout(yaxis=dict(autorange="reversed"), height=440)
    st.plotly_chart(fig_mc, use_container_width=True)
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

# --- TAB 4: VENUE & TEAM CLUSTERING (MEMBER 4) ---
with tab4:
    st.subheader("K-Means Venue & Team Tier Segmentation")
    v_df = (
        sched_df.groupby(["venue", "city", "host_country", "stadium_capacity", "pitch_profile"])
        .size()
        .reset_index(name="fixtures")
    )
    scaler = StandardScaler()
    X_v = scaler.fit_transform(v_df[["stadium_capacity", "fixtures"]])
    km_v = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_v)
    v_df["Venue Tier"] = km_v.labels_.astype(str)

    fig_vc = px.scatter(
        v_df,
        x="stadium_capacity",
        y="fixtures",
        color="Venue Tier",
        symbol="host_country",
        hover_name="venue",
        text="city",
        size="stadium_capacity",
        title="12 Host Venues Clustered by Capacity & Published Fixtures",
    )
    fig_vc.update_traces(textposition="top center")
    st.plotly_chart(fig_vc, use_container_width=True)
    st.dataframe(v_df, use_container_width=True, hide_index=True)