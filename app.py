import streamlit as st
import pandas as pd
from pathlib import Path

from projection_core import (
    load_roster,
    load_schedule,
    load_current_score,
    project_matchup_rest_of_week,
    compute_win_probability,
)

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "src" / "data"

MY_TEAM_PATH = DATA_DIR / "my_team.csv"
OPP_TEAM_PATH = DATA_DIR / "opp_team.csv"
SCHEDULE_PATH = DATA_DIR / "nba_schedule_next_7_days.csv"
CURRENT_SCORE_PATH = DATA_DIR / "current_score.csv"

DEFAULT_STREAM_FP_PER_ADD = 20.0
TOTAL_SEASON_ADDS_DEFAULT = 75
REGULAR_SEASON_WEEKS = 19
PLAYOFF_WEEKS = 3
DEFAULT_PLAYOFF_ADD_RESERVE = 15

MAX_ACTIVE_PER_DAY = 10  # must match projection_core logic

# -------------------------------------------------------------------
# Helper functions for streaming logic
# -------------------------------------------------------------------

def build_dnp_dataframe(dnp_entries):
    """
    Convert list of dicts [{player_id, date}, ...] into a normalized DataFrame.
    """
    if not dnp_entries:
        return pd.DataFrame(columns=["player_id", "date"])
    df = pd.DataFrame(dnp_entries)
    df["player_id"] = df["player_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates(subset=["player_id", "date"])
    return df[["player_id", "date"]]


def compute_daily_slots(
    roster: pd.DataFrame,
    schedule_future: pd.DataFrame,
    dnp_entries,
    max_active_per_day: int = MAX_ACTIVE_PER_DAY,
) -> pd.DataFrame:
    """
    For a given roster and remaining schedule, compute:
      - active_players per date (top N by avg_fp among players whose team plays and are not DNP)
      - slots_left = max_active_per_day - active_players (>=0)
    """
    if schedule_future.empty:
        return pd.DataFrame(columns=["date", "active_players", "slots_left"])

    dnp_df = build_dnp_dataframe(dnp_entries)

    schedule_future = schedule_future.copy()
    schedule_future["date"] = pd.to_datetime(schedule_future["date"]).dt.normalize()
    sched_map = (
        schedule_future.groupby("date")["team"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )

    dates = sorted(sched_map.keys())
    rows = []

    for d in dates:
        teams_today = sched_map.get(d, set())
        if not teams_today:
            rows.append({"date": d, "active_players": 0, "slots_left": 0})
            continue

        todays_players = roster[roster["team_abbr"].isin(teams_today)].copy()

        if not dnp_df.empty:
            dnp_players = set(
                dnp_df.loc[dnp_df["date"] == d, "player_id"]
                .astype(str)
                .tolist()
            )
            todays_players = todays_players[
                ~todays_players["player_id"].astype(str).isin(dnp_players)
            ]

        if todays_players.empty:
            rows.append({"date": d, "active_players": 0, "slots_left": 0})
            continue

        todays_players = todays_players.sort_values("avg_fp", ascending=False)
        active_players = min(max_active_per_day, len(todays_players))
        slots_left = max(0, max_active_per_day - active_players)

        rows.append(
            {
                "date": d,
                "active_players": active_players,
                "slots_left": slots_left,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def recommend_streaming_teams(
    roster: pd.DataFrame,
    schedule_future: pd.DataFrame,
    dnp_entries,
    horizon_days: int,
    fp_per_add: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For the next `horizon_days`, compute:
      - daily open slots for this roster
      - best NBA teams to stream from, based on usable games on days with open slots.

    Returns:
      - slots_horizon: DataFrame with date, date_str, active_players, slots_left
      - team_agg:     DataFrame with team, games_usable, stream_value_fp
    """
    if schedule_future.empty:
        empty_slots = pd.DataFrame(columns=["date", "date_str", "active_players", "slots_left"])
        empty_teams = pd.DataFrame(columns=["team", "games_usable", "stream_value_fp"])
        return empty_slots, empty_teams

    schedule_future = schedule_future.copy()
    schedule_future["date"] = pd.to_datetime(schedule_future["date"]).dt.normalize()

    all_dates = sorted(schedule_future["date"].unique())
    if not all_dates:
        empty_slots = pd.DataFrame(columns=["date", "date_str", "active_players", "slots_left"])
        empty_teams = pd.DataFrame(columns=["team", "games_usable", "stream_value_fp"])
        return empty_slots, empty_teams

    horizon_days = max(1, min(horizon_days, len(all_dates)))
    horizon_dates = all_dates[:horizon_days]

    slots_all = compute_daily_slots(roster, schedule_future, dnp_entries)
    slots_horizon = slots_all[slots_all["date"].isin(horizon_dates)].copy()
    slots_horizon["date_str"] = slots_horizon["date"].dt.strftime("%a %Y-%m-%d")

    sched_horizon = schedule_future[schedule_future["date"].isin(horizon_dates)].copy()
    merged = sched_horizon.merge(
        slots_horizon[["date", "slots_left"]],
        on="date",
        how="left",
    )
    merged["slots_left"] = merged["slots_left"].fillna(0)

    usable = merged[merged["slots_left"] > 0].copy()

    if usable.empty:
        team_agg = pd.DataFrame(columns=["team", "games_usable", "stream_value_fp"])
    else:
        team_agg = (
            usable.groupby("team", as_index=False)
            .agg(games_usable=("date", "count"))
        )
        team_agg["stream_value_fp"] = team_agg["games_usable"] * fp_per_add
        team_agg = team_agg.sort_values("stream_value_fp", ascending=False)

    team_agg = team_agg[team_agg["stream_value_fp"] > 0]
    return slots_horizon, team_agg


def compute_drop_candidates(
    roster: pd.DataFrame,
    schedule_future: pd.DataFrame,
    drop_threshold_fp: float = 28.0,
) -> pd.DataFrame:
    """
    Identify realistic drop candidates:
      - avg_fp < drop_threshold_fp
      - ranked by remaining FP (few games + low avg_fp).
    """
    if schedule_future.empty:
        return pd.DataFrame(columns=["name", "pos", "team_abbr", "avg_fp", "games_remaining_team", "rem_fp"])

    schedule_future = schedule_future.copy()
    schedule_future["date"] = pd.to_datetime(schedule_future["date"]).dt.normalize()

    games_by_team = (
        schedule_future.groupby("team", as_index=False)
        .agg(games_remaining_team=("date", "count"))
        .rename(columns={"team": "team_abbr"})
    )

    r = roster.merge(games_by_team, on="team_abbr", how="left")
    r["games_remaining_team"] = r["games_remaining_team"].fillna(0).astype(int)
    r["rem_fp"] = r["games_remaining_team"] * r["avg_fp"]

    cand = r[r["avg_fp"] < drop_threshold_fp].copy()
    if cand.empty:
        return pd.DataFrame(columns=["name", "pos", "team_abbr", "avg_fp", "games_remaining_team", "rem_fp"])

    cand = cand.sort_values(["rem_fp", "avg_fp"], ascending=[True, True])
    return cand[["name", "pos", "team_abbr", "avg_fp", "games_remaining_team", "rem_fp"]]


# -------------------------------------------------------------------
# Data loading (cached)
# -------------------------------------------------------------------

@st.cache_data
def load_data():
    """Load rosters, schedule, and current scores from CSVs."""
    my_team = load_roster(MY_TEAM_PATH)
    opp_team = load_roster(OPP_TEAM_PATH)
    schedule = load_schedule(SCHEDULE_PATH)
    my_score, opp_score = load_current_score(CURRENT_SCORE_PATH)
    return my_team, opp_team, schedule, my_score, opp_score


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------

st.set_page_config(page_title="FBBK Decision Support", layout="wide")

st.title("🏀 Fantasy Basketball Decision Support (Yahoo)")

st.write(
    """
- Loads rosters, NBA schedule (next 7 days), and current matchup scores  
- Projects **rest-of-week** fantasy points (using current score + remaining games)  
- Enforces **10 active players per day**  
- Lets you mark **per-game** and **whole-week** DNPs for injuries/rest  
- Computes a **win probability**  
- Lets you run **add/drop streaming what-if** with season add budgeting  
"""
)

# Load core data
with st.spinner("Loading data..."):
    my_team, opp_team, schedule, my_score, opp_score = load_data()

schedule = schedule.copy()
schedule["date"] = pd.to_datetime(schedule["date"]).dt.normalize()

today = pd.Timestamp.today().normalize()
schedule_future = schedule[schedule["date"] >= today].copy()
schedule_dates = sorted(schedule_future["date"].dt.date.unique())

# -------------------------------------------------------------------
# Sidebar: streaming/adds settings
# -------------------------------------------------------------------

st.sidebar.header("Streaming & Adds Settings")

stream_fp_per_add = st.sidebar.number_input(
    "Assumed FP per streaming add",
    min_value=5.0,
    max_value=60.0,
    value=DEFAULT_STREAM_FP_PER_ADD,
    step=1.0,
)

st.sidebar.header("Season Adds Tracker")

total_season_adds = st.sidebar.number_input(
    "Total adds allowed per season",
    min_value=1,
    max_value=200,
    value=TOTAL_SEASON_ADDS_DEFAULT,
    step=1,
)

current_week = st.sidebar.number_input(
    "Current fantasy week (1–22)",
    min_value=1,
    max_value=REGULAR_SEASON_WEEKS + PLAYOFF_WEEKS,
    value=5,
    step=1,
)

adds_used_so_far = st.sidebar.number_input(
    "Adds already used this season (before this week)",
    min_value=0,
    max_value=200,
    value=0,
    step=1,
)

adds_used_this_week = st.sidebar.number_input(
    "Adds already used this week",
    min_value=0,
    max_value=30,
    value=0,
    step=1,
)

adds_reserved_for_playoffs = st.sidebar.number_input(
    "Adds to reserve for playoffs",
    min_value=0,
    max_value=50,
    value=DEFAULT_PLAYOFF_ADD_RESERVE,
    step=1,
)

# Baseline season-adds metrics
adds_total_used_base = adds_used_so_far + adds_used_this_week
adds_remaining_total_base = max(0, total_season_adds - adds_total_used_base)

if current_week <= REGULAR_SEASON_WEEKS:
    regular_weeks_left_including_this = REGULAR_SEASON_WEEKS - current_week + 1
else:
    regular_weeks_left_including_this = 0

adds_available_for_regular_base = max(
    0, adds_remaining_total_base - adds_reserved_for_playoffs
)

if regular_weeks_left_including_this > 0:
    avg_adds_per_week_base = (
        adds_available_for_regular_base / regular_weeks_left_including_this
    )
else:
    avg_adds_per_week_base = 0.0

# -------------------------------------------------------------------
# Current score
# -------------------------------------------------------------------

st.subheader("Current Matchup Score")
c1, c2 = st.columns(2)
with c1:
    st.metric("My Current Score", f"{my_score:.1f}")
with c2:
    st.metric("Opponent Current Score", f"{opp_score:.1f}")

st.write("---")

# -------------------------------------------------------------------
# DNP overrides
# -------------------------------------------------------------------

st.header("Injury / DNP Overrides")

# My team
st.subheader("My Team DNPs")

my_player_names = my_team["name"].tolist()

st.markdown("**Players OUT for the rest of this week**")
my_week_dnp_selected = st.multiselect(
    "Select my players who will NOT play again this week:",
    my_player_names,
    key="my_week_dnp",
)

my_week_dnp_entries = []
for name in my_week_dnp_selected:
    row = my_team.loc[my_team["name"] == name].iloc[0]
    pid = row["player_id"]
    team = row["team_abbr"]
    player_dates = schedule_future.loc[
        schedule_future["team"].eq(team), "date"
    ].dt.date.unique()
    for d in player_dates:
        my_week_dnp_entries.append({"player_id": str(pid), "date": str(d)})

st.markdown("**Specific game DNPs (My Team)**")
num_my_dnp = st.number_input(
    "Number of additional one-game DNP overrides for MY team",
    min_value=0,
    max_value=30,
    value=0,
    step=1,
    key="num_my_dnp",
)

my_dnp_entries = []
for i in range(num_my_dnp):
    st.markdown(f"**My one-game DNP #{i+1}**")
    col_p, col_d = st.columns(2)
    with col_p:
        name = st.selectbox(
            f"Player #{i+1}",
            my_player_names,
            key=f"my_dnp_player_{i}",
        )
    with col_d:
        d = st.selectbox(
            f"Date #{i+1}",
            schedule_dates,
            key=f"my_dnp_date_{i}",
        )

    pid = my_team.loc[my_team["name"] == name, "player_id"].iloc[0]
    my_dnp_entries.append({"player_id": str(pid), "date": str(d)})

st.write("---")

# Opponent
st.subheader("Opponent DNPs")

opp_player_names = opp_team["name"].tolist()

st.markdown("**Opponent players OUT for the rest of this week**")
opp_week_dnp_selected = st.multiselect(
    "Select opponent players who will NOT play again this week:",
    opp_player_names,
    key="opp_week_dnp",
)

opp_week_dnp_entries = []
for name in opp_week_dnp_selected:
    row = opp_team.loc[opp_team["name"] == name].iloc[0]
    pid = row["player_id"]
    team = row["team_abbr"]
    player_dates = schedule_future.loc[
        schedule_future["team"].eq(team), "date"
    ].dt.date.unique()
    for d in player_dates:
        opp_week_dnp_entries.append({"player_id": str(pid), "date": str(d)})

st.markdown("**Specific game DNPs (Opponent)**")
num_opp_dnp = st.number_input(
    "Number of additional one-game DNP overrides for OPPONENT",
    min_value=0,
    max_value=30,
    value=0,
    step=1,
    key="num_opp_dnp",
)

opp_dnp_entries = []
for i in range(num_opp_dnp):
    st.markdown(f"**Opponent one-game DNP #{i+1}**")
    col_p, col_d = st.columns(2)
    with col_p:
        name = st.selectbox(
            f"Opponent Player #{i+1}",
            opp_player_names,
            key=f"opp_dnp_player_{i}",
        )
    with col_d:
        d = st.selectbox(
            f"Date #{i+1}",
            schedule_dates,
            key=f"opp_dnp_date_{i}",
        )

    pid = opp_team.loc[opp_team["name"] == name, "player_id"].iloc[0]
    opp_dnp_entries.append({"player_id": str(pid), "date": str(d)})

st.write("---")

# -------------------------------------------------------------------
# Projection controls (always computed – no Run button)
# -------------------------------------------------------------------

st.header("Projection & Win Probability")

k = st.slider(
    "Win probability sensitivity (k for logistic model)",
    min_value=0.001,
    max_value=0.05,
    value=0.01,
    step=0.001,
)

# Compute projection every rerun, based on current inputs
all_my_dnp = my_dnp_entries + my_week_dnp_entries
all_opp_dnp = opp_dnp_entries + opp_week_dnp_entries

with st.spinner("Computing baseline projection..."):
    matchup = project_matchup_rest_of_week(
        my_roster=my_team,
        opp_roster=opp_team,
        schedule=schedule,
        my_current_score=my_score,
        opp_current_score=opp_score,
        my_dnp_overrides=all_my_dnp if all_my_dnp else None,
        opp_dnp_overrides=all_opp_dnp if all_opp_dnp else None,
        winprob_k=k,
    )

st.subheader("Baseline Summary (no streaming adds)")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("My Final Score", f"{matchup.my.final_total:.1f}")
with col_b:
    st.metric("Opponent Final Score", f"{matchup.opp.final_total:.1f}")
with col_c:
    st.metric("Win Probability", f"{100 * matchup.win_prob:.1f}%")

st.write("#### Baseline season-adds picture (before new what-if adds)")
c_sa1, c_sa2, c_sa3 = st.columns(3)
with c_sa1:
    st.metric(
        "Season adds remaining (baseline)",
        f"{adds_remaining_total_base}",
    )
with c_sa2:
    st.metric(
        "Adds reserved for playoffs (target)",
        f"{adds_reserved_for_playoffs}",
    )
with c_sa3:
    st.metric(
        "Avg adds/week for rest of regular season",
        f"{avg_adds_per_week_base:.2f}",
    )

st.write("---")

st.subheader("Daily Projection – My Team")
st.dataframe(matchup.my.daily)

st.subheader("Daily Projection – Opponent")
st.dataframe(matchup.opp.daily)

st.write("---")

# -------------------------------------------------------------------
# Add/Drop (Streaming) What-If
# -------------------------------------------------------------------

st.header("Add/Drop Streaming What-If")

st.write(
    f"""
Each streaming add is approximated as **+{stream_fp_per_add:.0f} FP**  
(you'll use the streaming team/schedule logic below to pick the best days/teams).
"""
)

c_my, c_opp = st.columns(2)
with c_my:
    my_adds = st.slider(
        "My additional adds this week (what-if)",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
    )
with c_opp:
    opp_adds = st.slider(
        "Opponent additional adds this week (what-if)",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
    )

my_final_stream = matchup.my.final_total + my_adds * stream_fp_per_add
opp_final_stream = matchup.opp.final_total + opp_adds * stream_fp_per_add

win_prob_stream = compute_win_probability(
    my_final_stream,
    opp_final_stream,
    k=k,
)

st.subheader("With Streaming Adds Applied")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "My Final (with adds)",
        f"{my_final_stream:.1f}",
        f"{my_final_stream - matchup.my.final_total:+.1f}",
    )
with col2:
    st.metric(
        "Opp Final (with adds)",
        f"{opp_final_stream:.1f}",
        f"{opp_final_stream - matchup.opp.final_total:+.1f}",
    )
with col3:
    st.metric(
        "Win Prob (with adds)",
        f"{100 * win_prob_stream:.1f}%",
        f"{100 * (win_prob_stream - matchup.win_prob):+.1f} pp",
    )

# Season-adds impact of my_adds
adds_total_used_new = adds_total_used_base + my_adds
adds_remaining_total_new = max(0, total_season_adds - adds_total_used_new)
adds_available_for_regular_new = max(
    0, adds_remaining_total_new - adds_reserved_for_playoffs
)

if regular_weeks_left_including_this > 0:
    avg_adds_per_week_new = (
        adds_available_for_regular_new / regular_weeks_left_including_this
    )
else:
    avg_adds_per_week_new = 0.0

st.subheader("Season Adds Impact (My Team Only)")

c_sb1, c_sb2, c_sb3 = st.columns(3)
with c_sb1:
    st.metric(
        "Season adds remaining (after what-if)",
        f"{adds_remaining_total_new}",
        f"{adds_remaining_total_new - adds_remaining_total_base:+}",
    )
with c_sb2:
    st.metric(
        "Adds available for regular season (after reserve)",
        f"{adds_available_for_regular_new}",
        f"{adds_available_for_regular_new - adds_available_for_regular_base:+}",
    )
with c_sb3:
    st.metric(
        "Avg adds/week for rest of regular season",
        f"{avg_adds_per_week_new:.2f}",
        f"{avg_adds_per_week_new - avg_adds_per_week_base:+.2f}",
    )

if my_adds > 0 and adds_available_for_regular_new < 0:
    st.warning(
        "This what-if uses more adds than your target once you reserve "
        "playoff adds. You’d be overspending your regular-season adds."
    )

# -------------------------------------------------------------------
# Streaming plan details (only if I'm using adds)
# -------------------------------------------------------------------

if my_adds > 0:
    st.write("---")
    st.subheader("Suggested Streaming Plan (My Team)")

    if schedule_dates:
        horizon_days = min(3, len(schedule_dates))
        slots_horizon, team_agg = recommend_streaming_teams(
            roster=my_team,
            schedule_future=schedule_future,
            dnp_entries=all_my_dnp,
            horizon_days=horizon_days,
            fp_per_add=stream_fp_per_add,
        )

        st.markdown(
            f"**Best days to stream over the next {horizon_days} day(s)** "
            "(based on open lineup slots):"
        )
        if slots_horizon.empty:
            st.write("No days with open slots in the next few days.")
        else:
            st.dataframe(
                slots_horizon[["date_str", "active_players", "slots_left"]]
                .sort_values("slots_left", ascending=False)
                .reset_index(drop=True)
            )

        st.markdown(
            f"**NBA teams to target for streaming (next {horizon_days} days)**"
        )
        if team_agg.empty:
            st.write(
                "No teams have usable games on days where you have open slots "
                "over the next few days."
            )
        else:
            st.dataframe(team_agg.reset_index(drop=True))

        st.markdown("**Drop candidates (avg FP < 28, low remaining value)**")
        drop_cands = compute_drop_candidates(
            my_team, schedule_future, drop_threshold_fp=28.0
        )
        if drop_cands.empty:
            st.write("No clear drop candidates under 28 FP with low remaining value.")
        else:
            st.dataframe(drop_cands.head(10).reset_index(drop=True))
    else:
        st.info("No remaining games in the schedule window – nothing to stream.")

st.write("---")
st.write("Made with ❤️ using Streamlit")
