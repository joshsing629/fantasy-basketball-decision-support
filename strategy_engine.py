"""
strategy_engine.py

Streaming / add-drop scenario analysis on top of projection_core.py.

Refinements:
- Baseline rest-of-week projection uses:
    * current_score.csv for points already earned
    * nba_schedule_next_7_days.csv for remaining games
    * 10-player-per-day cap via projection_core-style logic
- Streaming model:
    * Each add is assumed to be worth ADD_FP net points for the *rest of the week*
    * BUT only up to the team's remaining lineup capacity (days with <10 actives)
- Opponent is treated symmetrically:
    * Opp add value also capped by their streaming capacity
- Drop candidates:
    * Players under AVG_FP_DROP_THRESHOLD
    * With the fewest remaining games

CSV files expected in src/data/:
    my_team.csv
    opp_team.csv
    free_agents.csv   (same schema as rosters: player_id, name, pos, avg_fp, team_abbr)
    nba_schedule_next_7_days.csv
    current_score.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

import projection_core as core

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "src" / "data"

MY_TEAM_PATH = DATA_DIR / "my_team.csv"
OPP_TEAM_PATH = DATA_DIR / "opp_team.csv"
SCHEDULE_PATH = DATA_DIR / "nba_schedule_next_7_days.csv"
CURRENT_SCORE_PATH = DATA_DIR / "current_score.csv"

MAX_ADDS = 5          # 0–5 adds for scenario grid
ADD_FP = 20.0         # net FP per add for rest of week
STREAM_HORIZON_DAYS = 3  # for team streaming ranking
AVG_FP_DROP_THRESHOLD = 28.0  # drop candidates: avg_fp < 28.0

TOTAL_ADDS_CAP = 75            # Yahoo season add cap
REG_SEASON_WEEKS = 19          # regular season weeks
PLAYOFF_WEEKS = 3              # weeks 20, 21, 22
PLAYOFF_ADDS_RESERVED = 15     # how many adds you want to save for playoffs


# Update these two manually each week:
OPP_TEAM = 3                   # opponent's fantasy team ID in current_score.csv
CURRENT_WEEK = 5               # this week = 5 (through 2025-11-23)
ADDS_USED_TO_DATE = 22          # <-- you will update this as the season goes

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_roster(path: Path) -> pd.DataFrame:
    """
    Roster schema:
        player_id, name, pos, avg_fp, team_abbr
    """
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype(str)
    df["team_abbr"] = df["team_abbr"].astype(str).str.upper()
    df["avg_fp"] = pd.to_numeric(df["avg_fp"], errors="coerce").fillna(0.0)
    return df[["player_id", "name", "pos", "avg_fp", "team_abbr"]]



def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    """
    Load:
      - my roster
      - opponent roster
      - NBA schedule for next 7 days
      - current scores

    Works with either:
      - core.load_schedule_csv(...) / core.load_current_score_csv(...)
      - OR core.load_schedule(...) / core.load_current_score(...)
    """
    my_roster = load_roster(MY_TEAM_PATH)
    opp_roster = load_roster(OPP_TEAM_PATH)

    # ---- Schedule loader: handle both possible core APIs ----
    if hasattr(core, "load_schedule_csv"):
        schedule = core.load_schedule_csv(SCHEDULE_PATH)
    elif hasattr(core, "load_schedule"):
        schedule = core.load_schedule(SCHEDULE_PATH)
    else:
        raise AttributeError("projection_core has neither 'load_schedule_csv' nor 'load_schedule'")

    # IMPORTANT: do NOT rename 'team' here. projection_core.attach_schedule_to_roster
    # expects the schedule to have a 'team' column.
    # We will create 'team_abbr' only in the streaming helpers when needed.

    # ---- Current score loader: read CSV directly so we control team IDs ----
    score_df = pd.read_csv(CURRENT_SCORE_PATH)
    score_df["team_id"] = pd.to_numeric(score_df["team_id"], errors="coerce")
    score_df["points"] = pd.to_numeric(score_df["points"], errors="coerce").fillna(0.0)

    def _get_score(team_id: int) -> float:
        sub = score_df.loc[score_df["team_id"] == team_id, "points"]
        if sub.empty:
            raise ValueError(f"Team_id {team_id} not found in current_score.csv")
        return float(sub.iloc[0])

    # Hard-code your team as 10, opponent as OPP_TEAM
    my_score = _get_score(10)
    opp_score = _get_score(OPP_TEAM)

    return my_roster, opp_roster, schedule, my_score, opp_score




# ---------------------------------------------------------------------
# Baseline projection dataclass
# ---------------------------------------------------------------------

@dataclass
class BaselineProjection:
    dates: List[pd.Timestamp]
    my_daily: pd.Series
    opp_daily: pd.Series
    my_rest: float
    opp_rest: float
    my_final: float
    opp_final: float
    win_prob: float
    my_slots_capacity: int
    opp_slots_capacity: int
    my_active_table: pd.DataFrame
    opp_active_table: pd.DataFrame
    summary: pd.DataFrame

# ---------------------------------------------------------------------
# Helpers: daily active/slots & drop candidates
# ---------------------------------------------------------------------

def compute_daily_active_and_slots(
    roster: pd.DataFrame,
    schedule_future: pd.DataFrame,
    max_active_per_day: int = core.MAX_ACTIVE_PER_DAY,
) -> pd.DataFrame:
    """
    For a given roster and future schedule, compute per-date:
        - active_players (top N by avg_fp among players whose team plays)
        - slots_left = max(0, max_active_per_day - active_players)

    Returns DataFrame:
        date, active_players, slots_left
    """
    # Attach schedule: one row per (player, date) where team plays
    rs = core.attach_schedule_to_roster(roster, schedule_future)
    if rs.empty:
        # No remaining games
        return pd.DataFrame(columns=["date", "active_players", "slots_left"])

    rs["date"] = pd.to_datetime(rs["date"]).dt.normalize()

    rows = []
    for d, sub in rs.groupby("date"):
        # Players whose NBA team plays that day
        playing = sub.copy()

        if playing.empty:
            rows.append({"date": d, "active_players": 0, "slots_left": max_active_per_day})
            continue

        playing.sort_values("avg_fp", ascending=False, inplace=True)
        active = playing.head(max_active_per_day)

        active_players = len(active)
        slots_left = max(0, max_active_per_day - active_players)

        rows.append(
            {
                "date": d,
                "active_players": active_players,
                "slots_left": slots_left,
            }
        )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def rank_drop_candidates(
    roster: pd.DataFrame,
    schedule_future: pd.DataFrame,
    avg_fp_threshold: float = AVG_FP_DROP_THRESHOLD,
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Rank drop candidates based on:
      - avg_fp < avg_fp_threshold
      - fewest remaining games (rest-of-week)
      - then lowest avg_fp

    Returns DataFrame:
      player_id, name, pos, avg_fp, team_abbr, games_remaining, priority_rank
    """
    rs = core.attach_schedule_to_roster(roster, schedule_future)
    if rs.empty:
        # No remaining games – in theory everyone is droppable, but we return empty for now.
        return pd.DataFrame(columns=[
            "player_id", "name", "pos", "avg_fp", "team_abbr",
            "games_remaining", "priority_rank"
        ])

    rs["date"] = pd.to_datetime(rs["date"]).dt.normalize()

    # Count remaining games per player
    games_remaining = (
        rs.groupby("player_id", as_index=False)["date"]
          .nunique()
          .rename(columns={"date": "games_remaining"})
    )

    # Join back to roster to get names/positions/avg_fp
    merged = roster.merge(games_remaining, on="player_id", how="left")
    merged["games_remaining"] = merged["games_remaining"].fillna(0).astype(int)

    # Filter to under threshold
    cand = merged[merged["avg_fp"] < avg_fp_threshold].copy()
    if cand.empty:
        return pd.DataFrame(columns=[
            "player_id", "name", "pos", "avg_fp", "team_abbr",
            "games_remaining", "priority_rank"
        ])

    cand.sort_values(
        ["games_remaining", "avg_fp"],
        ascending=[True, True],
        inplace=True,
    )

    cand["priority_rank"] = range(1, len(cand) + 1)
    return cand.head(top_n).reset_index(drop=True)

# ---------------------------------------------------------------------
# Baseline projection (rest-of-week only)
# ---------------------------------------------------------------------

def build_baseline_projection(
    today: Optional[pd.Timestamp] = None,
) -> BaselineProjection:
    my_roster, opp_roster, schedule, my_score, opp_score = load_all_data()

    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.to_datetime(today).normalize()

    schedule = schedule.copy()
    schedule["date"] = pd.to_datetime(schedule["date"]).dt.normalize()
    schedule_future = schedule[schedule["date"] >= today].copy()

    if schedule_future.empty:
        # No remaining games
        win_prob = core.compute_win_probability(my_score, opp_score, k=core.K_LOGIT)
        empty_series = pd.Series(dtype=float)
        empty_df = pd.DataFrame(columns=["date", "active_players", "slots_left"])
        summary = pd.DataFrame(columns=["date", "my_proj_fp_rest", "opp_proj_fp_rest"])
        return BaselineProjection(
            dates=[],
            my_daily=empty_series,
            opp_daily=empty_series,
            my_rest=0.0,
            opp_rest=0.0,
            my_final=my_score,
            opp_final=opp_score,
            win_prob=win_prob,
            my_slots_capacity=0,
            opp_slots_capacity=0,
            my_active_table=empty_df,
            opp_active_table=empty_df,
            summary=summary,
        )

    # Attach schedule & daily projections (FP)
    my_rs = core.attach_schedule_to_roster(my_roster, schedule_future)
    opp_rs = core.attach_schedule_to_roster(opp_roster, schedule_future)

    dnp_set = core.build_dnp_set([]) if hasattr(core, "build_dnp_set") else set()
    my_rs = core.apply_dnp_overrides(my_rs, side_label="my", dnp_set=dnp_set)
    opp_rs = core.apply_dnp_overrides(opp_rs, side_label="opp", dnp_set=dnp_set)

    my_daily_fp = core.project_daily_points(my_rs, core.MAX_ACTIVE_PER_DAY)
    opp_daily_fp = core.project_daily_points(opp_rs, core.MAX_ACTIVE_PER_DAY)

    # Align dates
    all_dates = sorted(set(my_daily_fp.index) | set(opp_daily_fp.index))
    my_daily_fp = my_daily_fp.reindex(all_dates, fill_value=0.0)
    opp_daily_fp = opp_daily_fp.reindex(all_dates, fill_value=0.0)

    # Remaining totals
    my_rest = float(my_daily_fp.sum())
    opp_rest = float(opp_daily_fp.sum())

    # Final totals
    my_final = my_score + my_rest
    opp_final = opp_score + opp_rest

    # Win probability
    win_prob = core.compute_win_probability(my_final, opp_final, k=core.K_LOGIT)

    # Daily active & slots-left tables
    my_active = compute_daily_active_and_slots(my_roster, schedule_future, max_active_per_day=core.MAX_ACTIVE_PER_DAY)
    opp_active = compute_daily_active_and_slots(opp_roster, schedule_future, max_active_per_day=core.MAX_ACTIVE_PER_DAY)

    my_slots_capacity = int(my_active["slots_left"].sum()) if not my_active.empty else 0
    opp_slots_capacity = int(opp_active["slots_left"].sum()) if not opp_active.empty else 0

    # Summary table for inspection / UI
    summary = pd.DataFrame(
        {
            "date": all_dates,
            "my_proj_fp_rest": my_daily_fp.values,
            "opp_proj_fp_rest": opp_daily_fp.values,
        }
    )
    summary["my_cum_rest"] = summary["my_proj_fp_rest"].cumsum()
    summary["opp_cum_rest"] = summary["opp_proj_fp_rest"].cumsum()

    return BaselineProjection(
        dates=all_dates,
        my_daily=my_daily_fp,
        opp_daily=opp_daily_fp,
        my_rest=my_rest,
        opp_rest=opp_rest,
        my_final=my_final,
        opp_final=opp_final,
        win_prob=win_prob,
        my_slots_capacity=my_slots_capacity,
        opp_slots_capacity=opp_slots_capacity,
        my_active_table=my_active,
        opp_active_table=opp_active,
        summary=summary,
    )

# ---------------------------------------------------------------------
# Scenario grid (0..MAX_ADDS, respecting capacity)
# ---------------------------------------------------------------------

def build_add_drop_scenarios(
    baseline: BaselineProjection,
    max_adds: int = MAX_ADDS,
    add_fp: float = ADD_FP,
) -> pd.DataFrame:
    """
    0..max_adds for my team and opponent, but effective adds are capped by
    each team's streaming capacity (sum of slots_left over remaining days).
    """
    # Recover current scores
    my_current = baseline.my_final - baseline.my_rest
    opp_current = baseline.opp_final - baseline.opp_rest

    rows = []
    for my_adds in range(0, max_adds + 1):
        for opp_adds in range(0, max_adds + 1):
            my_effective = min(my_adds, baseline.my_slots_capacity)
            opp_effective = min(opp_adds, baseline.opp_slots_capacity)

            my_rest_prime = baseline.my_rest + my_effective * add_fp
            opp_rest_prime = baseline.opp_rest + opp_effective * add_fp

            my_final_prime = my_current + my_rest_prime
            opp_final_prime = opp_current + opp_rest_prime

            win_prob = core.compute_win_probability(
                my_final_prime, opp_final_prime, k=core.K_LOGIT
            )

            rows.append(
                {
                    "my_adds": my_adds,
                    "opp_adds": opp_adds,
                    "my_effective_adds": my_effective,
                    "opp_effective_adds": opp_effective,
                    "my_final": my_final_prime,
                    "opp_final": opp_final_prime,
                    "margin": my_final_prime - opp_final_prime,
                    "win_prob": win_prob,
                }
            )

    df = pd.DataFrame(rows).sort_values(["my_adds", "opp_adds"]).reset_index(drop=True)
    return df


def compute_add_budget(
    adds_used_to_date: int,
    current_week: int,
    total_adds_cap: int = TOTAL_ADDS_CAP,
    reg_season_weeks: int = REG_SEASON_WEEKS,
    playoff_adds_reserved: int = PLAYOFF_ADDS_RESERVED,
) -> dict:
    """
    Compute how many adds you have left for the regular season,
    and how many adds per week (on average) that implies.

    - total_adds_cap: total season cap (75)
    - playoff_adds_reserved: adds you want to keep for playoffs (e.g. 15)
    - reg_season_weeks: number of regular-season weeks (19)
    - current_week: 1-based index of current week (e.g. 5)
    """
    # Total budget for regular season
    reg_add_budget_total = max(total_adds_cap - playoff_adds_reserved, 0)

    # Treat "adds used so far" as coming out of the regular-season budget
    adds_used_reg_to_date = min(adds_used_to_date, reg_add_budget_total)
    adds_remaining_reg = max(reg_add_budget_total - adds_used_reg_to_date, 0)

    # Weeks remaining
    weeks_remaining_reg_incl = max(reg_season_weeks - current_week + 1, 1)  # include current
    weeks_remaining_reg_excl = max(reg_season_weeks - current_week, 0)      # future only

    avg_adds_per_week_incl = adds_remaining_reg / weeks_remaining_reg_incl
    avg_adds_per_week_excl = (
        adds_remaining_reg / weeks_remaining_reg_excl
        if weeks_remaining_reg_excl > 0
        else adds_remaining_reg
    )

    return {
        "reg_add_budget_total": reg_add_budget_total,
        "adds_used_reg_to_date": adds_used_reg_to_date,
        "adds_remaining_reg": adds_remaining_reg,
        "weeks_remaining_reg_incl": weeks_remaining_reg_incl,
        "weeks_remaining_reg_excl": weeks_remaining_reg_excl,
        "avg_adds_per_week_incl": avg_adds_per_week_incl,
        "avg_adds_per_week_excl": avg_adds_per_week_excl,
    }




def build_streaming_what_if_grid(
    baseline: BaselineProjection,
    max_my_adds: int = MAX_ADDS,
    max_opp_adds: int = MAX_ADDS,
    add_fp: float = ADD_FP,
) -> pd.DataFrame:
    """
    "Streaming" what-if grid using the same placeholder add value (add_fp)
    and the same capacity constraints as build_add_drop_scenarios, but
    with extra columns to show:
      - extra FP vs baseline
      - updated final totals
      - win probability

    No free_agents.csv is used here. Each *effective* add is worth add_fp
    for the rest of the week, capped by:
        baseline.my_slots_capacity
        baseline.opp_slots_capacity
    """
    # Recover current scores from baseline
    my_current = baseline.my_final - baseline.my_rest
    opp_current = baseline.opp_final - baseline.opp_rest

    rows = []
    for my_adds in range(0, max_my_adds + 1):
        for opp_adds in range(0, max_opp_adds + 1):
            # Capacity cap: you can't get more "effective" adds than slots_left
            my_eff = min(my_adds, baseline.my_slots_capacity)
            opp_eff = min(opp_adds, baseline.opp_slots_capacity)

            my_extra_fp = my_eff * add_fp
            opp_extra_fp = opp_eff * add_fp

            # Rest-of-week FP including extra streaming
            my_rest_prime = baseline.my_rest + my_extra_fp
            opp_rest_prime = baseline.opp_rest + opp_extra_fp

            # New projected finals
            my_final_prime = my_current + my_rest_prime
            opp_final_prime = opp_current + opp_rest_prime

            win_prob = core.compute_win_probability(
                my_final_prime, opp_final_prime, k=core.K_LOGIT
            )

            rows.append(
                {
                    "my_adds": my_adds,
                    "opp_adds": opp_adds,
                    "my_effective_adds": my_eff,
                    "opp_effective_adds": opp_eff,
                    "my_extra_fp": my_extra_fp,
                    "opp_extra_fp": opp_extra_fp,
                    "my_proj_final": my_final_prime,
                    "opp_proj_final": opp_final_prime,
                    "point_diff": my_final_prime - opp_final_prime,
                    "win_prob": win_prob,
                    "delta_my_vs_baseline": my_final_prime - baseline.my_final,
                    "delta_opp_vs_baseline": opp_final_prime - baseline.opp_final,
                }
            )

    df = pd.DataFrame(rows).sort_values(["my_adds", "opp_adds"]).reset_index(drop=True)
    return df



# ---------------------------------------------------------------------
# Streaming team ranking (unchanged logic)
# ---------------------------------------------------------------------


def compute_efficient_frontier(
    streaming_what_if: pd.DataFrame,
    adds_budget_info: dict,
    opp_adds_assumed: int = 0,
) -> pd.DataFrame:
    """
    Efficient frontier for my_adds, holding opp_adds fixed.

    - For a given opp_adds_assumed (e.g. 0), we:
        * filter streaming_what_if to that opp_adds
        * for each my_adds, keep the row with max win_prob
        * then keep only rows where win_prob strictly improves as my_adds increases
    - Adds budget info is used to compute how much of your remaining
      regular-season budget each my_adds choice consumes.
    """
    df = streaming_what_if.copy()
    df = df[df["opp_adds"] == opp_adds_assumed].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "my_adds",
                "opp_adds",
                "my_proj_final",
                "opp_proj_final",
                "win_prob",
                "win_prob_gain",
                "win_prob_gain_per_add",
                "adds_share_of_reg_remaining",
            ]
        )

    # Baseline win prob for this opp_adds (my_adds = 0)
    base_rows = df[df["my_adds"] == 0]
    if base_rows.empty:
        baseline_wp = float(df["win_prob"].iloc[0])
    else:
        baseline_wp = float(base_rows["win_prob"].iloc[0])

    # For each my_adds, keep the row with highest win_prob
    best_per_add = (
        df.sort_values("win_prob", ascending=False)
          .groupby("my_adds", as_index=False)
          .first()
          .sort_values("my_adds")
          .reset_index(drop=True)
    )

    # Efficient frontier: only keep rows where win_prob strictly improves
    frontier_rows = []
    best_seen = -1.0
    for _, row in best_per_add.iterrows():
        wp = float(row["win_prob"])
        if wp > best_seen + 1e-9:
            frontier_rows.append(row)
            best_seen = wp

    if not frontier_rows:
        return pd.DataFrame(
            columns=[
                "my_adds",
                "opp_adds",
                "my_proj_final",
                "opp_proj_final",
                "win_prob",
                "win_prob_gain",
                "win_prob_gain_per_add",
                "adds_share_of_reg_remaining",
            ]
        )

    frontier = pd.DataFrame(frontier_rows)

    # Extra metrics
    frontier["win_prob_gain"] = frontier["win_prob"] - baseline_wp
    frontier["win_prob_gain_per_add"] = np.where(
        frontier["my_adds"] > 0,
        frontier["win_prob_gain"] / frontier["my_adds"],
        np.nan,
    )

    adds_remaining_reg = adds_budget_info.get("adds_remaining_reg", 0)
    frontier["adds_share_of_reg_remaining"] = np.where(
        adds_remaining_reg > 0,
        frontier["my_adds"] / adds_remaining_reg,
        np.nan,
    )

    # Reorder useful columns
    cols = [
        "my_adds",
        "opp_adds",
        "my_proj_final",
        "opp_proj_final",
        "win_prob",
        "win_prob_gain",
        "win_prob_gain_per_add",
        "adds_share_of_reg_remaining",
    ]
    frontier = frontier[cols]
    return frontier




def rank_streaming_teams(
    schedule: pd.DataFrame,
    horizon_days: int = STREAM_HORIZON_DAYS,
    today: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.to_datetime(today).normalize()

    end_date = today + pd.Timedelta(days=horizon_days - 1)

    sched = schedule.copy()
    sched["date"] = pd.to_datetime(sched["date"]).dt.normalize()
    window_sched = sched[(sched["date"] >= today) & (sched["date"] <= end_date)]

    if window_sched.empty:
        return pd.DataFrame(columns=["team_abbr", "games_next_n_days"])

    # from projection_core.load_schedule, schedule likely has "team" col
    key_col = "team_abbr" if "team_abbr" in window_sched.columns else "team"
    window_sched[key_col] = window_sched[key_col].astype(str).str.upper()

    counts = (
        window_sched.groupby(key_col, as_index=False)
        .size()
        .rename(columns={"size": "games_next_n_days"})
    )

    return counts.sort_values("games_next_n_days", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------

def rank_streaming_teams_with_slots(
    schedule_future: pd.DataFrame,
    active_table: pd.DataFrame,
    horizon_days: int = STREAM_HORIZON_DAYS,
    today: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Rank NBA teams as streaming targets over the next `horizon_days`,
    explicitly accounting for:
      - What days they play
      - How many *slots_left* you actually have on those days

    Inputs:
      schedule_future: DataFrame with at least ['date', 'team'] or ['date', 'team_abbr']
      active_table   : DataFrame from compute_daily_active_and_slots(...) for a
                       specific fantasy team, with columns:
                           ['date', 'active_players', 'slots_left']
    """
    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.to_datetime(today).normalize()

    end_date = today + pd.Timedelta(days=horizon_days - 1)

    sched = schedule_future.copy()
    sched["date"] = pd.to_datetime(sched["date"]).dt.normalize()

    # Ensure we have a 'team_abbr' column for grouping, without breaking 'team'
    if "team_abbr" not in sched.columns:
        if "team" in sched.columns:
            sched["team_abbr"] = sched["team"].astype(str).str.upper()
        else:
            raise KeyError("schedule_future must contain either 'team' or 'team_abbr'")

    window_sched = sched[(sched["date"] >= today) & (sched["date"] <= end_date)]
    if window_sched.empty or active_table.empty:
        return pd.DataFrame(columns=["team_abbr", "games_next_n_days", "usable_stream_games"])

    act = active_table.copy()
    act["date"] = pd.to_datetime(act["date"]).dt.normalize()

    merged = window_sched.merge(
        act[["date", "slots_left"]],
        on="date",
        how="left",
    )
    merged["slots_left"] = merged["slots_left"].fillna(0).astype(int)
    merged["usable_game_flag"] = (merged["slots_left"] > 0).astype(int)

    # If schedule has a 'games' column (core.load_schedule usually does), use it
    if "games" in merged.columns:
        merged["usable_games_count"] = merged["games"] * merged["usable_game_flag"]
        games_col = "games"
    else:
        # Fallback: assume 1 game per row
        merged["usable_games_count"] = merged["usable_game_flag"]
        games_col = "usable_game_flag"

    grouped = (
        merged.groupby("team_abbr", as_index=False)
        .agg(
            games_next_n_days=(games_col, "sum"),
            usable_stream_games=("usable_games_count", "sum"),
        )
    )

    grouped = grouped.sort_values(
        ["usable_stream_games", "games_next_n_days"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return grouped



def main() -> None:
    print("=== FBBK Strategy Engine (refined streaming) ===\n")

    # 1) Baseline projection
    baseline = build_baseline_projection()
    print("Baseline rest-of-week projection:")
    print(f"  My remaining FP        : {baseline.my_rest:.1f}")
    print(f"  Opp remaining FP       : {baseline.opp_rest:.1f}")
    print(f"  My projected final     : {baseline.my_final:.1f}")
    print(f"  Opp projected final    : {baseline.opp_final:.1f}")
    print(f"  Baseline win prob      : {baseline.win_prob * 100:5.1f}%")
    print(f"  My slots capacity      : {baseline.my_slots_capacity}")
    print(f"  Opp slots capacity     : {baseline.opp_slots_capacity}\n")

    print("My daily active / slots_left:")
    print(baseline.my_active_table, "\n")
    print("Opponent daily active / slots_left:")
    print(baseline.opp_active_table, "\n")

    # 2) Scenario grid (0..MAX_ADDS adds for each side, capped by slot capacity)
    scenarios = build_add_drop_scenarios(baseline, max_adds=MAX_ADDS, add_fp=ADD_FP)
    print(f"Add/Drop Scenario Grid (each effective add ≈ {ADD_FP:.1f} FP), "
          "respecting 10-player cap:\n")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(scenarios)

    # 2b) Streaming what-if grid using the same add_fp and capacity,
    # but with more explicit "extra FP" columns vs baseline.
    streaming_what_if = build_streaming_what_if_grid(
        baseline=baseline,
        max_my_adds=MAX_ADDS,
        max_opp_adds=MAX_ADDS,
        add_fp=ADD_FP,
    )

    print(f"\nStreaming what-if grid (each effective add ≈ {ADD_FP:.1f} FP, "
          "respecting slots capacity):\n")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(streaming_what_if)

    # 3) Season add budget summary
    budget = compute_add_budget(
        adds_used_to_date=ADDS_USED_TO_DATE,
        current_week=CURRENT_WEEK,
    )
    print("\nSeason add budget summary:")
    print(f"  Total adds cap              : {TOTAL_ADDS_CAP}")
    print(f"  Adds reserved for playoffs  : {PLAYOFF_ADDS_RESERVED}")
    print(f"  Regular-season add budget   : {budget['reg_add_budget_total']}")
    print(f"  Adds used to date (reg est.): {budget['adds_used_reg_to_date']}")
    print(f"  Adds remaining (regular)    : {budget['adds_remaining_reg']}")
    print(f"  Weeks remain incl this week : {budget['weeks_remaining_reg_incl']}")
    print(f"  Avg adds/week (incl this)   : {budget['avg_adds_per_week_incl']:.2f}")
    print(f"  Avg adds/week (future only) : {budget['avg_adds_per_week_excl']:.2f}")

    # 4) Efficient frontier over my_adds, holding opp_adds fixed
    #    (e.g., assume opponent uses 0 adds for this view)
    opp_adds_assumed = 0
    frontier = compute_efficient_frontier(
        streaming_what_if=streaming_what_if,
        adds_budget_info=budget,
        opp_adds_assumed=opp_adds_assumed,
    )

    print(f"\nEfficient frontier for my adds (assuming opp_adds={opp_adds_assumed}):")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(frontier)


    # 4) Drop candidates (under X avg_fp and fewest remaining games)
    my_roster, opp_roster, schedule, _, _ = load_all_data()
    schedule["date"] = pd.to_datetime(schedule["date"]).dt.normalize()
    today = pd.Timestamp.today().normalize()
    schedule_future = schedule[schedule["date"] >= today].copy()

    my_drops = rank_drop_candidates(my_roster, schedule_future)
    opp_drops = rank_drop_candidates(opp_roster, schedule_future)

    print("\nMy drop candidates (avg_fp < 30 and fewest remaining games):")
    print(my_drops)

    print("\nOpponent drop candidates (avg_fp < 30 and fewest remaining games):")
    print(opp_drops)

    # 5) Streaming teams, tied to *your* actual open slots
    my_team_stream_ranks = rank_streaming_teams_with_slots(
        schedule_future=schedule_future,
        active_table=baseline.my_active_table,
        horizon_days=STREAM_HORIZON_DAYS,
        today=today,
    )
    print(f"\nBest NBA teams to stream from for *my team* over next {STREAM_HORIZON_DAYS} days, "
          "accounting for actual open slots:\n")
    print(my_team_stream_ranks)

    # 6) Streaming teams, tied to *opponent's* open slots
    opp_team_stream_ranks = rank_streaming_teams_with_slots(
        schedule_future=schedule_future,
        active_table=baseline.opp_active_table,
        horizon_days=STREAM_HORIZON_DAYS,
        today=today,
    )
    print(f"\nBest NBA teams to stream from for *opponent* over next {STREAM_HORIZON_DAYS} days, "
          "accounting for their open slots:\n")
    print(opp_team_stream_ranks)

    print("\nDone.")


if __name__ == "__main__":
    main()

