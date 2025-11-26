"""
projection_core.py

Daily projection engine for FBBK decision support.

- Uses current score and remaining schedule (next_7_days file)
- Projects daily & total rest-of-week fantasy points for:
    * My team (MY_TEAM_ID)
    * Opponent (OPP_TEAM_ID)
- Enforces max 10 active players per day.
- Supports game-level DNP overrides (per player, per date).
- Computes win probability with logistic model, k = 0.01.

CSV inputs (relative to repo root):
    src/data/my_team.csv
    src/data/opp_team.csv
    src/data/nba_schedule_next_7_days.csv
    src/data/current_score.csv
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Fantasy team IDs in current_score.csv (defaults)
MY_TEAM_ID = 10
OPP_TEAM_ID = 3

# Path setup (relative to this file)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "src" / "data"

MY_TEAM_PATH = DATA_DIR / "my_team.csv"
OPP_TEAM_PATH = DATA_DIR / "opp_team.csv"
SCHEDULE_PATH = DATA_DIR / "nba_schedule_next_7_days.csv"
CURRENT_SCORE_PATH = DATA_DIR / "current_score.csv"

# Fantasy rules
MAX_ACTIVE_PER_DAY = 10

# Win-probability slope
K_LOGIT = 0.01

# ---------------------------------------------------------------------
# Dataclasses used by UI / strategy layer
# ---------------------------------------------------------------------


@dataclass
class TeamProjection:
    daily: pd.DataFrame  # columns: date, proj_points (and optionally active_players)
    rest_total: float    # projected points from today forward
    final_total: float   # rest_total + current_score


@dataclass
class MatchupProjection:
    my: TeamProjection
    opp: TeamProjection
    point_diff: float    # my_final - opp_final
    win_prob: float      # P(win) from logistic model


# ---------------------------------------------------------------------
# DNP OVERRIDES (global, used only by main())
# ---------------------------------------------------------------------
"""
DNP_OVERRIDES is a list of dicts. Each row marks that a player will
NOT play in a specific game, for a specific side ("my" or "opp").

Example:

DNP_OVERRIDES = [
    {"side": "my",  "player_id": "5185", "date": "2025-11-15"},  # Giannis out
    {"side": "opp", "player_id": "5352", "date": "2025-11-16"},  # Jokic out
]
"""

DNP_OVERRIDES: List[Dict[str, str]] = [
    # {"side": "my",  "player_id": "5185", "date": "2025-11-15"},
    # {"side": "opp", "player_id": "5352", "date": "2025-11-16"},
]

# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------


def load_roster(path: Path) -> pd.DataFrame:
    """
    Load a roster CSV with columns:
        player_id, name, pos, avg_fp, team_abbr

    Ensures player_id is string for consistent joining.
    """
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype(str)

    if "avg_fp" not in df.columns:
        raise ValueError(f"'avg_fp' column not found in {path}")
    if "team_abbr" not in df.columns:
        raise ValueError(f"'team_abbr' column not found in {path}")

    df["team_abbr"] = df["team_abbr"].astype(str).str.upper()
    df["avg_fp"] = pd.to_numeric(df["avg_fp"], errors="coerce").fillna(0.0)

    return df


def load_schedule(path: Path) -> pd.DataFrame:
    """
    Load NBA schedule CSV with columns:
        date, team, opponent, home_away

    Returns DataFrame with date as datetime64[ns].
    """
    df = pd.read_csv(path)
    if "date" not in df.columns or "team" not in df.columns:
        raise ValueError("Schedule file must contain 'date' and 'team' columns.")

    df["date"] = pd.to_datetime(df["date"])
    df["team"] = df["team"].astype(str).str.upper()
    df["opponent"] = df["opponent"].astype(str).str.upper()

    return df


def load_current_score(path: Path) -> Tuple[float, float]:
    """
    Always reads scores for:
        MY_TEAM_ID
        OPP_TEAM_ID

    No overrides allowed.
    """
    df = pd.read_csv(path)
    if "team_id" not in df.columns or "points" not in df.columns:
        raise ValueError("current_score.csv must contain 'team_id' and 'points'.")

    def _get_score(team_id: int) -> float:
        mask = df["team_id"] == team_id
        if not mask.any():
            raise ValueError(f"Team_id {team_id} not found in current_score.csv")
        return float(df.loc[mask, "points"].iloc[0])

    my_score = _get_score(MY_TEAM_ID)
    opp_score = _get_score(OPP_TEAM_ID)

    return my_score, opp_score



# ---------------------------------------------------------------------
# DNP logic helpers for CLI (global overrides)
# ---------------------------------------------------------------------


def build_dnp_set(overrides: Iterable[Dict[str, str]]) -> set:
    """
    Convert DNP_OVERRIDES list into a set of keys for fast lookup.

    Key format: (side, player_id, date_str) where date_str is 'YYYY-MM-DD'.
    """
    dnp = set()
    for row in overrides:
        side = row["side"]
        pid = str(row["player_id"])
        date_str = str(row["date"])
        dnp.add((side, pid, date_str))
    return dnp


# ---------------------------------------------------------------------
# Projection engine primitives
# ---------------------------------------------------------------------


def attach_schedule_to_roster(
    roster: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join player roster with NBA schedule on team abbreviation.

    Returns DataFrame with one row per (player, date where his team plays),
    with at minimum:
        player_id, name, pos, avg_fp, team_abbr, date, opponent, home_away
    """
    merged = roster.merge(
        schedule,
        left_on="team_abbr",
        right_on="team",
        how="left",
        suffixes=("", "_sched"),
    )
    merged = merged.dropna(subset=["date"])
    keep_cols = [
        "player_id",
        "name",
        "pos",
        "avg_fp",
        "team_abbr",
        "date",
        "opponent",
        "home_away",
    ]
    merged = merged[keep_cols].copy()
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def apply_dnp_overrides(
    roster_with_sched: pd.DataFrame,
    side_label: str,
    dnp_set: set,
) -> pd.DataFrame:
    """
    Remove rows where a DNP override exists for (side, player_id, date).

    side_label: "my" or "opp"
    dnp_set: set of (side, player_id, date_str)
    """
    if not dnp_set:
        return roster_with_sched

    mask = []
    for idx, row in roster_with_sched.iterrows():
        key = (
            side_label,
            str(row["player_id"]),
            row["date"].strftime("%Y-%m-%d"),
        )
        mask.append(key not in dnp_set)
    mask = pd.Series(mask, index=roster_with_sched.index)
    return roster_with_sched[mask].copy()


def project_daily_points(
    roster_with_sched: pd.DataFrame,
    max_active_per_day: int = MAX_ACTIVE_PER_DAY,
) -> pd.Series:
    """
    Given a roster with attached schedule (one row per player-game), compute
    projected fantasy points per date by:
        - For each date, sort players by avg_fp descending
        - Take top `max_active_per_day` players
        - Sum their avg_fp for that date

    Returns a Series: index = date (datetime), value = projected points.
    """
    if roster_with_sched.empty:
        return pd.Series(dtype=float)

    df = roster_with_sched.sort_values(["date", "avg_fp"], ascending=[True, False])

    def _top_n_sum(group: pd.DataFrame) -> float:
        top = group.head(max_active_per_day)
        return float(top["avg_fp"].sum())

    daily = df.groupby("date").apply(_top_n_sum)
    daily.name = "proj_fp"
    return daily


def compute_win_probability(
    final_my: float,
    final_opp: float,
    k: float = K_LOGIT,
) -> float:
    """
    Logistic win probability with slope k.
    P(win) = 1 / (1 + exp(-k * (my - opp))).
    """
    diff = final_my - final_opp
    prob = 1.0 / (1.0 + math.exp(-k * diff))
    return float(prob)


# ---------------------------------------------------------------------
# High-level matchup projection (used by Streamlit UI)
# ---------------------------------------------------------------------


def _apply_simple_dnp(
    roster_with_sched: pd.DataFrame,
    overrides: Optional[Iterable[Dict[str, str]]],
) -> pd.DataFrame:
    """
    Apply per-side DNP overrides where each override is a dict:
        {"player_id": "...", "date": "YYYY-MM-DD"}
    """
    if not overrides:
        return roster_with_sched

    df = roster_with_sched.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    remove_mask = pd.Series(False, index=df.index)
    for ov in overrides:
        pid = str(ov["player_id"])
        date_str = pd.to_datetime(ov["date"]).strftime("%Y-%m-%d")
        remove_mask |= ((df["player_id"] == pid) & (df["date_str"] == date_str))

    df = df[~remove_mask].drop(columns=["date_str"])
    return df


def project_matchup_rest_of_week(
    my_roster: pd.DataFrame,
    opp_roster: pd.DataFrame,
    schedule: pd.DataFrame,
    my_current_score: float,
    opp_current_score: float,
    my_dnp_overrides: Optional[Iterable[Dict[str, str]]] = None,
    opp_dnp_overrides: Optional[Iterable[Dict[str, str]]] = None,
    winprob_k: float = K_LOGIT,
    max_active_per_day: int = MAX_ACTIVE_PER_DAY,
) -> MatchupProjection:
    """
    High-level function for UI / strategy layer.

    - Uses *today* as the cutoff.
    - Filters schedule to today..end.
    - Attaches schedule to both rosters.
    - Applies per-side DNP overrides (per-game).
    - Projects rest-of-week fantasy points.
    - Adds current score.
    - Returns MatchupProjection dataclass.
    """
    today = pd.Timestamp.today().normalize()
    schedule = schedule.copy()
    schedule["date"] = pd.to_datetime(schedule["date"]).dt.normalize()

    schedule_future = schedule[schedule["date"] >= today].copy()
    if schedule_future.empty:
        # No remaining games: everything is current-score only
        my_team_proj = TeamProjection(
            daily=pd.DataFrame(columns=["date", "proj_points"]),
            rest_total=0.0,
            final_total=float(my_current_score),
        )
        opp_team_proj = TeamProjection(
            daily=pd.DataFrame(columns=["date", "proj_points"]),
            rest_total=0.0,
            final_total=float(opp_current_score),
        )
        win_prob = compute_win_probability(my_team_proj.final_total,
                                           opp_team_proj.final_total,
                                           k=winprob_k)
        return MatchupProjection(
            my=my_team_proj,
            opp=opp_team_proj,
            point_diff=my_team_proj.final_total - opp_team_proj.final_total,
            win_prob=win_prob,
        )

    # Attach schedule for remaining days only
    my_rs = attach_schedule_to_roster(my_roster, schedule_future)
    opp_rs = attach_schedule_to_roster(opp_roster, schedule_future)

    # Apply simple DNP overrides from UI
    my_rs = _apply_simple_dnp(my_rs, my_dnp_overrides)
    opp_rs = _apply_simple_dnp(opp_rs, opp_dnp_overrides)

    # Daily projections (rest-of-week only)
    my_daily_series = project_daily_points(my_rs, max_active_per_day)
    opp_daily_series = project_daily_points(opp_rs, max_active_per_day)

    # Align dates & fill missing with 0
    all_dates = sorted(set(my_daily_series.index) | set(opp_daily_series.index))
    my_daily_series = my_daily_series.reindex(all_dates, fill_value=0.0)
    opp_daily_series = opp_daily_series.reindex(all_dates, fill_value=0.0)

    # Build daily DataFrames for UI
    my_daily_df = pd.DataFrame(
        {"date": all_dates, "proj_points": my_daily_series.values}
    )
    opp_daily_df = pd.DataFrame(
        {"date": all_dates, "proj_points": opp_daily_series.values}
    )

    # Totals
    my_rest = float(my_daily_series.sum())
    opp_rest = float(opp_daily_series.sum())

    my_final = float(my_current_score) + my_rest
    opp_final = float(opp_current_score) + opp_rest

    win_prob = compute_win_probability(my_final, opp_final, k=winprob_k)

    my_team_proj = TeamProjection(
        daily=my_daily_df,
        rest_total=my_rest,
        final_total=my_final,
    )
    opp_team_proj = TeamProjection(
        daily=opp_daily_df,
        rest_total=opp_rest,
        final_total=opp_final,
    )

    return MatchupProjection(
        my=my_team_proj,
        opp=opp_team_proj,
        point_diff=my_final - opp_final,
        win_prob=win_prob,
    )


# ---------------------------------------------------------------------
# CLI main (unchanged behavior)
# ---------------------------------------------------------------------


def main() -> None:
    print("=== FBBK Daily Projection Engine (rest-of-week only) ===\n")

    print("Loading CSVs from", DATA_DIR)
    my_roster = load_roster(MY_TEAM_PATH)
    opp_roster = load_roster(OPP_TEAM_PATH)
    schedule = load_schedule(SCHEDULE_PATH)
    my_score, opp_score = load_current_score(CURRENT_SCORE_PATH)

    print(f"My current score      : {my_score:.1f}")
    print(f"Opponent current score: {opp_score:.1f}\n")

    today = pd.Timestamp.today().normalize()
    print(f"Treating TODAY as: {today.date()}\n")

    schedule = schedule.copy()
    schedule["date"] = pd.to_datetime(schedule["date"]).dt.normalize()
    schedule_future = schedule[schedule["date"] >= today].copy()

    if schedule_future.empty:
        print("No remaining scheduled games from today onward.")
        total_my = my_score
        total_opp = opp_score
        win_prob = compute_win_probability(total_my, total_opp, k=K_LOGIT)
        print(f"My projected final   : {total_my:.1f}")
        print(f"Opp projected final  : {total_opp:.1f}")
        print(f"Win probability (k={K_LOGIT}): {win_prob * 100:5.1f}%")
        return

    my_rs = attach_schedule_to_roster(my_roster, schedule_future)
    opp_rs = attach_schedule_to_roster(opp_roster, schedule_future)

    dnp_set = build_dnp_set(DNP_OVERRIDES)
    my_rs = apply_dnp_overrides(my_rs, side_label="my", dnp_set=dnp_set)
    opp_rs = apply_dnp_overrides(opp_rs, side_label="opp", dnp_set=dnp_set)

    my_daily = project_daily_points(my_rs, MAX_ACTIVE_PER_DAY)
    opp_daily = project_daily_points(opp_rs, MAX_ACTIVE_PER_DAY)

    all_dates = sorted(set(my_daily.index) | set(opp_daily.index))
    my_daily = my_daily.reindex(all_dates, fill_value=0.0)
    opp_daily = opp_daily.reindex(all_dates, fill_value=0.0)

    summary = pd.DataFrame(
        {
            "date": all_dates,
            "my_proj_fp_rest": my_daily.values,
            "opp_proj_fp_rest": opp_daily.values,
        }
    )
    summary["my_cum_rest"] = summary["my_proj_fp_rest"].cumsum()
    summary["opp_cum_rest"] = summary["opp_proj_fp_rest"].cumsum()

    my_rest = float(summary["my_proj_fp_rest"].sum())
    opp_rest = float(summary["opp_proj_fp_rest"].sum())

    total_my = my_score + my_rest
    total_opp = opp_score + opp_rest
    win_prob = compute_win_probability(total_my, total_opp, k=K_LOGIT)

    print("Daily projection (rest of week from today onward):\n")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(summary)

    print("\n=== Totals ===")
    print(f"Remaining proj (me)  : {my_rest:.1f}")
    print(f"Remaining proj (opp) : {opp_rest:.1f}")
    print(f"My projected final   : {total_my:.1f}")
    print(f"Opp projected final  : {total_opp:.1f}")
    print(f"Projected margin     : {total_my - total_opp:.1f}")
    print(f"Win probability (k={K_LOGIT}): {win_prob * 100:5.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
