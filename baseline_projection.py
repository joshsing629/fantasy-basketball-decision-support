"""
baseline_projection.py

Daily, no-add/drop projection using:
- src/data/my_team.csv
- src/data/opp_team.csv
- src/data/free_agents.csv  (currently unused, but loaded for future extensions)
- src/data/nba_schedule_next_7_days.csv
- src/data/current_score.csv

Assumptions:
- my_team.csv, opp_team.csv columns:
    player_id, name, pos, avg_fp, team_abbr
- nba_schedule_next_7_days.csv columns:
    date, team, opponent, home_away
- current_score.csv columns:
    team_id, points, projected   (we ignore 'projected')

- Your team_id  = 10
- Opponent id   = 8
- Max active per day = 10 (ignoring positional constraints for now)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd


# --------- Config / constants ---------

MY_TEAM_ID = 10
OPP_TEAM_ID = 8

# Yahoo lineup: PG, SG, G, SF, PF, F, C, Util, Util, Util  -> 10 slots
MAX_ACTIVE_PER_DAY = 10


# --------- CSV loaders ---------

def load_roster(path: Path) -> pd.DataFrame:
    """
    Load a roster (my_team or opp_team).

    Expected columns:
        player_id, name, pos, avg_fp, team_abbr
    """
    df = pd.read_csv(path)

    # Basic sanity checks / coercions
    needed_cols = {"player_id", "name", "pos", "avg_fp", "team_abbr"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df["player_id"] = df["player_id"].astype(str)
    df["avg_fp"] = pd.to_numeric(df["avg_fp"], errors="coerce").fillna(0.0)
    df["team_abbr"] = df["team_abbr"].astype(str).str.upper()

    return df[["player_id", "name", "pos", "team_abbr", "avg_fp"]]


def load_schedule(path: Path) -> pd.DataFrame:
    """
    Load NBA team schedule for the next 7 days.

    Expected columns:
        date, team, opponent, home_away

    'team' and 'opponent' should be NBA abbreviations like 'PHI', 'DAL'.
    """
    df = pd.read_csv(path)

    needed_cols = {"date", "team", "opponent", "home_away"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["team"] = df["team"].astype(str).str.upper()
    df["opponent"] = df["opponent"].astype(str).str.upper()
    df["home_away"] = df["home_away"].astype(str)

    return df[["date", "team", "opponent", "home_away"]]


def load_current_score(path: Path) -> Tuple[float, float]:
    """
    Load current team scores from current_score.csv.

    Columns:
        team_id, points, projected

    Returns:
        my_score, opp_score  (for MY_TEAM_ID and OPP_TEAM_ID)
    """
    df = pd.read_csv(path)

    needed_cols = {"team_id", "points"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)

    def _score_for(team_id: int) -> float:
        sub = df.loc[df["team_id"] == team_id, "points"]
        if sub.empty:
            raise ValueError(f"No row in {path} for team_id={team_id}")
        return float(sub.iloc[0])

    my_score = _score_for(MY_TEAM_ID)
    opp_score = _score_for(OPP_TEAM_ID)

    return my_score, opp_score


# --------- Daily simulation helpers ---------

def simulate_team_future_points(
    roster: pd.DataFrame,
    schedule: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_active_per_day: int = MAX_ACTIVE_PER_DAY,
) -> Tuple[float, pd.DataFrame]:
    """
    Simulate remaining fantasy points from start_date through end_date (inclusive)
    for a given team's roster.

    For each date:
      - Find players whose NBA team has a game that day (using team_abbr -> schedule.team)
      - Sort those players by avg_fp descending
      - Take up to `max_active_per_day` players
      - Sum their avg_fp as that day's projection

    Returns:
        total_future_points, daily_detail_df
        where daily_detail_df has columns:
            date, active_players, proj_points
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    total = 0.0
    rows = []

    # Pre-group schedule by date for efficiency
    sched_by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        d: g for d, g in schedule.groupby("date")
    }

    for d in dates:
        day = d.normalize()

        sched_today = sched_by_date.get(day)
        if sched_today is None or sched_today.empty:
            # No games today
            rows.append({"date": day, "active_players": 0, "proj_points": 0.0})
            continue

        # Which NBA teams play today?
        teams_today = set(sched_today["team"].unique())

        # Players whose team_abbr is in teams_today
        mask_playing = roster["team_abbr"].isin(teams_today)
        todays_players = roster.loc[mask_playing].copy()

        if todays_players.empty:
            rows.append({"date": day, "active_players": 0, "proj_points": 0.0})
            continue

        todays_players = todays_players.sort_values("avg_fp", ascending=False)

        active = todays_players.head(max_active_per_day)
        day_points = float(active["avg_fp"].sum())
        total += day_points

        rows.append(
            {
                "date": day,
                "active_players": len(active),
                "proj_points": day_points,
            }
        )

    detail_df = pd.DataFrame(rows)
    return total, detail_df


def logistic_win_prob(point_diff: float, k: float = 0.01) -> float:
    """
    Simple logistic win probability function.

    point_diff > 0 means we're projected ahead.
    """
    return 1.0 / (1.0 + np.exp(-k * point_diff))


# --------- Main script ---------

def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # 1. Load data
    my_team_path = base_dir / "src/data/my_team.csv"
    opp_team_path = base_dir / "src/data/opp_team.csv"
    free_agents_path = base_dir / "src/data/free_agents.csv"  # not yet used
    schedule_path = base_dir / "src/data/nba_schedule_next_7_days.csv"
    current_score_path = base_dir / "src/data/current_score.csv"

    print("Loading CSVs...")
    my_team = load_roster(my_team_path)
    opp_team = load_roster(opp_team_path)
    schedule = load_schedule(schedule_path)
    my_score, opp_score = load_current_score(current_score_path)

    # Optional: just verify free_agents loads cleanly for future use
    try:
        fa_df = load_roster(free_agents_path)
        print(f"Loaded free agents: {len(fa_df)} players")
    except Exception as e:
        print(f"Warning: could not load free_agents.csv cleanly: {e}")

    print("\n--- Basic info ---")
    print(f"My players: {len(my_team)}")
    print(f"Opponent players: {len(opp_team)}")
    print(
        f"Schedule dates: {schedule['date'].min().date()} "
        f"to {schedule['date'].max().date()}"
    )
    print(f"Current score -> Me: {my_score:.1f}, Opp: {opp_score:.1f}")

    # 2. Decide simulation window: from today to last date in schedule
    today = pd.Timestamp.today().normalize()
    last_sched_date = schedule["date"].max()

    # If today is before the schedule window, start at the first schedule date
    start_date = max(today, schedule["date"].min())
    end_date = last_sched_date

    if start_date > end_date:
        print("\nNo remaining games in the schedule window. Projection = current score only.")
        my_total = my_score
        opp_total = opp_score
        diff = my_total - opp_total
        win_prob = logistic_win_prob(diff)

        print(f"Projected final -> Me: {my_total:.1f}, Opp: {opp_total:.1f}")
        print(f"Win probability (logistic, no future games): {win_prob:.1%}")
        return

    print(f"\nSimulating from {start_date.date()} to {end_date.date()} (inclusive)...")

    # Filter schedule to remaining days
    remaining_schedule = schedule.loc[schedule["date"] >= start_date].copy()

    # 3. Simulate future points for each team
    my_future, my_daily = simulate_team_future_points(
        my_team, remaining_schedule, start_date, end_date
    )
    opp_future, opp_daily = simulate_team_future_points(
        opp_team, remaining_schedule, start_date, end_date
    )

    print("\n--- Daily projection (my team) ---")
    print(my_daily.to_string(index=False))

    print("\n--- Daily projection (opponent) ---")
    print(opp_daily.to_string(index=False))

    # 4. Combine with current score
    my_total = my_score + my_future
    opp_total = opp_score + opp_future
    diff = my_total - opp_total
    win_prob = logistic_win_prob(diff)

    print("\n--- Baseline projection ---")
    print(f"Remaining points -> Me: {my_future:.1f}, Opp: {opp_future:.1f}")
    print(f"Projected final -> Me: {my_total:.1f}, Opp: {opp_total:.1f}")
    print(f"Projected point diff (Me - Opp): {diff:.1f}")
    print(f"Win probability (logistic): {win_prob:.1%}")


if __name__ == "__main__":
    main()
