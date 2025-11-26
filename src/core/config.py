"""
Configuration objects for your fantasy basketball assistant.

These wrap constants in simple data classes so they can be passed
around the codebase cleanly (to the optimizer, simulator, etc.).
"""

from dataclasses import dataclass
from typing import Dict, Set, List, Optional

from .constants import (
    YAHOO_GAME_KEY,
    YAHOO_LEAGUE_ID,
    YAHOO_TEAM_ID,
    ROSTER_SLOTS,
    BENCH_SLOTS,
    IL_SLOTS,
    MAX_ACTIVE_PER_DAY,
    SEASON_ADD_LIMIT,
    WEEKLY_ADD_LIMIT,
    REGULAR_SEASON_WEEKS,
    PLAYOFF_WEEKS,
    YAHOO_POINTS_SCORING,
    SLOT_POSITION_RULES,
)


@dataclass
class LeagueConfig:
    game_key: str
    league_id: str
    team_id: Optional[str]

    roster_slots: List[str]
    bench_slots: int
    il_slots: int
    max_active_per_day: int

    season_add_limit: int
    weekly_add_limit: int
    regular_season_weeks: int
    playoff_weeks: int

    scoring_weights: Dict[str, float]
    slot_position_rules: Dict[str, Set[str]]


@dataclass
class SimulationConfig:
    """
    Config specific to the simulation / modeling engine.

    You can tune these to change how aggressive or conservative
    the projections and win probabilities are.
    """
    sd_factor_per_game: float = 0.35
    questionable_multiplier: float = 0.75
    default_add_value_threshold: float = 0.05


def get_league_config() -> LeagueConfig:
    """
    Factory function to build the LeagueConfig used across the project.
    """
    slot_rules: Dict[str, Set[str]] = {
        slot: set(positions) for slot, positions in SLOT_POSITION_RULES.items()
    }

    return LeagueConfig(
        game_key=YAHOO_GAME_KEY,
        league_id=YAHOO_LEAGUE_ID,
        team_id=YAHOO_TEAM_ID,
        roster_slots=list(ROSTER_SLOTS),
        bench_slots=BENCH_SLOTS,
        il_slots=IL_SLOTS,
        max_active_per_day=MAX_ACTIVE_PER_DAY,
        season_add_limit=SEASON_ADD_LIMIT,
        weekly_add_limit=WEEKLY_ADD_LIMIT,
        regular_season_weeks=REGULAR_SEASON_WEEKS,
        playoff_weeks=PLAYOFF_WEEKS,
        scoring_weights=dict(YAHOO_POINTS_SCORING),
        slot_position_rules=slot_rules,
    )


def get_simulation_config() -> SimulationConfig:
    """
    Factory function to build the SimulationConfig used by the engine.
    """
    return SimulationConfig()
