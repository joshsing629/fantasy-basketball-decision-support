"""
Yahoo Fantasy Sports API integration for your fantasy basketball assistant.

This module:
- Handles OAuth2 with Yahoo using yahoo_oauth
- Creates a League object via yahoofantasy
- Provides helper functions to fetch league info, teams, and rosters
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from yahoo_oauth import OAuth2
from yahoofantasy import context as ycontext

from src.core.config import get_league_config


# Where your oauth2.json lives (../config/oauth2.json from project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OAUTH_PATH = PROJECT_ROOT / "config" / "oauth2.json"


def get_oauth() -> OAuth2:
    """
    Create (or reuse) an OAuth2 session using the config/oauth2.json file.
    Automatically refreshes the access token if needed.
    """
    if not OAUTH_PATH.exists():
        raise FileNotFoundError(
            f"oauth2.json not found at {OAUTH_PATH}. "
            "Make sure you created config/oauth2.json with your Yahoo keys."
        )

    oauth = OAuth2(None, None, from_file=str(OAUTH_PATH))

    # Refresh token if needed
    if not oauth.token_is_valid():
        oauth.refresh_access_token()

    return oauth


def get_league():
    """
    Return a yahoofantasy League object for your NBA league.
    """
    lc = get_league_config()
    oauth = get_oauth()

    # Create a Game context for NBA
    game = ycontext.Game(oauth, lc.game_key)

    # Build league key: "<game_id>.l.<league_id>"
    # Example: "430.l.9927"
    league_key = f"{game.game_id}.l.{lc.league_id}"

    league = game.to_league(league_key)
    return league


def list_teams() -> Dict[str, Any]:
    """
    Return a dict of all teams in the league.
    Keys are team keys like '430.l.9927.t.1', values are team info dicts.
    """
    league = get_league()
    teams = league.teams()
    return teams


def get_team_key_by_name(team_name_substring: str) -> Optional[str]:
    """
    Find a team key by a case-insensitive substring of the team name.

    Example:
        get_team_key_by_name("Josh") -> '430.l.9927.t.3'
    """
    teams = list_teams()
    for team_key, info in teams.items():
        name = info.get("name", "")
        if team_name_substring.lower() in name.lower():
            return team_key
    return None


def get_team_roster(team_key: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get the roster for a given team in a given scoring week.

    If week is None, the current scoring week is used.
    Returns a list of player dicts.
    """
    league = get_league()
    team = league.to_team(team_key)
    roster = team.roster(week=week)
    return roster


def get_my_team_roster(team_name_hint: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: find your team by name substring and return its roster.
    """
    team_key = get_team_key_by_name(team_name_hint)
    if team_key is None:
        raise ValueError(f"Could not find a team with name containing '{team_name_hint}'")
    return get_team_roster(team_key, week=week)
