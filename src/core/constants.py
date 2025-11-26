"""
Core constants for your fantasy basketball assistant.

These are league-level settings that rarely change.
If the league rules change, you update them here and
the rest of the code can stay the same.
"""

# ---------- LEAGUE & TEAM IDENTIFIERS ----------

# You will fill these in once you integrate Yahoo Fantasy API.
YAHOO_GAME_KEY = "nba"          # Yahoo game code for NBA
YAHOO_LEAGUE_ID = "9927"        # Your league numeric ID as string
YAHOO_TEAM_ID = None            # Your team ID (to be filled from API later)

# Example: full league key often looks like "430.l.9927"
# We'll derive that in the Yahoo API module later.


# ---------- ROSTER STRUCTURE ----------

# Starting lineup slots in your league (order matters for optimizer)
# Your league format:
# PG, SG, G, SF, PF, F, C, Util, Util, Util
ROSTER_SLOTS = [
    "PG",
    "SG",
    "G",
    "SF",
    "PF",
    "F",
    "C",
    "UTIL",
    "UTIL",
    "UTIL",
]

# Bench & IL slots
BENCH_SLOTS = 4
IL_SLOTS = 3

# Max number of players who can actually score in a given day.
# For your league: typically 10 starters, matches the slots above.
MAX_ACTIVE_PER_DAY = 10


# ---------- ADD LIMITS & STRATEGY CONSTANTS ----------

# Season-long adds
SEASON_ADD_LIMIT = 75

# Weekly adds (per scoring period)
WEEKLY_ADD_LIMIT = 5

# How many regular-season weeks + playoff weeks to plan around.
# You can adjust these once you know the exact schedule.
REGULAR_SEASON_WEEKS = 18
PLAYOFF_WEEKS = 3


# ---------- SCORING SETTINGS (YAHOO POINTS LEAGUE) ----------

# Typical Yahoo points scoring (you can update if your league is custom)
# These are here for reference / future use if you want to
# recompute projections from box score stats.
YAHOO_POINTS_SCORING = {
    "PTS": 1.0,     # Points
    "3PM": 1.0,     # 3-pointers made
    "REB": 1.2,     # Rebounds
    "AST": 1.5,     # Assists
    "STL": 3.0,     # Steals
    "BLK": 3.0,     # Blocks
    "TOV": -1.0,    # Turnovers
}

# For now, your engine will primarily consume precomputed
# fantasy point projections, but having the weights here is useful.


# ---------- POSITION ELIGIBILITY HELPERS ----------

# Positions that qualify for each flexible slot.
# This will be used by the lineup optimizer.
SLOT_POSITION_RULES = {
    "PG": {"PG"},
    "SG": {"SG"},
    "G": {"PG", "SG"},
    "SF": {"SF"},
    "PF": {"PF"},
    "F": {"SF", "PF"},
    "C": {"C"},
    "UTIL": {"PG", "SG", "SF", "PF", "C"},
}

# You don’t need to touch this often.
# If Yahoo eligibility rules change, adjust SLOT_POSITION_RULES.
