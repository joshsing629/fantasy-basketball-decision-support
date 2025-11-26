from src.core.config import get_league_config, get_simulation_config

lc = get_league_config()
sc = get_simulation_config()

print("Roster slots:", lc.roster_slots)
print("Bench slots:", lc.bench_slots)
print("IL slots:", lc.il_slots)
print("Max active / day:", lc.max_active_per_day)
print("Default add threshold:", sc.default_add_value_threshold)
