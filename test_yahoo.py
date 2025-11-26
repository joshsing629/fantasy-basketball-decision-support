from src.data.yahoo_api import list_teams

if __name__ == "__main__":
    teams = list_teams()
    print("Teams in league:")
    for key, info in teams.items():
        print(f"{key}: {info.get('name')}")
