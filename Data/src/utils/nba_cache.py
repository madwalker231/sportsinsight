import os, time, pandas as pd
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats
from nba_api.stats.static import teams as static_teams

def _cache_path(player_id: int, season: str, root="data/cache/nba_api"):
    os.makedirs(os.path.join(root, str(player_id)), exist_ok=True)
    return os.path.join(root, str(player_id), f"{season}_gamelog.csv")

def fetch_gamelog_cached(player_id: int, season: str, sleep_sec: float = 0.7) -> pd.DataFrame:
    """Read from CSV cache if available; otherwise call nba_api and cache."""
    path = _cache_path(player_id, season)
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["GAME_DATE"])
    # network call (throttled)
    time.sleep(sleep_sec)
    df = PlayerGameLog(player_id=player_id, season=season,
                       season_type_all_star="Regular Season").get_data_frames()[0]
    if df.empty:
        return df
    df.columns = [c.upper() for c in df.columns]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df

def team_def_rating_map_cached(season: str, root="data/cache/nba_api", sleep_sec: float = 0.7) -> dict:
    """Cache opponent DEF_RATING per season."""
    path = os.path.join(root, f"team_def_{season}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        time.sleep(sleep_sec)
        df = LeagueDashTeamStats(season=season, per_mode_detailed="PerGame",
                                 measure_type_detailed_defense="Advanced").get_data_frames()[0]
        df.to_csv(path, index=False)
    from nba_api.stats.static import teams as static_teams
    teams = static_teams.get_teams()
    id_to_abbr = {t["id"]: t["abbreviation"] for t in teams}
    df.columns = [c.upper() for c in df.columns]
    if "TEAM_ID" not in df.columns or "DEF_RATING" not in df.columns:
        return {}
    df["TEAM_ABBR"] = df["TEAM_ID"].map(id_to_abbr)
    return dict(zip(df["TEAM_ABBR"], df["DEF_RATING"]))

# Replace old fetcher with:
# from src.utils.nba_cache import fetch_gamelog_cached as fetch_gamelog
# from src.utils.nba_cache import team_def_rating_map_cached as team_def_rating_map
