import os
import time
import pandas as pd
from pymongo import MongoClient
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats
from nba_api.stats.static import teams as static_teams

# -----------------------------
# Config
# -----------------------------
SEASONS = [f"{y}-{str((y+1) % 100).zfill(2)}" for y in range(2013, 2024)]  # '2013-14' ... '2023-24'
PLAYERS = {
    "Nikola Jokic":            203999,
    "Luka Doncic":             1629029,
    "Giannis Antetokounmpo":   203507,
    "Shai Gilgeous-Alexander": 1628983,
    "Joel Embiid":             203954,
    "Jayson Tatum":            1628369,
}
SLEEP_SEC = 0.7  # throttle API calls a bit

# MongoDB
MONGO_URI = "mongodb+srv://madwalker231:x91EwbKtLj7b6Ai8@sportsinsight.kkyclry.mongodb.net/?retryWrites=true&w=majority&appName=sportsinsight"
DB_NAME   = "sportsinsight"
COLL_NAME = "features_player_games_2013_2024"

# -----------------------------
# Helpers
# -----------------------------
def parse_opponent_abbr(matchup: str):
    """'DEN vs BOS' or 'DEN @ BOS' -> ('BOS', is_home:int)"""
    if not isinstance(matchup, str):
        return None, 0
    toks = matchup.replace(".", "").split()
    if len(toks) < 3:
        return None, 0
    opp = toks[-1].upper()
    is_home = 1 if ("vs" in toks or "VS" in toks) else 0
    return opp, is_home

def get_team_def_map(season: str):
    """Return dict: TEAM_ABBR -> DEF_RATING for the season."""
    teams = static_teams.get_teams()
    id_to_abbr = {t["id"]: t["abbreviation"] for t in teams}
    df = LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    if "TEAM_ID" not in df.columns or "DEF_RATING" not in df.columns:
        return {}
    df["TEAM_ABBR"] = df["TEAM_ID"].map(id_to_abbr)
    return dict(zip(df["TEAM_ABBR"], df["DEF_RATING"]))

def fetch_player_logs(player_id: int, season: str) -> pd.DataFrame:
    """Fetch per-game logs; keep essential columns and sort chronologically."""
    df = PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season"
    ).get_data_frames()[0]
    # Normalize column names used later
    # (nba_api returns mixed case like 'Game_ID', 'GAME_DATE')
    keep = ["Game_ID","GAME_DATE","MATCHUP","TEAM_ID","MIN","PTS","REB","AST","STL","BLK","TOV","FGA","FGM","FG3A","FG3M","FTA","FTM","PLUS_MINUS","WL"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["Game_Date"] = pd.to_datetime(df["Game_Date"])
    df = df.sort_values("Game_Date").reset_index(drop=True)
    return df

def build_features(df_player: pd.DataFrame, def_map: dict, season: str, player_id: int, player_name: str) -> pd.DataFrame:
    """Past-only features: L5 rolling means (shifted), rest days, home/away, opponent DEF_RATING."""
    if df_player.empty:
        return df_player

    # Opponent & home/away flags from MATCHUP
    opp_home = df_player["MATCHUP"].apply(parse_opponent_abbr)
    df_player["OPP_ABBR"] = opp_home.apply(lambda x: x[0])
    df_player["IS_HOME"]  = opp_home.apply(lambda x: x[1]).astype(int)

    # Opponent defensive rating (fallback to league median)
    league_median_def = pd.Series(def_map).median() if def_map else 110.0
    df_player["OPP_DEF_RATING"] = df_player["OPP_ABBR"].map(def_map).fillna(league_median_def)

    # Rest days since prior game (past-only)
    df_player["REST_DAYS"] = df_player["Game_Date"].diff().dt.days.fillna(3).clip(lower=0)

    # Rolling L5 (shift by 1 to avoid peeking)
    for col in ["PTS","REB","AST","MIN","FGA","FTA","FG3A"]:
        if col in df_player.columns:
            df_player[f"{col}_L5"] = df_player[col].rolling(5).mean().shift(1)

    # Select feature/target columns and drop rows without enough history
    cols = ["Game_ID","Game_Date","MATCHUP","TEAM_ID","OPP_ABBR","IS_HOME","OPP_DEF_RATING","REST_DAYS",
            "PTS_L5","REB_L5","AST_L5","MIN_L5","FGA_L5","FTA_L5","FG3A_L5",
            "PTS","REB","AST","MIN"]
    cols = [c for c in cols if c in df_player.columns]
    out = df_player[cols].dropna().reset_index(drop=True)

    # Add metadata
    out["PLAYER_ID"] = player_id
    out["PLAYER_NAME"] = player_name
    out["SEASON"] = season

    # Convert date to ISO for Mongo
    out["Game_Date"] = out["Game_Date"].dt.strftime("%Y-%m-%d")
    return out

def save_to_mongo(df: pd.DataFrame, collection, player_id: int, season: str):
    """Replace existing docs for this player+season, then insert fresh batch."""
    if df.empty:
        return 0
    collection.delete_many({"PLAYER_ID": player_id, "SEASON": season})
    collection.insert_many(df.to_dict("records"))
    return len(df)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    coll = db[COLL_NAME]

    # Optional helpful indexes (safe to re-run)
    try:
        coll.create_index([("PLAYER_ID", 1), ("SEASON", 1), ("Game_Date", 1)])
        coll.create_index([("PLAYER_ID", 1), ("Game_ID", 1)])
    except Exception as e:
        print(f"Index creation warning: {e}")

    total_inserted = 0

    for season in SEASONS:
        print(f"\n=== Season {season} ===")
        def_map = get_team_def_map(season)

        for player_name, pid in PLAYERS.items():
            try:
                time.sleep(SLEEP_SEC)  # be kind to the API
                raw_df = fetch_player_logs(pid, season)
                if raw_df.empty:
                    print(f"  {player_name}: no games.")
                    continue

                feat_df = build_features(raw_df, def_map, season, pid, player_name)
                n = save_to_mongo(feat_df, coll, pid, season)
                total_inserted += n
                print(f"  {player_name}: inserted {n} feature rows.")
            except Exception as e:
                print(f"  ERROR {player_name} ({season}): {e}")

    print(f"\nDone. Inserted {total_inserted} documents into {DB_NAME}.{COLL_NAME}")