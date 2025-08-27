import time
from typing import List, Tuple
import pandas as pd
from pymongo import MongoClient, UpdateOne
from nba_api.stats.endpoints import PlayerGameLog

PLAYER_ID   = 203999            # Nikola Jokic
PLAYER_NAME = "Nikola Jokic"
SEASONS     = [f"{y}-{str((y+1) % 100).zfill(2)}" for y in range(2013, 2024)]  # '2013-14' ... '2023-24'
SLEEP_SEC   = 0.7               # be kind to the API

MONGO_URI = "mongodb+srv://madwalker231:x91EwbKtLj7b6Ai8@sportsinsight.kkyclry.mongodb.net/?retryWrites=true&w=majority&appName=sportsinsight"

DB_NAME     = "sportsinsight"
COLLECTION  = "raw_jokic_games_2013_2024"

# Keep a stable subset of columns; we will UPPERCASE once.
KEEP_COLS = [
    "GAME_ID","GAME_DATE","MATCHUP","TEAM_ID","TEAM_ABBREVIATION","WL",
    "MIN","PTS","REB","AST","STL","BLK","TOV",
    "FGA","FGM","FG_PCT","FG3A","FG3M","FG3_PCT","FTA","FTM","FT_PCT",
    "PLUS_MINUS","SEASON_ID"
]

def to_minutes_decimal(min_val) -> float:
    """Convert 'MM:SS' to decimal minutes; pass through numeric; NaN on error."""
    if pd.isna(min_val):
        return float("nan")
    if isinstance(min_val, (int, float)):
        return float(min_val)
    try:
        s = str(min_val)
        if ":" in s:
            mm, ss = s.split(":")
            return int(mm) + int(ss) / 60.0
        return float(s)
    except Exception:
        return float("nan")

def fetch_player_season(player_id: int, season: str) -> pd.DataFrame:
    """Fetch one season of regular-season logs; uppercase cols; keep a stable subset."""
    resp = PlayerGameLog(player_id=player_id, season=season, season_type_all_star="Regular Season")
    df = resp.get_data_frames()[0]
    df.columns = [c.upper() for c in df.columns]

    keep = [c for c in KEEP_COLS if c in df.columns]
    if not keep:
        return pd.DataFrame()

    df = df[keep].copy()
    if "GAME_DATE" not in df.columns or df.empty:
        return pd.DataFrame()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Convenience decimal minutes
    if "MIN" in df.columns:
        df["MIN_DEC"] = df["MIN"].apply(to_minutes_decimal)

    return df

def upsert_games(coll, docs: List[dict]) -> Tuple[int, int]:
    """Bulk upsert by (PLAYER_ID, GAME_ID). Returns (matched_count, upserted_count)."""
    if not docs:
        return (0, 0)
    ops = []
    for d in docs:
        d["PLAYER_ID"] = PLAYER_ID
        d["PLAYER_NAME"] = PLAYER_NAME
        # Ensure types / formatting
        if "GAME_ID" in d:
            d["GAME_ID"] = str(d["GAME_ID"])
        if "GAME_DATE" in d and not isinstance(d["GAME_DATE"], str):
            d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"]).strftime("%Y-%m-%d")
        # Helpful explicit season string (in addition to SEASON_ID)
        if "SEASON" not in d:
            d["SEASON"] = d.get("SEASON_ID", "")

        key = {"PLAYER_ID": PLAYER_ID, "GAME_ID": d["GAME_ID"]}
        ops.append(UpdateOne(key, {"$set": d}, upsert=True))

    res = coll.bulk_write(ops, ordered=False)
    matched = res.matched_count
    upserted = len(res.upserted_ids) if res.upserted_ids else 0
    return (matched, upserted)

if __name__ == "__main__":
    # Connect to MongoDB (hard-coded URI)
    print("Connecting to MongoDB…")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    coll = db[COLLECTION]

    # Helpful indexes (safe to run repeatedly)
    try:
        coll.create_index([("PLAYER_ID", 1), ("GAME_DATE", 1)])
        coll.create_index([("PLAYER_ID", 1), ("GAME_ID", 1)], unique=True)
    except Exception as e:
        print(f"Index creation warning: {e}")

    total_upserted = 0
    print(f"Ingesting {PLAYER_NAME} seasons {SEASONS[0]} .. {SEASONS[-1]}")

    for season in SEASONS:
        try:
            time.sleep(SLEEP_SEC)  # throttle to avoid request issues
            df = fetch_player_season(PLAYER_ID, season)
            if df.empty:
                print(f"  {season}: no games (pre-debut/inactive).")
                continue

            # Attach explicit season string
            df["SEASON"] = season

            matched, upserted = upsert_games(coll, df.to_dict("records"))
            total_upserted += upserted
            print(f"  {season}: rows={len(df)}, upserted={upserted}, matched={matched}")

        except Exception as e:
            print(f"  ERROR {season}: {e}")

    print(f"\nDone. Upserted {total_upserted} new documents into {DB_NAME}.{COLLECTION}")
