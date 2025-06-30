from nba_api.stats.endpoints import LeagueGameLog
from pymongo import MongoClient
import time

def get_mongo_db(uri="mongodb+srv://madwalker231:x91EwbKtLj7b6Ai8@sportsinsight.kkyclry.mongodb.net/?retryWrites=true&w=majority&appName=sportsinsight", db_name="sportsinsight"):
    """
    Connect to MongoDB and return the database handle.
    """
    client = MongoClient(uri)
    return client[db_name]

def fetch_game_logs(season="2022-23"):
    """
    Fetch raw game logs for a given season and return
    a list of dicts (one per row).
    """
    endpoint = LeagueGameLog(season=season)
    df = endpoint.get_data_frames()[0]
    return df.to_dict("records")

def store_records(collection, records, unique_keys=None):
    """
    Insert records into MongoDB. If unique_keys is provided,
    uses upsert to avoid duplicates.
    """
    if not records:
        print("No records fetched.")
        return

    if unique_keys:
        for rec in records:
            query = {k: rec[k] for k in unique_keys}
            collection.update_one(query, {"$set": rec}, upsert=True)
        print(f"→ Upserted {len(records)} records (on keys {unique_keys}).")
    else:
        res = collection.insert_many(records)
        print(f"→ Inserted {len(res.inserted_ids)} records.")

def main():
    db        = get_mongo_db()  
    coll_name = "game_logs"
    coll      = db[coll_name]

    # Optional: ensure an index for your dedupe/upsert
    coll.create_index([("GAME_ID", 1)], unique=True)

    seasons = ["2022-23"]  # expand this list as you go
    for season in seasons:
        print(f"\nFetching season {season}...")
        recs = fetch_game_logs(season)
        store_records(coll, recs, unique_keys=["GAME_ID"])
        time.sleep(1)  # be kind to the API rate limits

if __name__ == "__main__":
    main()
