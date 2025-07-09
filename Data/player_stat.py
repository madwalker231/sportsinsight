from nba_api.stats.endpoints import LeagueDashPlayerStats
from pymongo import MongoClient
import time

def get_mongo_db(uri="mongodb+srv://madwalker231:x91EwbKtLj7b6Ai8@sportsinsight.kkyclry.mongodb.net/?retryWrites=true&w=majority&appName=sportsinsight", db_name="sportsinsight"):
    """
    Connect to MongoDB and return the database handle.
    """
    client = MongoClient(uri)
    return client[db_name]

def fetch_player_log(season="2023-24"):
    """
    Fetch raw player stats logs for given season and return
    a list of dicts (one per row)
    """
    endpoint = LeagueDashPlayerStats(season=season)
    df = endpoint.get_data_frames()[0]
    return df.to_dict("records")

def store_player_records(collection, records, unique_keys=None):
    """
    Insert records into MongoDB.
    """
    if not records:
        print("No records fetched.")
        return
    
    if unique_keys:
        for rec in records:
            query = {k: rec[k] for k in unique_keys}
            collection.update_one(query, {"$set": rec}, upsert=True)
        print(f"-> Upserted {len(records)} records (on keys {unique_keys}).")
    else:
        res = collection.insert_many(records)
        print(f"-> Inserted {len(res.inserted_ids)} records.")

def main():
    db = get_mongo_db()
    coll_name = "player_stats"
    coll = db[coll_name]

    coll.create_index([("PLAYER_ID", 1)], unique=True)

    seasons = ["2023-24"]
    for season in seasons:
        print(f"\nFetching season {season}...")
        recs = fetch_player_log(season)
        store_player_records(coll, recs, unique_keys=["PLAYER_ID"])
        time.sleep(1)

if __name__ == "__main__":
    main()
