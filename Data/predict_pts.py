import joblib
import pandas as pd
from pymongo import MongoClient

from memory_profiler import profile

@profile
def main():
    # 1) Load your trained model
    model = joblib.load('player_pts_model_rf.joblib')

    # 2) Connect to Mongo and pull the latest feature row
    client = MongoClient("mongodb+srv://madwalker231:x91EwbKtLj7b6Ai8@sportsinsight.kkyclry.mongodb.net/?retryWrites=true&w=majority&appName=sportsinsight")
    db = client["sportsinsight"]
    collection = db["lebron_perf_features"]

    # Load into a DataFrame
    df = pd.DataFrame(list(collection.find({})))

    # Optional: sort by date and pick the last game
    df['Game_Date'] = pd.to_datetime(df['Game_Date'])
    df = df.sort_values('Game_Date')
    latest = df.iloc[-1]

    # 3) Extract the feature vector
    features = latest[['AVG_PTS_L5','AVG_REB_L5','AVG_AST_L5','OPP_DEF_RATING','MIN']].to_list()
    print("Features for game on", latest['Game_Date'].date(), ":", features)

    # 4) Predict
    pred = model.predict([features])[0]
    print(f"Model prediction for next-game points: {pred:.1f}")

    # 5) (Optional) Compare to actual
    actual = latest['PTS']
    print(f"Actual points in that game: {actual}")

if __name__ == "__main__":
    main()