import os
from dotenv import load_dotenv
import pandas as pd
import joblib
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# MongoDB setup
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "")
client = MongoClient(MONGO_URI)
db = client["sportsinsight"]
collection = db["lebron_perf_features"]

def load_data(collection):
    data = list(collection.find({}))
    df = pd.DataFrame(data)
    # Convert date back if you need it for sorting, but we only need features here
    return df

def main():
    df = load_data(collection)
    # Select features and label
    X = df[['AVG_PTS_L5','AVG_REB_L5','AVG_AST_L5','OPP_DEF_RATING','MIN']]
    y = df['PTS']
    # Time-based split: train on first 80%, test on last 20%
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Use RandomForest instead of XGBoost
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"MAE:  {mean_absolute_error(y_test, preds):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.3f}")

    # Save the trained model
    joblib.dump(model, 'player_pts_model_rf.joblib')
    print("✅ Model trained and saved as player_pts_model_rf.joblib")

if __name__ == "__main__":
    main()