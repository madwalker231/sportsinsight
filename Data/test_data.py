import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog
from pymongo import MongoClient

# MongoDB setup
MONGO_URI = "mongodb+srv://madwalker231:x91EwbKtLj7b6Ai8@sportsinsight.kkyclry.mongodb.net/?retryWrites=true&w=majority&appName=sportsinsight"
client = MongoClient(MONGO_URI)
db = client["sportsinsight"]
collection = db["ad_perf_features"]

def fetch_player_logs(player_id, season):
    # 1) Pull the PlayerGameLog DataFrame
    df = PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]

    # 2) Select exactly the columns we need
    df = df[['Game_ID', 'GAME_DATE', 'PTS', 'REB', 'AST', 'MIN']]

    # 3) Normalize column names
    df = df.rename(columns={
        'GAME_DATE': 'Game_Date',
        'Game_ID':  'Game_ID',     # no change, but explicit
        'PTS':      'PTS',
        'REB':      'REB',
        'AST':      'AST',
        'MIN':      'MIN'
    })

    # 4) Convert to datetime and sort
    df['Game_Date'] = pd.to_datetime(df['Game_Date'])
    df = df.sort_values('Game_Date')

    return df

def fetch_team_defense_rating(game_id):
    # Placeholder: implement real lookup if you have a source
    return 110

def build_features(df_player):
    df = df_player.copy()

    # Rolling averages on last 5 games (shifted so they only use past data)
    df['AVG_PTS_L5'] = df['PTS'].rolling(5).mean().shift(1)
    df['AVG_REB_L5'] = df['REB'].rolling(5).mean().shift(1)
    df['AVG_AST_L5'] = df['AST'].rolling(5).mean().shift(1)

    # Add opponent defensive rating (static placeholder here)
    df['OPP_DEF_RATING'] = df['Game_ID'].apply(fetch_team_defense_rating)

    # Fill NaNs (first few games) with the column mean
    df.fillna(df.mean(), inplace=True)

    # Convert date back to string for storage in MongoDB
    df['Game_Date'] = df['Game_Date'].dt.strftime('%Y-%m-%d')

    return df.dropna()

def save_to_mongo(df, collection):
    collection.delete_many({})
    collection.insert_many(df.to_dict('records'))

if __name__ == "__main__":
    raw_df   = fetch_player_logs(203076, '2023-24')
    feature_df = build_features(raw_df)
    save_to_mongo(feature_df, collection)
    print("Dataset built and saved to MongoDB.")