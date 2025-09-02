import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "")

def load_and_prepare():
    # 1) Load features from MongoDB
    client = MongoClient(MONGO_URI)
    df = pd.DataFrame(list(client["sportsinsight"]["ad_perf_features"].find({})))
    # 2) Convert types and sort
    df['Game_Date'] = pd.to_datetime(df['Game_Date'])
    df.sort_values('Game_Date', inplace=True)
    # 3) Define features & target
    X = df[['AVG_PTS_L5', 'AVG_REB_L5', 'AVG_AST_L5', 'OPP_DEF_RATING', 'MIN']].astype(float).values
    y = df['PTS'].astype(float).values
    return X, y

def split_and_scale(X, y):
    # 4) Chronological train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    # 5) Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def benchmark_models(X_train, X_test, y_train, y_test):
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'KNeighbors': KNeighborsRegressor(),
        'MLP': MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500, random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        results.append({'Model': name, 'MAE': mae, 'RMSE': rmse})
    res_df = pd.DataFrame(results).sort_values('MAE')
    print("=== Benchmark Results ===")
    print(res_df)
    return res_df

def hyperparameter_tuning(X, y):
    print("\n=== Hyperparameter Tuning for Top Models ===")
    tscv = TimeSeriesSplit(n_splits=5)
    tuned_results = []
    # Example: tune RandomForest and XGBoost
    grid_searches = {
        'RandomForest': (RandomForestRegressor(random_state=42),
                         {'n_estimators': [50, 100, 200]})
    }
    for name, (model, params) in grid_searches.items():
        gs = GridSearchCV(model, params, cv=tscv, scoring='neg_mean_absolute_error')
        gs.fit(X, y)
        tuned_results.append({
            'Model': name,
            'Best Params': gs.best_params_,
            'CV MAE': -gs.best_score_
        })
        print(f"{name}: Best Params={gs.best_params_}, CV MAE={-gs.best_score_:.3f}")
    return pd.DataFrame(tuned_results)

def plot_results(res_df):
    plt.figure(figsize=(10, 5))
    plt.bar(res_df['Model'], res_df['MAE'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('MAE')
    plt.title('Model MAE Comparison')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    res_df = benchmark_models(X_train, X_test, y_train, y_test)
    plot_results(res_df)
    tune_df = hyperparameter_tuning(np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)))
    print("\n=== Tuning Summary ===")
    print(tune_df) 
