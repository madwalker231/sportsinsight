import os
from dotenv import load_dotenv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymongo import MongoClient, UpdateOne

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

try:
    from src.utils.repro import set_all_seeds
    from src.utils.nba_cache import (
        fetch_gamelog_cached,
        team_def_rating_map_cached,
    )
except ImportError:
    # Fallback if you run the script directly (not as a module)
    import sys, os
    sys.path.append(os.path.abspath("."))  # project root
    from src.utils.repro import set_all_seeds
    from src.utils.nba_cache import (
        fetch_gamelog_cached,
        team_def_rating_map_cached,
    )

set_all_seeds(42)

PLAYER_ID   = 1629029
PLAYER_NAME = "Luka Doncic"

TRAIN_SEASONS = [f"{y}-{str((y+1) % 100).zfill(2)}" for y in range(2013, 2024)]  # 2013-14..2023-24
TEST_SEASON   = "2024-25"
SLEEP_SEC     = 0.7


load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME   = "sportsinsight"
PRED_COLL = "predictions_2024_25"

PARAM_GRID = {
    "n_estimators":    [100, 200, 400],
    "max_depth":       [None, 10, 16],
    "min_samples_leaf":[1, 2, 4],
    "min_samples_split":[2, 5],
    "max_features":    [1.0, "sqrt"],  # regression default=1.0; try sqrt too
}

FEATURE_COLS = [
    "PTS_L5","REB_L5","AST_L5","MIN_L5","FGA_L5","FTA_L5","FG3A_L5",
    "OPP_DEF_RATING","REST_DAYS","IS_HOME","IS_B2B",
    "PTS_EWMA3","PTS_MED5","PTSxMIN_L5",
]

def fetch_gamelog(player_id: int, season: str) -> pd.DataFrame:
    """Use cached fetch, then normalize columns and attach SEASON."""
    df = fetch_gamelog_cached(player_id=player_id, season=season, sleep_sec=SLEEP_SEC)
    if df is None or df.empty:
        print(f"{season}: no games")
        return pd.DataFrame()

    df = df.copy()

    # Normalize column names → UPPER_CASE_WITH_UNDERSCORES
    norm = {c: c.upper().replace(" ", "_") for c in df.columns}
    df.rename(columns=norm, inplace=True)

    # Attach SEASON if missing
    if "SEASON" not in df.columns:
        df["SEASON"] = season

    # Parse GAME_DATE, sort
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    else:
        raise KeyError("GAME_DATE not found after normalization.")

    # Keep a stable subset if present
    keep = ["GAME_ID","GAME_DATE","MATCHUP","TEAM_ID","MIN","PTS","REB","AST","FGA","FG3A","FTA","SEASON"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values("GAME_DATE").reset_index(drop=True)

    return df

def parse_opp_abbr(matchup: str):
    # "DAL vs BOS" or "DAL @ BOS" -> ("BOS", is_home 1/0)
    if not isinstance(matchup, str): return None, 0
    toks = matchup.replace(".", "").split()
    if len(toks) < 3: return None, 0
    opp = toks[-1].upper()
    is_home = 1 if ("vs" in toks or "VS" in toks) else 0
    return opp, is_home

def team_def_rating_map(season: str) -> dict:
    """Wrapper that uses cached LeagueDashTeamStats for DEF_RATING."""
    return team_def_rating_map_cached(season=season, sleep_sec=SLEEP_SEC)

def build_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """Compute past-only features across concatenated seasons, then drop rows missing history."""
    # --- Guards: ensure required columns exist after fetch step ---
    if "SEASON" not in df_all.columns:
        raise KeyError("SEASON column missing before feature engineering. Ensure fetch_gamelog adds it.")
    if "GAME_DATE" not in df_all.columns:
        raise KeyError("GAME_DATE column missing before feature engineering.")
    if "MATCHUP" not in df_all.columns:
        # If missing, create a harmless placeholder (treated as away)
        df_all["MATCHUP"] = "UNK @ UNK"
    # Opponent + home/away
    opp_home = df_all["MATCHUP"].apply(parse_opp_abbr)
    df_all["OPP_ABBR"] = opp_home.apply(lambda x: x[0])
    df_all["IS_HOME"]  = opp_home.apply(lambda x: x[1]).astype(int)

    # Opponent DEF_RATING per season (fallback to median)
    def_maps = {}
    seasons = sorted(df_all["SEASON"].unique())
    for s in seasons:
        try:
            time.sleep(SLEEP_SEC)
            def_maps[s] = team_def_rating_map(s)
        except Exception:
            def_maps[s] = {}
    df_all["OPP_DEF_RATING"] = 0.0
    for s in seasons:
        m = def_maps.get(s, {})
        median_val = pd.Series(m).median() if m else 110.0
        idx = df_all["SEASON"] == s
        df_all.loc[idx, "OPP_DEF_RATING"] = df_all.loc[idx, "OPP_ABBR"].map(m).fillna(median_val)

    # Rest days (per season)
    df_all["REST_DAYS"] = df_all.groupby("SEASON")["GAME_DATE"].diff().dt.days
    df_all["REST_DAYS"] = df_all["REST_DAYS"].fillna(3).clip(lower=0)
    df_all["IS_B2B"] = (df_all["REST_DAYS"] == 0).astype(int)

    # Rolling L5 (shifted) across full chronology
    df_all = df_all.sort_values("GAME_DATE").reset_index(drop=True)
    for col in ["PTS","REB","AST","MIN","FGA","FTA","FG3A"]:
        if col in df_all.columns:
            df_all[f"{col}_L5"] = df_all[col].rolling(5).mean().shift(1)

    # Light, non-leaky smoothers for the target
    df_all["PTS_EWMA3"] = df_all["PTS"].ewm(span=3, adjust=False).mean().shift(1)
    df_all["PTS_MED5"]  = df_all["PTS"].rolling(5).median().shift(1)

    # simple interaction used a lot by models
    if {"PTS_L5","MIN_L5"}.issubset(df_all.columns):
        df_all["PTSxMIN_L5"] = df_all["PTS_L5"] * df_all["MIN_L5"]

    # Season-to-date baseline (shifted)
    df_all["PTS_SEASON_TD"] = df_all.groupby("SEASON")["PTS"] \
                                    .apply(lambda s: s.expanding().mean().shift(1))

    # Keep rows with full feature availability
    need = set(FEATURE_COLS + ["PTS","GAME_ID","GAME_DATE","SEASON","PTS_SEASON_TD"])
    df_all = df_all[[c for c in df_all.columns if c in need]].dropna().reset_index(drop=True)
    return df_all

def expanding_window_splits(df_train: pd.DataFrame):
    seasons = sorted(df_train["SEASON"].unique())
    for i in range(1, len(seasons)):
        tr_idx = df_train["SEASON"].isin(seasons[:i]).values
        va_idx = (df_train["SEASON"] == seasons[i]).values
        yield np.where(tr_idx)[0], np.where(va_idx)[0]

def evaluate_and_report(y_true, y_pred, y_l5, y_season_td):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    med  = np.median(np.abs(y_true - y_pred))
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan

    mae_l5  = mean_absolute_error(y_true, y_l5)
    mae_std = mean_absolute_error(y_true, y_season_td)
    skill_l5  = 100.0 * (1.0 - mae / mae_l5) if mae_l5 > 0 else np.nan
    skill_std = 100.0 * (1.0 - mae / mae_std) if mae_std > 0 else np.nan

    print("\n=== 2024–25 Test Metrics (RandomForest) ===")
    print(f"MAE:        {mae:.3f}")
    print(f"RMSE:       {rmse:.3f}")
    print(f"Median AE:  {med:.3f}")
    print(f"R^2:        {r2:.3f}")
    print(f"MAE (L5):   {mae_l5:.3f}   → Skill vs L5: {skill_l5:.1f}%")
    print(f"MAE (STD):  {mae_std:.3f}  → Skill vs Season-to-date: {skill_std:.1f}%")
    return dict(MAE=mae, RMSE=rmse, MedAE=med, R2=r2,
                MAE_L5=mae_l5, MAE_STD=mae_std,
                Skill_L5=skill_l5, Skill_STD=skill_std)

def save_plots(pred_df: pd.DataFrame):
    # Pred vs Actual
    plt.figure(figsize=(10,5))
    dates = pd.to_datetime(pred_df["GAME_DATE"])
    plt.plot(dates, pred_df["PTS"], marker="o", label="Actual PTS")
    plt.plot(dates, pred_df["PRED_PTS"], marker="o", label="RandomForest")
    plt.plot(dates, pred_df["L5_BASELINE"], linestyle="--", label="L5 baseline")
    plt.xticks(rotation=45, ha="right")
    plt.title("Luka Doncic – Predicted vs Actual (2024–25)")
    plt.ylabel("Points")
    plt.tight_layout()
    plt.legend()
    plt.savefig("doncic_pred_vs_actual_2024_25.png", dpi=140)

    # Residuals histogram
    plt.figure(figsize=(6,4))
    resid = pred_df["PTS"] - pred_df["PRED_PTS"]
    plt.hist(resid, bins=12)
    plt.title("Residuals (Actual − Predicted) – 2024–25")
    plt.xlabel("Points")
    plt.tight_layout()
    plt.savefig("doncic_residuals_hist_2024_25.png", dpi=140)

def save_predictions_to_mongo(pred_df: pd.DataFrame):
    try:
        client = MongoClient(MONGO_URI)
        coll = client[DB_NAME][PRED_COLL]
        coll.create_index([("PLAYER_ID",1), ("GAME_ID",1)], unique=True)
        ops = []
        for _, r in pred_df.iterrows():
            doc = {
                "PLAYER_ID": PLAYER_ID,
                "PLAYER_NAME": PLAYER_NAME,
                "GAME_ID": str(r["GAME_ID"]),
                "GAME_DATE": r["GAME_DATE"],
                "SEASON": r["SEASON"],
                "PRED_TARGET": "PTS",
                "PRED_PTS": float(r["PRED_PTS"]),
                "ACTUAL_PTS": float(r["PTS"]),
                "BASELINE_L5": float(r["L5_BASELINE"]),
                "BASELINE_SEASON_TD": float(r["STD_BASELINE"])
            }
            ops.append(UpdateOne({"PLAYER_ID": PLAYER_ID, "GAME_ID": doc["GAME_ID"]},
                                 {"$set": doc}, upsert=True))
        if ops:
            res = coll.bulk_write(ops, ordered=False)
            print(f"Mongo upsert complete: matched={res.matched_count}, upserts={len(res.upserted_ids) if res.upserted_ids else 0}")
    except Exception as e:
        print(f"(Mongo warning) Could not save predictions: {e}")

if __name__ == "__main__":
    # 1) Fetch raw data (expect 'no games' before NBA debut 2018–19)
    all_seasons = TRAIN_SEASONS + [TEST_SEASON]
    dfs = []
    for s in all_seasons:
        try:
            time.sleep(SLEEP_SEC)
            df = fetch_gamelog(PLAYER_ID, s)
            if not df.empty:
                dfs.append(df)
            else:
                print(f"{s}: no games")
        except Exception as e:
            print(f"Fetch error {s}: {e}")
    if not dfs:
        raise SystemExit("No data fetched.")
    raw = pd.concat(dfs, ignore_index=True).sort_values("GAME_DATE").reset_index(drop=True)

    # 2) Features (past-only)
    feats = build_features(raw.copy())

    # 3) Split
    train = feats[feats["SEASON"].isin(TRAIN_SEASONS)].copy()
    test  = feats[feats["SEASON"] == TEST_SEASON].copy()
    if train.empty or test.empty:
        raise SystemExit("Train or test set is empty after feature building.")
    X_train, y_train = train[FEATURE_COLS].values, train["PTS"].values
    X_test,  y_test  = test[FEATURE_COLS].values,  test["PTS"].values

    # Baselines on test
    l5_base  = test["PTS_L5"].values
    std_base = test["PTS_SEASON_TD"].values

    # 4) Expanding-window CV for RandomForest (no process parallelism to avoid Windows loky bug)
    cv_splits = list(expanding_window_splits(train))
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,   # estimator uses threads internally
    )
    gs = GridSearchCV(
        rf,
        PARAM_GRID,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=1     # ← run sequentially to avoid loky on Windows
    )
    gs.fit(X_train, y_train)

    best_mae = -gs.best_score_
    print("\n=== CV (2013–24) – RandomForest ===")
    print(f"Best params: {gs.best_params_}")
    print(f"CV MAE:      {best_mae:.3f}")

    # 5) Final fit + predict 2024–25
    best_model = gs.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    last_tr_idx, last_va_idx = list(expanding_window_splits(train))[-1]
    X_val = train.iloc[last_va_idx][FEATURE_COLS].values
    y_val = train.iloc[last_va_idx]["PTS"].values

    y_val_hat = best_model.predict(X_val)

    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    corr = safe_corr(y_val_hat, y_val)
    a_cal, b_cal = 0.0, 1.0  # identity
    if np.isfinite(corr) and corr > 0.20:
        A = np.column_stack([np.ones_like(y_val_hat), y_val_hat])
        try:
            a_tmp, b_tmp = np.linalg.lstsq(A, y_val, rcond=None)[0]
        except np.linalg.LinAlgError:
            a_tmp, b_tmp = np.nan, np.nan
        if np.isfinite(b_tmp) and 0.25 <= b_tmp <= 1.75:
            a_cal, b_cal = float(a_tmp), float(b_tmp)
    print(f"Calibration used: y ≈ {a_cal:.3f} + {b_cal:.3f}·ŷ, corr={np.nan if not np.isfinite(corr) else corr:.3f}")

    # ---------- Choose stacked weights across folds ----------
    def choose_stack_weights(train_df, feat_cols, cv_splits, base_cols):
    # grid over simplex w0+w1+...=1, step 0.05
        grid = np.arange(0.0, 1.01, 0.05)
        best_w, best_mae = None, 9e9

        for w0 in grid:               # model weight
            for w1 in grid:           # MED5 weight
                w2 = 1.0 - w0 - w1    # STD weight
                if w2 < -1e-9: 
                    continue
                fold_mae = []
                for tr_idx, va_idx in cv_splits:
                    X_tr = train_df.iloc[tr_idx][feat_cols].values
                    y_tr = train_df.iloc[tr_idx]["PTS"].values
                    X_va = train_df.iloc[va_idx][feat_cols].values
                    y_va = train_df.iloc[va_idx]["PTS"].values

                    m = clone(best_model)
                    m.fit(X_tr, y_tr)
                    y_hat = m.predict(X_va)

                    # quick fold-local calibration (same rules)
                    a_, b_ = 0.0, 1.0
                    if np.std(y_hat) > 1e-8 and np.std(y_va) > 1e-8:
                        c = float(np.corrcoef(y_hat, y_va)[0, 1])
                    else:
                        c = np.nan
                    if np.isfinite(c) and c > 0.20:
                        A = np.column_stack([np.ones_like(y_hat), y_hat])
                        try:
                            a__, b__ = np.linalg.lstsq(A, y_va, rcond=None)[0]
                        except np.linalg.LinAlgError:
                            a__, b__ = np.nan, np.nan
                        if np.isfinite(b__) and 0.25 <= b__ <= 1.75:
                            a_, b_ = float(a__), float(b__)
                    y_hat_cal = a_ + b_ * y_hat

                    med5 = train_df.iloc[va_idx][base_cols["MED5"]].values
                    std  = train_df.iloc[va_idx][base_cols["STD"]].values

                    y_blend = w0*y_hat_cal + w1*med5 + w2*std
                    fold_mae.append(mean_absolute_error(y_va, y_blend))

                m_mae = np.mean(fold_mae)
                if m_mae < best_mae:
                    best_mae, best_w = m_mae, (w0, w1, w2)
        return best_w

    base_cols = {"MED5": "PTS_MED5", "STD": "PTS_SEASON_TD"}
    w_model, w_med5, w_std = choose_stack_weights(train, FEATURE_COLS, cv_splits, base_cols)
    print(f"Stacked weights (validation across folds): model={w_model:.2f}, MED5={w_med5:.2f}, STD={w_std:.2f}")

    y_pred_raw = best_model.predict(X_test)
    y_pred_cal = a_cal + b_cal * y_pred_raw

    y_pred = w_model * y_pred_cal \
       + w_med5 * test["PTS_MED5"].values \
       + w_std  * test["PTS_SEASON_TD"].values

    # keep in a sane range for points
    y_pred = np.clip(y_pred, 0, 80)

    # 6) Evaluate
    metrics = evaluate_and_report(y_test, y_pred, l5_base, std_base)

    # 7) Save artifacts
    out = test[["GAME_ID","GAME_DATE","SEASON","PTS"]].copy()
    out["PRED_PTS"]     = y_pred
    out["L5_BASELINE"]  = l5_base
    out["STD_BASELINE"] = std_base
    out.to_csv("doncic_randomforest_2024_25_predictions.csv", index=False)
    print("Saved CSV: doncic_randomforest_2024_25_predictions.csv")

    save_plots(out)
    print("Saved plots: doncic_pred_vs_actual_2024_25.png, doncic_residuals_hist_2024_25.png")

    save_predictions_to_mongo(out)
    print("Done.")

