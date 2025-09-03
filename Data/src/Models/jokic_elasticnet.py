import os
from dotenv import load_dotenv

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pymongo import MongoClient, UpdateOne

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

PLAYER_ID   = 203999
PLAYER_NAME = "Nikola Jokic"
TRAIN_SEASONS = [f"{y}-{str((y+1) % 100).zfill(2)}" for y in range(2013, 2024)]  # 2013-14..2023-24
TEST_SEASON   = "2024-25"  # final hold-out
SLEEP_SEC     = 0.7

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME   = "sportsinsight"
PRED_COLL = "predictions_2024_25"

# modest, pre-declared HP grid (no optimization theater)
PARAM_GRID = [{
  "model": [ElasticNet(max_iter=50000, tol=1e-5, random_state=42)],
  "model__alpha":    [0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
  "model__l1_ratio": [0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
}]

FEATURE_COLS = [
    "PTS_L5","REB_L5","AST_L5","MIN_L5","FGA_L5","FTA_L5","FG3A_L5",
    "OPP_DEF_RATING","REST_DAYS","IS_HOME"
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
    # "DEN vs BOS" or "DEN @ BOS" -> ("BOS", is_home 1/0)
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
    if "SEASON" not in df_all.columns:
        raise KeyError("SEASON column missing before feature engineering. Ensure fetch_gamelog adds it.")
    if "GAME_DATE" not in df_all.columns:
        raise KeyError("GAME_DATE column missing before feature engineering.")
    if "MATCHUP" not in df_all.columns:
        # If missing, create a harmless placeholder (treated as away)
        df_all["MATCHUP"] = "UNK @ UNK"
    # Opponent + home flag per row
    opp_home = df_all["MATCHUP"].apply(parse_opp_abbr)
    df_all["OPP_ABBR"] = opp_home.apply(lambda x: x[0])
    df_all["IS_HOME"]  = opp_home.apply(lambda x: x[1]).astype(int)

    df_all = df_all.sort_values("GAME_DATE").reset_index(drop=True)

    df_all["TRAVEL"] = (
        df_all.groupby("SEASON")["IS_HOME"].diff().fillna(0).ne(0).astype(int)
    )

    # Opponent DEF_RATING by season (fallback to median)
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

    # REST_DAYS (past-only)
    df_all["REST_DAYS"] = df_all.groupby("SEASON")["GAME_DATE"].diff().dt.days
    # For season openers, use days since previous season game if available; otherwise default to 3
    df_all["REST_DAYS"] = df_all["REST_DAYS"].fillna(3).clip(lower=0)

    df_all["IS_B2B"] = (df_all["REST_DAYS"] == 0).astype(int)

    # Rolling L5 means (shifted) across full chronology
    df_all = df_all.sort_values("GAME_DATE").reset_index(drop=True)
    stats_for_rolling = ["PTS", "REB", "AST", "MIN", "FGA", "FTA", "FG3A"]
    windows = [3, 5]

    for stat in stats_for_rolling:
        if stat in df_all.columns:
            for w in windows:
                df_all[f"{stat}_L{w}"] = df_all[stat].rolling(window=w).mean().shift(1)
    
    if {"PTS_L5", "MIN_L5"}.issubset(df_all.columns):
        df_all["PTSxMIN_L5"] = df_all["PTS_L5"] * df_all["MIN_L5"]

    # Season-to-date average baseline (shifted, per season)
    df_all["PTS_SEASON_TD"] = df_all.groupby("SEASON")["PTS"] \
                                    .apply(lambda s: s.expanding().mean().shift(1))

    # Keep rows with full feature availability
    keep_extra = ["PTS_SEASON_TD"]                    # <-- keep the baseline
    need = set(FEATURE_COLS + ["PTS","GAME_ID","GAME_DATE","SEASON"] + keep_extra)
    df_all = df_all[[c for c in df_all.columns if c in need]].dropna().reset_index(drop=True)
    return df_all

def expanding_window_splits(df_train: pd.DataFrame):
    """Yield (train_idx, val_idx) pairs for expanding-window CV across TRAIN_SEASONS."""
    seasons = sorted(df_train["SEASON"].unique())
    for i in range(1, len(seasons)):
        train_seasons = set(seasons[:i])
        val_season    = seasons[i]
        tr_idx = df_train["SEASON"].isin(train_seasons).values
        va_idx = (df_train["SEASON"] == val_season).values
        yield np.where(tr_idx)[0], np.where(va_idx)[0]

def evaluate_and_report(y_true, y_pred, y_l5, y_season_td):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    med  = np.median(np.abs(y_true - y_pred))
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan

    mae_l5  = mean_absolute_error(y_true, y_l5)
    mae_std = mean_absolute_error(y_true, y_season_td)
    skill_l5 = 100.0 * (1.0 - mae / mae_l5) if mae_l5 > 0 else np.nan
    skill_std = 100.0 * (1.0 - mae / mae_std) if mae_std > 0 else np.nan

    print("\n=== 2024–25 Test Metrics (ElasticNet) ===")
    print(f"MAE:        {mae:.3f}")
    print(f"RMSE:       {rmse:.3f}")
    print(f"Median AE:  {med:.3f}")
    print(f"R^2:        {r2:.3f}")
    print(f"MAE (L5):   {mae_l5:.3f}   -> Skill vs L5: {skill_l5:.1f}%")
    print(f"MAE (STD):  {mae_std:.3f}  -> Skill vs Season-to-date: {skill_std:.1f}%")
    return dict(MAE=mae, RMSE=rmse, MedAE=med, R2=r2, MAE_L5=mae_l5, MAE_STD=mae_std,
                Skill_L5=skill_l5, Skill_STD=skill_std)

def save_plots(pred_df: pd.DataFrame):
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)

    # Pred vs Actual
    plt.figure(figsize=(10,5))
    plt.plot(pd.to_datetime(pred_df["GAME_DATE"]), pred_df["PTS"], marker="o", label="Actual PTS")
    plt.plot(pd.to_datetime(pred_df["GAME_DATE"]), pred_df["PRED_PTS"], marker="o", label="ElasticNet (calibrated)")
    plt.plot(pd.to_datetime(pred_df["GAME_DATE"]), pred_df["L5_BASELINE"], linestyle="--", label="L5 baseline")
    plt.xticks(rotation=45, ha="right")
    plt.title("Nikola Jokic – Predicted vs Actual (2024–25)")
    plt.ylabel("Points")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir / f"{PLAYER_NAME.replace(' ', '_')}_pred_vs_actual_2024_25.png", dpi=140)

    # Residuals histogram
    plt.figure(figsize=(6,4))
    resid = pred_df["PTS"] - pred_df["PRED_PTS"]
    plt.hist(resid, bins=12)
    plt.title("Residuals (Actual − Predicted) – 2024–25")
    plt.xlabel("Points")
    plt.tight_layout()
    plt.savefig(output_dir / f"{PLAYER_NAME.replace(' ', '_')}_pred_vs_actual_2024_25.png", dpi=140)

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

def show_coefs(pipe, feat_cols):
    lr = pipe.named_steps["model"]
    w = lr.coef_
    ranked = sorted(zip(np.abs(w), w, feat_cols), reverse=True)
    print("\nTop |w| coefficients:")
    for _, val, name in ranked[:12]:
        print(f"{name:>14s}: {val:+.3f}")


if __name__ == "__main__":
    # 1) Fetch raw data
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
            print(f"Network error fetching {s}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {s}: {e}")
    if not dfs:
        raise SystemExit("No data fetched.")
    raw = pd.concat(dfs, ignore_index=True).sort_values("GAME_DATE").reset_index(drop=True)

    # 2) Build features across all seasons (past-only via shift)
    feats = build_features(raw.copy())

    # 3) Split into train (<=2023–24) and test (2024–25)
    train = feats[feats["SEASON"].isin(TRAIN_SEASONS)].copy()
    test  = feats[feats["SEASON"] == TEST_SEASON].copy()
    feat_cols = [c for c in FEATURE_COLS if c in train.columns and c in test.columns]
    if not feat_cols:
        raise SystemExit("No usable features after intersection. Check feature engineering.")
    if train.empty or test.empty:
        raise SystemExit("Train or test set is empty after feature building.")

    X_train = train[feat_cols].values
    y_train = train["PTS"].values

    X_test  = test[feat_cols].values
    y_test  = test["PTS"].values

    # Baselines for test
    l5_base  = test["PTS_L5"].values
    std_base = test["PTS_SEASON_TD"].values

    # 4) Expanding-window CV (inside 2013–24) for ElasticNet
    cv_splits = list(expanding_window_splits(train))
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("model", ElasticNet(max_iter=50000, tol=1e-5, random_state=42)),
    ])


    gs = GridSearchCV(
        pipe,
        PARAM_GRID,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=1,          # ← was -1; avoid loky on Windows
    )
    gs.fit(X_train, y_train)

    best_mae = -gs.best_score_
    print("\n=== CV (2013–24) – ElasticNet ===")
    print(f"Best params: {gs.best_params_}")
    print(f"CV MAE:      {best_mae:.3f}")

    # 5) Final fit on all 2013–24 and predict 2024–25
    best_model = gs.best_estimator_
    best_model.fit(X_train, y_train)
    show_coefs(best_model, feat_cols)

    last_tr_idx, last_va_idx = list(expanding_window_splits(train))[-1]
    X_val = train.iloc[last_va_idx][feat_cols].values
    y_val = train.iloc[last_va_idx]["PTS"].values

    y_val_hat = best_model.predict(X_val)
    A = np.vstack([np.ones_like(y_val_hat), y_val_hat]).T
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0.1:
        # guard: skip bad/near-flat/negative calibration
        a, b = 0.0, 1.0
        print("Calibration skipped (unstable slope); using identity mapping.")
    
    y_pred_raw = best_model.predict(X_test)
    y_pred = np.clip(a + b * y_pred_raw, 0, 80)

    # 6) Evaluate on 2024–25
    metrics = evaluate_and_report(y_test, y_pred, l5_base, std_base)

    # 7) Save predictions to CSV / plots / Mongo
    out = test[["GAME_ID","GAME_DATE","SEASON","PTS"]].copy()
    out["PRED_PTS"]     = y_pred
    out["L5_BASELINE"]  = l5_base
    out["STD_BASELINE"] = std_base
    out.to_csv(f"{PLAYER_NAME.replace(' ', '_')}_elasticnet_2024_25_predictions.csv", index=False)
    print("Saved CSV: jokic_elasticnet_2024_25_predictions.csv")

    save_plots(out)
    print("Saved plots: jokic_pred_vs_actual_2024_25.png, jokic_residuals_hist_2024_25.png")

    save_predictions_to_mongo(out)
    print("Done.")
