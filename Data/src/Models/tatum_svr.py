import os
from dotenv import load_dotenv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymongo import MongoClient, UpdateOne

from gridfs import GridFS
from pathlib import Path
from shutil import copy2
import json
from datetime import datetime 

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

PLAYER_ID   = 1628369
PLAYER_NAME = "Jayson Tatum"

TRAIN_SEASONS = [f"{y}-{str((y+1) % 100).zfill(2)}" for y in range(2013, 2024)]  # 2013-14..2023-24
TEST_SEASON   = "2024-25"
SLEEP_SEC     = 0.7  # throttle nba_api


load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME   = "sportsinsight"
PRED_COLL = "predictions_2024_25"

# Tight, predeclared SVR grid (no optimization theater)
PARAM_GRID = {
    "model__C":       [1.0, 10.0, 50.0],
    "model__epsilon": [0.1, 0.2, 0.5],
    "model__gamma":   ["scale", "auto"],   # RBF kernel width
}

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
    # "PHI vs BOS" or "PHI @ BOS" -> ("BOS", is_home 1/0)
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

    # Rolling L5 (shifted) across full chronology
    df_all = df_all.sort_values("GAME_DATE").reset_index(drop=True)
    for col in ["PTS","REB","AST","MIN","FGA","FTA","FG3A"]:
        if col in df_all.columns:
            df_all[f"{col}_L5"] = df_all[col].rolling(5).mean().shift(1)

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

    print("\n=== 2024–25 Test Metrics (SVR-RBF) ===")
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
    plt.plot(dates, pred_df["PRED_PTS"], marker="o", label="SVR (RBF)")
    plt.plot(dates, pred_df["L5_BASELINE"], linestyle="--", label="L5 baseline")
    plt.xticks(rotation=45, ha="right")
    plt.title("Tatum – Predicted vs Actual (2024–25)")
    plt.ylabel("Points")
    plt.tight_layout()
    plt.legend()
    plt.savefig("tatum_pred_vs_actual_2024_25.png", dpi=140)

    # Residuals histogram
    plt.figure(figsize=(6,4))
    resid = pred_df["PTS"] - pred_df["PRED_PTS"]
    plt.hist(resid, bins=12)
    plt.title("Residuals (Actual − Predicted) – 2024–25")
    plt.xlabel("Points")
    plt.tight_layout()
    plt.savefig("tatum_residuals_hist_2024_25.png", dpi=140)

def save_predictions_to_mongo(pred_df: pd.DataFrame):
    if not MONGO_URI or "<user>:" in MONGO_URI:
        return
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

def _safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:
        return None

def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def upload_csv_and_run_record(csv_path: str,
                              metrics: dict,
                              best_params: dict,
                              calibration: dict,
                              feature_cols: list,
                              row_count: int,
                              model_name: str = "ElasticNet"):
    """Store CSV in GridFS and insert one model_runs document (skip if no MONGO_URI)."""
    if not MONGO_URI:
        print("(Mongo) MONGO_URI not set; skipping CSV + run record upload.")
        return None
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        # 1) CSV → GridFS
        fs = GridFS(db)
        with open(csv_path, "rb") as f:
            csv_file_id = fs.put(
                f,
                filename=os.path.basename(csv_path),
                content_type="text/csv",
                player_id=PLAYER_ID,
                player_name=PLAYER_NAME,
                season=TEST_SEASON,
                model=model_name,
                pred_target="PTS",
            )

        # 2) model_runs doc
        hp = {}
        if isinstance(best_params, dict):
            # capture SVR keys (handles Pipeline "model__*" or plain)
            for k in ("model__C","model__epsilon","model__gamma","C","epsilon","gamma"):
                if k in best_params:
                    hp[k.split("__")[-1]] = _num(best_params[k])

        run_doc = {
            "player_id":     PLAYER_ID,
            "player_name":   PLAYER_NAME,
            "season":        TEST_SEASON,
            "model":         model_name,
            "pred_target":   "PTS",
            "created_at":    datetime.utcnow(),
            "csv_file_id":   csv_file_id,
            "csv_filename":  os.path.basename(csv_path),
            "row_count":     int(row_count),
            "features":      list(feature_cols),
            "hp":            hp,
            "calibration": {
                "a": _num((calibration or {}).get("a")),
                "b": _num((calibration or {}).get("b")),
                "corr": _num((calibration or {}).get("corr")),
                "desc": (calibration or {}).get("desc"),
            },
            "metrics": {k: _num(v) for k, v in (metrics or {}).items()},
        }
        run_id = db["model_runs"].insert_one(run_doc).inserted_id
        print(f"(Mongo) CSV stored; run record _id={run_id}")
        return {"run_id": run_id, "csv_file_id": csv_file_id}
    except Exception as e:
        print(f"(Mongo warning) upload_csv_and_run_record failed: {e}")
        return None

def export_static_for_pages(player_id, player_name, season,
                            pred_df, metrics,
                            csv_src_path, plot_pred_path=None, plot_resid_path=None,
                            export_root_choices=("docs", "public")):
    """
    Write minimal payload for the static UI:

      <export_root>/data/<player_id>/
        metrics.json
        history.json
        next_game.json
        predictions.csv
        plots/pred_vs_actual.png  (optional)
        plots/residuals_hist.png  (optional)

    Chooses 'docs' if present (GitHub Pages), else 'public'. Searches upward for repo root.
    """
    if pred_df is None or pred_df.empty:
        print("[publish] pred_df empty; nothing to write.")
        return

    # --- find repo root (look upward for docs/public/.git) ---
    env_docs = os.getenv("DOCS_ROOT")  # optional override
    if env_docs:
        export_root = Path(env_docs).expanduser().resolve()
        export_root.mkdir(parents=True, exist_ok=True)
    else:
        # script lives under .../sportsinsight/Data/src/Models/...
        # go up 3 -> .../sportsinsight/, then use /docs as the site root
        export_root = Path(__file__).resolve().parents[3] / "docs"
        export_root.mkdir(parents=True, exist_ok=True)

    out_dir   = export_root / "data" / player_id
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_clean = {k: _safe_float(v) for k, v in (metrics or {}).items()}
    (out_dir / "metrics.json").write_text(json.dumps({
        "player_id": player_id,
        "player_name": player_name,
        "model": "ElasticNet",
        "season": season,
        "metrics": metrics_clean
    }, indent=2, allow_nan=False))

    # history.json
    series = []
    for _, r in pred_df.sort_values("GAME_DATE").iterrows():
        series.append({
            "game_date": pd.to_datetime(r["GAME_DATE"]).strftime("%Y-%m-%d"),
            "actual": _safe_float(r["PTS"]),
            "predicted": _safe_float(r["PRED_PTS"]),
        })
    (out_dir / "history.json").write_text(json.dumps({
        "player_id": player_id,
        "series": series
    }, indent=2, allow_nan=False))

    # next_game.json (show one decimal for the UI)
    last = pred_df.sort_values("GAME_DATE").iloc[-1]
    (out_dir / "next_game.json").write_text(json.dumps({
        "player_id": player_id,
        "predicted_points": _safe_float(np.round(last["PRED_PTS"], 1)),
        "as_of": datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }, indent=2, allow_nan=False))

    # predictions.csv
    if csv_src_path and Path(csv_src_path).exists():
        copy2(csv_src_path, out_dir / "predictions.csv")
    else:
        pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # plots (optional)
    try:
        if plot_pred_path and Path(plot_pred_path).exists():
            copy2(plot_pred_path, plots_dir / "pred_vs_actual.png")
        if plot_resid_path and Path(plot_resid_path).exists():
            copy2(plot_resid_path, plots_dir / "residuals_hist.png")
    except Exception as e:
        print(f"(copy plots) {e}")

    print(f"[publish] wrote static files to {out_dir}")

if __name__ == "__main__":
    # 1) Fetch raw data (expect 'no games' prior to debut due to injuries)
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

    # 4) Expanding-window CV for SVR (sequential to avoid Windows loky bug)
    cv_splits = list(expanding_window_splits(train))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf"))
    ])
    gs = GridSearchCV(
        pipe,
        PARAM_GRID,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=1   # sequential; stable on Windows
    )
    gs.fit(X_train, y_train)

    best_mae = -gs.best_score_
    print("\n=== CV (2013–24) – SVR (RBF) ===")
    print(f"Best params: {gs.best_params_}")
    print(f"CV MAE:      {best_mae:.3f}")

    # 5) Final fit + predict 2024–25
    best_model = gs.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # 6) Evaluate
    metrics = evaluate_and_report(y_test, y_pred, l5_base, std_base)

    # 7) Save artifacts
    out = test[["GAME_ID","GAME_DATE","SEASON","PTS"]].copy()
    out["PRED_PTS"]     = y_pred
    out["L5_BASELINE"]  = l5_base
    out["STD_BASELINE"] = std_base
    csv_path = "tatum_svr_2024_25_predictions.csv"
    out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    save_plots(out)
    print("Saved plots: tatum_pred_vs_actual_2024_25.png, tatum_residuals_hist_2024_25.png")

    export_static_for_pages(
        player_id="tatum",
        player_name=PLAYER_NAME,
        season=TEST_SEASON,
        pred_df=out,
        metrics=metrics,
        csv_src_path=csv_path,
        plot_pred_path="tatum_pred_vs_actual_2024_25.png",
        plot_resid_path="tatum_residuals_hist_2024_25.png",
    )

    last_tr_idx, last_va_idx = list(expanding_window_splits(train))[-1]
    X_val = train.iloc[last_va_idx][FEATURE_COLS].values
    y_val = train.iloc[last_va_idx]["PTS"].values
    y_val_hat = best_model.predict(X_val)
    corr_val = float(np.corrcoef(y_val_hat, y_val)[0, 1]) if (np.std(y_val_hat) > 1e-8 and np.std(y_val) > 1e-8) else np.nan

    calib_info = {
        "a": 0.0,
        "b": 1.0,
        "corr": None if not np.isfinite(corr_val) else corr_val,
        "desc": "identity (no calibration)"
    }

    upload_csv_and_run_record(
        csv_path=csv_path,
        metrics=metrics,
        best_params=gs.best_params_,
        calibration=calib_info,
        feature_cols=FEATURE_COLS,
        row_count=len(out),
        model_name="SVR",
    )

    save_predictions_to_mongo(out)

    print("Done.")

