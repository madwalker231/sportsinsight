import os, glob, math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# map short keys from filenames to display names
PLAYER_NAME = {
    "jokic": "Nikola Jokic",
    "doncic": "Luka Doncic",
    "embiid": "Joel Embiid",
    "giannis": "Giannis Antetokounmpo",
    "tatum": "Jayson Tatum",
    "sga": "Shai Gilgeous-Alexander",
}
MODEL_NAME = {
    "elasticnet": "ElasticNet",
    "randomforest": "RandomForest",
    "svr": "SVR-RBF",
    "mlp": "MLP",
}

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_metrics(df: pd.DataFrame):
    # support both naming styles from your scripts
    y_true_col = pick_col(df, ["PTS", "Pts", "points", "PTS_actual"])
    y_pred_col = pick_col(df, ["PRED_PTS", "Pred", "PTS_pred"])
    l5_col     = pick_col(df, ["L5_BASELINE", "L5_pred", "PTS_pred_L5"])
    std_col    = pick_col(df, ["STD_BASELINE", "STD_pred", "PTS_SEASON_TD", "PTS_pred_STD"])

    if not y_true_col or not y_pred_col:
        raise ValueError(f"Missing prediction/target columns. Found: {df.columns.tolist()}")

    y_true = df[y_true_col].astype(float).values
    y_pred = df[y_pred_col].astype(float).values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    med  = float(np.median(np.abs(y_true - y_pred)))
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")

    mae_l5 = mae_std = skill_l5 = skill_std = float("nan")
    if l5_col is not None:
        mae_l5 = mean_absolute_error(y_true, df[l5_col].astype(float).values)
        if mae_l5 > 0:
            skill_l5 = 100.0 * (1.0 - mae / mae_l5)
    if std_col is not None:
        mae_std = mean_absolute_error(y_true, df[std_col].astype(float).values)
        if mae_std > 0:
            skill_std = 100.0 * (1.0 - mae / mae_std)

    return dict(
        Rows=len(df),
        MAE=mae, RMSE=rmse, MedAE=med, R2=r2,
        MAE_L5=mae_l5, MAE_STD=mae_std,
        Skill_L5=skill_l5, Skill_STD=skill_std,
    )

def parse_name(fp: str):
    base = os.path.basename(fp)
    stem = base.replace("_2024_25_predictions.csv", "")
    parts = stem.split("_")
    # expected: "<player>_<model>"
    player_key = parts[0].lower() if parts else "unknown"
    model_key  = parts[1].lower() if len(parts) > 1 else "unknown"
    player = PLAYER_NAME.get(player_key, player_key.title())
    model  = MODEL_NAME.get(model_key, model_key.title())
    return player, model

def main():
    # find all prediction CSVs anywhere under the repo/Data
    files = sorted(glob.glob("**/*_2024_25_predictions.csv", recursive=True))
    if not files:
        print("No prediction CSVs found matching *_2024_25_predictions.csv")
        return

    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            player, model = parse_name(fp)
            m = compute_metrics(df)
            m.update({"Player": player, "Model": model, "File": fp})
            rows.append(m)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")

    if not rows:
        print("No metrics computed.")
        return

    out = pd.DataFrame(rows)
    # stable sort if Player exists, otherwise just by MAE
    if "Player" in out.columns:
        out = out.sort_values(["Player", "MAE", "Model"]).reset_index(drop=True)
    else:
        out = out.sort_values(["MAE"]).reset_index(drop=True)

    out_dir = os.path.join("artifacts", "2024-25")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "metrics_summary.csv")
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    # quick console peek
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(out)

if __name__ == "__main__":
    main()
