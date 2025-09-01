import sys, glob, pandas as pd, numpy as np
from pathlib import Path

def summarize_one(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])
    mae = np.mean(np.abs(df["PTS"] - df["PRED_PTS"]))
    rmse = np.sqrt(np.mean((df["PTS"] - df["PRED_PTS"])**2))
    medae = np.median(np.abs(df["PTS"] - df["PRED_PTS"]))
    mae_l5 = np.mean(np.abs(df["PTS"] - df["L5_BASELINE"]))
    mae_std = np.mean(np.abs(df["PTS"] - df["STD_BASELINE"]))
    skill_l5  = 100.0*(1.0 - mae/mae_l5) if mae_l5 > 0 else np.nan
    skill_std = 100.0*(1.0 - mae/mae_std) if mae_std > 0 else np.nan
    return dict(
        file=Path(csv_path).name, MAE=mae, RMSE=rmse, MedAE=medae,
        MAE_L5=mae_l5, MAE_STD=mae_std, Skill_L5=skill_l5, Skill_STD=skill_std
    )

if __name__ == "__main__":
    paths = sys.argv[1:] or glob.glob("**/*_2024_25_predictions.csv", recursive=True)
    rows = [summarize_one(p) for p in paths]
    out = pd.DataFrame(rows).sort_values("MAE")
    out.to_csv("artifacts/2024-25/metrics_summary.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
