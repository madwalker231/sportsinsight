# target_sweep_baselines.py
import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats
from nba_api.stats.static import teams as static_teams

SEASONS = [f"{y}-{str((y+1) % 100).zfill(2)}" for y in range(2013, 2024)]  # 2013-14..2023-24
PLAYERS = {
    "Jayson Tatum": 1628369,
    "Shai Gilgeous-Alexander": 1628983,
}
SLEEP_SEC = 0.7
WINDOW_L5 = 5
OUT_CSV = True

TARGETS = ["PTS", "REB", "AST"]

def fetch_gamelog(player_id: int, season: str) -> pd.DataFrame:
    """Fetch a season's regular-season per-game logs, uppercase columns, sort by date."""
    df = PlayerGameLog(player_id=player_id, season=season,
                       season_type_all_star="Regular Season").get_data_frames()[0]
    if df.empty:
        return df
    df.columns = [c.upper() for c in df.columns]
    keep = ["GAME_ID","GAME_DATE","MATCHUP","TEAM_ID","MIN","PTS","REB","AST","FGA","FG3A","FTA","WL"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["SEASON"] = season
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df

def team_def_rating_map(season: str) -> dict:
    """Optional opponent DEF_RATING (not used in baselines, but you may keep for context later)."""
    try:
        teams = static_teams.get_teams()
        id2abbr = {t["id"]: t["abbreviation"] for t in teams}
        tdf = LeagueDashTeamStats(season=season, per_mode_detailed="PerGame",
                                  measure_type_detailed_defense="Advanced").get_data_frames()[0]
        tdf.columns = [c.upper() for c in tdf.columns]
        if "TEAM_ID" not in tdf.columns or "DEF_RATING" not in tdf.columns:
            return {}
        tdf["TEAM_ABBR"] = tdf["TEAM_ID"].map(id2abbr)
        return dict(zip(tdf["TEAM_ABBR"], tdf["DEF_RATING"]))
    except Exception:
        return {}

def l5_baseline(x: pd.Series, win: int = 5) -> pd.Series:
    return x.rolling(win).mean().shift(1)

def std_baseline(x: pd.Series) -> pd.Series:
    return x.expanding().mean().shift(1)

def series_autocorr_lag1(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.autocorr(lag=1)

def baseline_metrics(df: pd.DataFrame, target: str) -> dict:
    """Compute baselines across all seasons concatenated."""
    if target not in df.columns:
        return {"N": 0}
    y = df[target].astype(float)
    pred_l5  = l5_baseline(y, WINDOW_L5)
    pred_std = std_baseline(y)

    mask = (~pred_l5.isna()) & (~y.isna())
    y_l5 = y[mask]
    p_l5 = pred_l5[mask]

    mask2 = (~pred_std.isna()) & (~y.isna())
    y_std = y[mask2]
    p_std = pred_std[mask2]

    mae_l5  = np.mean(np.abs(y_l5 - p_l5)) if len(y_l5) else np.nan
    rmse_l5 = np.sqrt(np.mean((y_l5 - p_l5) ** 2)) if len(y_l5) else np.nan
    mae_std  = np.mean(np.abs(y_std - p_std)) if len(y_std) else np.nan
    rmse_std = np.sqrt(np.mean((y_std - p_std) ** 2)) if len(y_std) else np.nan

    # Normalize MAE by the variability of the target over the *evaluated* samples
    sig_l5  = y_l5.std(ddof=1) if len(y_l5) > 1 else np.nan
    sig_std = y_std.std(ddof=1) if len(y_std) > 1 else np.nan
    nmae_l5  = mae_l5 / sig_l5 if sig_l5 and not np.isnan(sig_l5) and sig_l5 > 0 else np.nan
    nmae_std = mae_std / sig_std if sig_std and not np.isnan(sig_std) and sig_std > 0 else np.nan

    ac1 = series_autocorr_lag1(y)

    return {
        "TARGET": target,
        "N_L5": int(len(y_l5)),
        "MAE_L5": mae_l5, "RMSE_L5": rmse_l5, "MAE_L5_over_sigma": nmae_l5,
        "N_STD": int(len(y_std)),
        "MAE_STD": mae_std, "RMSE_STD": rmse_std, "MAE_STD_over_sigma": nmae_std,
        "Lag1_Autocorr": ac1,
        "Sigma_all": y.std(ddof=1)
    }

if __name__ == "__main__":
    rows = []
    for name, pid in PLAYERS.items():
        # Fetch & concat seasons (skip empty seasons automatically)
        frames = []
        for s in SEASONS:
            time.sleep(SLEEP_SEC)
            df = fetch_gamelog(pid, s)
            if not df.empty:
                frames.append(df)
            else:
                print(f"{name} – {s}: no games")
        if not frames:
            print(f"[WARN] No data for {name} in {SEASONS[0]}..{SEASONS[-1]}")
            continue

        all_df = pd.concat(frames, ignore_index=True).sort_values("GAME_DATE").reset_index(drop=True)

        print(f"\n=== {name} — Baseline sweep on {SEASONS[0]}..{SEASONS[-1]} ===")
        per_target = []
        for tgt in TARGETS:
            m = baseline_metrics(all_df, tgt)
            per_target.append(m)
            print(f"{tgt:>3}: "
                  f"N(L5)={m['N_L5']:4d}  MAE_L5={m['MAE_L5']:.3f}  NMAE_L5={m['MAE_L5_over_sigma']:.3f} | "
                  f"N(STD)={m['N_STD']:4d}  MAE_STD={m['MAE_STD']:.3f}  NMAE_STD={m['MAE_STD_over_sigma']:.3f} | "
                  f"Lag1={m['Lag1_Autocorr']:.3f}  Sigma={m['Sigma_all']:.3f}")
        # Pick “best” target by (lowest normalized MAE among the two baselines) + higher lag-1
        df_sel = pd.DataFrame(per_target)
        df_sel["Score"] = df_sel[["MAE_L5_over_sigma","MAE_STD_over_sigma"]].min(axis=1)
        pick = df_sel.sort_values(["Score", "Lag1_Autocorr"], ascending=[True, False]).iloc[0]["TARGET"]
        print(f"→ Suggested target for {name}: {pick}\n")

        df_sel.insert(0, "PLAYER", name)
        rows.append(df_sel)

        if OUT_CSV:
            df_sel.to_csv(f"baseline_sweep_{name.replace(' ','_')}.csv", index=False)

    if rows:
        summary = pd.concat(rows, ignore_index=True)
        print("\n=== Summary (save this for your thesis appendix) ===")
        print(summary[["PLAYER","TARGET","N_L5","MAE_L5","MAE_L5_over_sigma",
                       "N_STD","MAE_STD","MAE_STD_over_sigma","Lag1_Autocorr","Sigma_all","Score"]]
              .to_string(index=False))
        if OUT_CSV:
            summary.to_csv("baseline_sweep_summary.csv", index=False)
            print("\nWrote: baseline_sweep_summary.csv")
