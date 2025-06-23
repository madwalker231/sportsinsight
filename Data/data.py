import os
from nba_api.stats.endpoints import LeagueDashPlayerStats
import pandas as pd

def fetch_player_stats(season: str = '2021-22', per_mode: str = 'PerGame') -> pd.DataFrame:
    """
    Fetch league-wide player stats for given NBA season.
    season must be in the form 'YYYY-YY' (e.g. '2021-22').
    per_mode options: 'PerGame', 'Totals', 'Per36', 'Per48'
    """

    endpoint = LeagueDashPlayerStats(
        season=season,
        per_mode_detailed=per_mode,
        season_type_all_star='Regular Season'
    )
    return endpoint.get_data_frames()[0]

def main():
    season = '2021-22'
    stats_df = fetch_player_stats(season, per_mode='PerGame')

    here = os.path.dirname(__file__)
    out_dir = os.path.join(here, 'CSV_data')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir,f'player_stats_{season}_pergame.csv')
    stats_df.to_csv(out_path, index=False)
    print(f"[✔] Saved {len(stats_df)} rows to:\n    {out_path}")

if __name__ =='__main__':
    main()
