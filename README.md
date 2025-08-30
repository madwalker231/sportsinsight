Data window. Train/evaluate baselines and model selection using Regular Season 2013–14 → 2023–24; hold out 2024–25 strictly for final tests and the demo UI.
Players. Nikola Jokić (203999), Luka Dončić (1629029), Joel Embiid (203954), Jayson Tatum (1628369), Giannis Antetokounmpo (203507), Shai Gilgeous-Alexander (1628983).
Features (for this phase). Past-only: PTS_L5, REB_L5, AST_L5, MIN_L5, FGA_L5, FG3A_L5, FTA_L5, REST_DAYS, IS_HOME, OPP_DEF_RATING.
Baselines.

L5 rolling mean (window=5), shifted by 1 game (no lookahead).

Season-to-date expanding mean, shifted by 1.
Metrics. MAE, RMSE, Skill vs L5, Skill vs STD, MAE/σᵧ, and lag-1 autocorr of the target.
Model selection rule (per player). Choose target (PTS vs REB vs AST) using 2013–24 baseline predictability: lower MAE/σᵧ and higher lag-1 autocorr wins (must also be basketball-reasonable for that player).