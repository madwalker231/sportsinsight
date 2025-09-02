import pandas as pd
import os

p = r"artifacts\2024-25\metrics_summary.csv"
df = pd.read_csv(p)

# Handle slight column-name differences gracefully
skill_l5  = next((c for c in df.columns if c.lower().startswith("skill_l5")), None)
skill_std = next((c for c in df.columns if c.lower().startswith("skill_std")), None)
player_col = "Player" if "Player" in df.columns else df.columns[0]

w = df.sort_values([player_col, "MAE"]).groupby(player_col, as_index=False).first()

cols_to_show = [c for c in [player_col, "Model", "MAE", skill_l5, skill_std] if c and c in w.columns]
print(w[cols_to_show].to_string(index=False))

out = r"artifacts\2024-25\best_per_player.csv"
w.to_csv(out, index=False)
print("\nSaved", out)
