import pandas as pd
import matplotlib.pyplot as plt

# Load
df = pd.read_csv('C:\\Users\\madwa\\Desktop\\SportsInsight\\sportsinsight\\Data\\CSV_data\\player_stats_2021-22_pergame.csv')

# Compute metrics
df['TS%']  = df['PTS'] / (2*(df['FGA'] + 0.44*df['FTA']))
df['eFG%'] = (df['FGM'] + 0.5*df['FG3M']) / df['FGA']

# Quick inspection
print(df[['PLAYER_NAME','TS%','eFG%']].head())
print(df[['TS%','eFG%']].describe())

# Plot distribution
plt.hist(df['TS%'].dropna(), bins=20)
plt.title('2021-22 True Shooting %')
plt.xlabel('TS%')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('ts_distribution.png')
plt.show()
