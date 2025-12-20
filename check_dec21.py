import pandas as pd

df = pd.read_csv('data/nba_schedule_2025_26.csv')
dec21 = df[df['game_date'] == '2025-12-21']

print(f'Total games on Dec 21: {len(dec21)}')
print('\nGames:')
for _, row in dec21.iterrows():
    print(f'  {row["away_team"]} @ {row["home_team"]} - {row["game_time"]}')
