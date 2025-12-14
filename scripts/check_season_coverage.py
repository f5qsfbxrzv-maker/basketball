import pandas as pd

df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])

s2024 = df[df['season'] == '2024-25']
print(f'Total 2024-25 games in training data: {len(s2024)}')
print(f'Date range: {s2024["date"].min().date()} to {s2024["date"].max().date()}')

# Check what we fetched
import sqlite3
conn = sqlite3.connect('data/live/historical_closing_odds.db')
odds_df = pd.read_sql('SELECT * FROM moneyline_odds', conn)
conn.close()

print(f'\nOdds fetched: {len(odds_df)} games')
print(f'Odds range: {odds_df["game_date"].min()} to {odds_df["game_date"].max()}')

print(f'\nFull NBA season: ~1230 games (Oct 2024 - Apr 2025)')
print(f'Training data has: {len(s2024)} games')
print(f'Odds coverage: {len(odds_df)}/{len(s2024)} games ({len(odds_df)/len(s2024)*100:.1f}%)')
