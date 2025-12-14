import pandas as pd

df = pd.read_csv('data/training_data_with_temporal_features.csv', usecols=['date', 'season', 'home_team', 'away_team'])
df['date'] = pd.to_datetime(df['date'])
s2024 = df[df['season'] == '2024-25']

print(f'2024-25 season: {len(s2024)} games')
print(f'Date range: {s2024["date"].min().date()} to {s2024["date"].max().date()}')
print(f'\nLast 10 games:')
print(s2024.tail(10)[['date', 'home_team', 'away_team']].to_string())
