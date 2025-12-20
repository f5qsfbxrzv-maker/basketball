import pandas as pd

df = pd.read_csv('data/master_training_data_v6.csv')

print(f'Total games: {len(df):,}')
print(f'Games with scores: {df["home_score"].notna().sum():,}')
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
print(f'\nColumns: {df.columns.tolist()}')
print(f'\nSample data:')
print(df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'actual_spread']].head(10))
print(f'\nScore statistics:')
print(f'Home score range: {df["home_score"].min():.0f} to {df["home_score"].max():.0f}')
print(f'Away score range: {df["away_score"].min():.0f} to {df["away_score"].max():.0f}')
print(f'Spread range: {df["actual_spread"].min():.0f} to {df["actual_spread"].max():.0f}')
print(f'Spread mean: {df["actual_spread"].mean():.2f}, std: {df["actual_spread"].std():.2f}')
