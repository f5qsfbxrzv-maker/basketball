import pandas as pd

df = pd.read_csv('data/training_data_matchup_with_injury_advantage_FIXED.csv')

print('Checking for moneyline odds columns:')
odds_cols = [c for c in df.columns if 'moneyline' in c.lower() or 'odds' in c.lower() or 'price' in c.lower()]
print(f'  Found: {odds_cols}')

print(f'\nSeasons available: {sorted(df["season"].unique())}')
print(f'\n2023-24 season games: {len(df[df["season"] == "2023-24"])}')
print(f'2024-25 season games: {len(df[df["season"] == "2024-25"])}')

# Check if we have any odds data at all
if odds_cols:
    for col in odds_cols:
        non_null = df[col].notna().sum()
        print(f'\n{col}: {non_null} non-null values ({non_null/len(df)*100:.1f}%)')
