import pandas as pd

df = pd.read_csv('data/training_data_with_temporal_features.csv')

print(f'Rows: {len(df):,}')
print(f'Columns: {len(df.columns)}')

feature_cols = [c for c in df.columns if c not in [
    'date', 'game_id', 'home_team', 'away_team', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_game_total', 'target_over_under', 'target_home_cover', 'target_over'
]]

print(f'Features: {len(feature_cols)}')

print(f'\naway_composite_elo: {"PRESENT" if "away_composite_elo" in df.columns else "MISSING"}')
print(f'home_composite_elo: {"PRESENT" if "home_composite_elo" in df.columns else "MISSING"}')

if 'away_composite_elo' in df.columns:
    print(f'\nSample ELO values (first 5 games):')
    print(df[['date', 'home_team', 'away_team', 'home_composite_elo', 'away_composite_elo']].head())
    
    print(f'\nELO Statistics:')
    print(df[['home_composite_elo', 'away_composite_elo']].describe())
