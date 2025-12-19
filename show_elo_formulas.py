import pandas as pd

print('\n' + '='*95)
print('COMPOSITE ELO FORMULA (from off_def_elo_system.py)')
print('='*95)

print('\n📐 FORMULA:')
print('  composite_elo = (off_elo + def_elo) / 2')
print('\n📊 BASELINE VALUES:')
print('  off_elo baseline: 1500 (higher = better offense)')
print('  def_elo baseline: 1500 (higher = better defense)')
print('\n⚠️  NOTE: Defense is NOT inverted in composite calculation')
print('    Higher def_elo = better defense (fewer points allowed)')
print('    Old buggy formula had: (off_elo + (2000 - def_elo)) / 2')
print('    New correct formula: (off_elo + def_elo) / 2')
print('\n' + '='*95)

# Now check how this maps to the training data
print('\n\nCHECKING TRAINING DATA STRUCTURE:')
print('='*95)

df = pd.read_csv('data/training_data_matchup_with_injury_advantage.csv')

# Find Atlanta home games
atl_home = df[df['home_team'] == 'ATL'].head(5)
print('\nATLANTA HOME GAMES:')
print(f"{'Date':<12} {'Home Team':<10} {'Away Team':<10} {'home_composite_elo':<20}")
print('-'*55)
for _, row in atl_home.iterrows():
    print(f"{row['date']:<12} {row['home_team']:<10} {row['away_team']:<10} {row['home_composite_elo']:<20.2f}")

# Find Atlanta away games  
atl_away = df[df['away_team'] == 'ATL'].head(5)
print('\nATLANTA AWAY GAMES:')
print(f"{'Date':<12} {'Home Team':<10} {'Away Team':<10} {'away_composite_elo':<20}")
print('-'*55)
for _, row in atl_away.iterrows():
    print(f"{row['date']:<12} {row['home_team']:<10} {row['away_team']:<10} {row['away_composite_elo']:<20.2f}")

print('\n' + '='*95)
print('DATASET STRUCTURE:')
print('  - Each row represents ONE game from HOME team perspective')
print('  - home_composite_elo = ELO rating of home_team on game date')
print('  - away_composite_elo = ELO rating of away_team on game date')
print('  - Both values come from elo_ratings table composite_elo column')
print('='*95)
