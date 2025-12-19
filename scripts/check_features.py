import pandas as pd

df = pd.read_csv('data/training_data_with_temporal_features.csv')

exclude = ['game_id', 'game_date', 'home_team', 'away_team', 'spread', 'home_cover']
feature_cols = [c for c in df.columns if c not in exclude]

print(f"Total features: {len(feature_cols)}")
print(f"\nAll columns ({len(df.columns)}):")
for i, c in enumerate(sorted(df.columns), 1):
    marker = " [FEATURE]" if c in feature_cols else ""
    print(f"  {i:2d}. {c}{marker}")
