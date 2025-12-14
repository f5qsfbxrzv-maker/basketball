import pandas as pd
import os
from datetime import datetime

print("="*60)
print("DATA REGENERATION STATUS CHECK")
print("="*60)

# Check training data
df = pd.read_csv('data/training_data_with_features.csv')

exclude = ['game_id', 'date', 'season', 'home_team', 'away_team', 
           'target_spread', 'target_spread_cover', 'target_moneyline_win', 
           'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

features = [c for c in df.columns if c not in exclude]

print(f"\nTotal columns: {len(df.columns)}")
print(f"Feature columns: {len(features)}")
print(f"Total games: {len(df):,}")
print(f"\nFile size: {os.path.getsize('data/training_data_with_features.csv'):,} bytes")

# Check if we have all expected features
from config.feature_whitelist import FEATURE_WHITELIST

print(f"\nExpected features (whitelist): {len(FEATURE_WHITELIST)}")
print(f"Actual features in CSV: {len(features)}")

missing = set(FEATURE_WHITELIST) - set(features)
extra = set(features) - set(FEATURE_WHITELIST)

if missing:
    print(f"\n⚠️ Missing features ({len(missing)}):")
    for f in sorted(missing):
        print(f"  - {f}")

if extra:
    print(f"\n➕ Extra features ({len(extra)}):")
    for f in sorted(extra):
        print(f"  - {f}")

if not missing:
    print("\n✅ All whitelist features present!")

print(f"\nFeatures in CSV:")
for f in sorted(features):
    print(f"  - {f}")

# Check file modification time
mod_time = datetime.fromtimestamp(os.path.getmtime('data/training_data_with_features.csv'))
print(f"\nLast modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
