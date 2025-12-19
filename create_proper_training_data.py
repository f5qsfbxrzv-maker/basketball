"""
Create training data with EXACT same 22 features as Trial 1306,
but using new Gold Standard ELO values instead of old K=32 ELO.

This is the proper apples-to-apples comparison.
"""
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\data\live\nba_betting_data.db"
OLD_TRAINING_DATA = r"data\training_data_matchup_with_injury_advantage_FIXED.csv"
NEW_TRAINING_DATA = r"data\training_data_GOLD_ELO_22_features.csv"

print("=" * 80)
print("CREATING PROPER TRAINING DATA")
print("=" * 80)
print("\nObjective: Same 22 features as Trial 1306, but with Gold Standard ELO\n")

# Load old training data
print("1. Loading old training data...")
old_df = pd.read_csv(OLD_TRAINING_DATA)
print(f"   Loaded {len(old_df)} games")
print(f"   Columns: {len(old_df.columns)}")
print(f"   Features: {list(old_df.columns)}")

# The 22 features used by Trial 1306:
TRIAL_1306_FEATURES = [
    "home_composite_elo",
    "away_composite_elo",
    "off_elo_diff",
    "def_elo_diff",
    "ewma_efg_diff",
    "ewma_pace_diff",
    "ewma_tov_diff",
    "ewma_orb_diff",
    "ewma_vol_3p_diff",
    "ewma_chaos_home",
    "injury_matchup_advantage",
    "net_fatigue_score",
    "ewma_foul_synergy_home",
    "total_foul_environment",
    "league_offensive_context",
    "season_progress",
    "pace_efficiency_interaction",
    "projected_possession_margin",
    "three_point_matchup",
    "net_free_throw_advantage",
    "star_power_leverage",
    "offense_vs_defense_matchup"
]

print(f"\n2. Trial 1306 used these 22 features:")
for i, feat in enumerate(TRIAL_1306_FEATURES, 1):
    present = "✓" if feat in old_df.columns else "✗"
    print(f"   {i:2d}. {present} {feat}")

# Check which features are missing
missing = [f for f in TRIAL_1306_FEATURES if f not in old_df.columns]
if missing:
    print(f"\n   WARNING: Missing features: {missing}")

# Load Gold Standard ELO ratings
print(f"\n3. Loading Gold Standard ELO ratings...")
conn = sqlite3.connect(DB_PATH)
elo_df = pd.read_sql_query("""
    SELECT team, season, game_date, 
           composite_elo, off_elo, def_elo
    FROM elo_ratings
    ORDER BY game_date
""", conn)
conn.close()

print(f"   Loaded {len(elo_df)} ELO ratings")
print(f"   Seasons: {elo_df['season'].unique()}")

# Create ELO lookup dictionary
print(f"\n4. Creating ELO lookup dictionary...")
elo_dict = {}
for _, row in elo_df.iterrows():
    key = (row['team'], row['season'], row['game_date'])
    elo_dict[key] = {
        'composite_elo': row['composite_elo'],
        'off_elo': row['off_elo'],
        'def_elo': row['def_elo']
    }
print(f"   Created {len(elo_dict)} ELO entries")

def get_latest_elo(team, season, before_date):
    """Get most recent ELO before a given date"""
    dates = [date for (t, s, date) in elo_dict.keys() if t == team and s == season and date < before_date]
    if dates:
        latest_date = max(dates)
        return elo_dict[(team, season, latest_date)]
    return None

# Replace ELO features in old training data
print(f"\n5. Replacing ELO features with Gold Standard values...")
new_df = old_df.copy()

# Track success rate
elo_found = 0
elo_missing = 0

for idx, row in new_df.iterrows():
    if idx % 1000 == 0:
        print(f"   Progress: {idx}/{len(new_df)} games...")
    
    home_elo = get_latest_elo(row['home_team'], row['season'], row['date'])
    away_elo = get_latest_elo(row['away_team'], row['season'], row['date'])
    
    if home_elo and away_elo:
        elo_found += 1
        # Replace the 4 ELO features
        new_df.at[idx, 'home_composite_elo'] = home_elo['composite_elo']
        new_df.at[idx, 'away_composite_elo'] = away_elo['composite_elo']
        new_df.at[idx, 'off_elo_diff'] = home_elo['off_elo'] - away_elo['off_elo']
        new_df.at[idx, 'def_elo_diff'] = home_elo['def_elo'] - away_elo['def_elo']
    else:
        elo_missing += 1

print(f"\n   ELO found: {elo_found}/{len(new_df)} ({elo_found/len(new_df)*100:.1f}%)")
print(f"   ELO missing: {elo_missing}/{len(new_df)} ({elo_missing/len(new_df)*100:.1f}%)")

# Drop rows with missing ELO
if elo_missing > 0:
    print(f"\n   Dropping {elo_missing} games with missing ELO...")
    new_df = new_df.dropna(subset=['home_composite_elo', 'away_composite_elo'])

# Verify we have all 22 features
print(f"\n6. Verifying feature set...")
for feat in TRIAL_1306_FEATURES:
    if feat not in new_df.columns:
        print(f"   ERROR: Missing {feat}")
    else:
        print(f"   ✓ {feat}: range [{new_df[feat].min():.2f}, {new_df[feat].max():.2f}]")

# Save new training data
print(f"\n7. Saving new training data...")
new_df.to_csv(NEW_TRAINING_DATA, index=False)
print(f"   Saved to: {NEW_TRAINING_DATA}")
print(f"   Total games: {len(new_df)}")

# Compare ELO distributions
print(f"\n8. ELO COMPARISON")
print("-" * 80)
print("Old ELO (K=32):")
print(f"   home_composite_elo: {old_df['home_composite_elo'].mean():.1f} ± {old_df['home_composite_elo'].std():.1f}")
print(f"   Range: [{old_df['home_composite_elo'].min():.1f}, {old_df['home_composite_elo'].max():.1f}]")
print(f"   Spread: {old_df['home_composite_elo'].max() - old_df['home_composite_elo'].min():.1f} points")

print("\nNew ELO (K=15, Gold Standard):")
print(f"   home_composite_elo: {new_df['home_composite_elo'].mean():.1f} ± {new_df['home_composite_elo'].std():.1f}")
print(f"   Range: [{new_df['home_composite_elo'].min():.1f}, {new_df['home_composite_elo'].max():.1f}]")
print(f"   Spread: {new_df['home_composite_elo'].max() - new_df['home_composite_elo'].min():.1f} points")

print("\n" + "=" * 80)
print("READY FOR PROPER COMPARISON")
print("=" * 80)
print(f"""
Now you can:
1. Train XGBoost with Trial 1306 hyperparameters on NEW data
2. Compare log-loss: old model on old ELO vs new model on new ELO
3. This is the TRUE test: did Gold Standard ELO improve predictions?

Old Model: 0.6222 log-loss on old ELO (K=32)
New Model: ??? log-loss on Gold ELO (K=15)

If new model < 0.6222, Gold Standard is BETTER
If new model > 0.6222, old system was BETTER (despite Brooklyn ghost)
""")
