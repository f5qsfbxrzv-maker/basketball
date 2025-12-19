"""Check which training dataset has the v6 features"""
import pandas as pd
from pathlib import Path

v6_features = [
    'vs_efg_diff', 'vs_tov', 'vs_reb_diff', 'vs_ftr_diff', 'vs_net_rating',
    'expected_pace', 'rest_days_diff', 'is_b2b_diff', 'h2h_win_rate_l3y',
    'injury_impact_diff', 'elo_diff', 'off_elo_diff', 'def_elo_diff',
    'composite_elo_diff', 'sos_diff', 'h_off_rating', 'h_def_rating',
    'a_off_rating', 'a_def_rating'
]

files_to_check = [
    'data/master_training_data_v6.csv',
    'data/training_data_matchup_optimized.csv',
    'data/training_data_lean.csv',
    'data/training_data_with_injury_shock.csv'
]

print("=" * 80)
print("SEARCHING FOR DATASET WITH 19 V6 FEATURES")
print("=" * 80)
print()

best_match = None
best_count = 0

for filepath in files_to_check:
    p = Path(filepath)
    if not p.exists():
        print(f"❌ {filepath}: NOT FOUND")
        continue
    
    try:
        df = pd.read_csv(filepath, nrows=1)
        matches = sum(1 for f in v6_features if f in df.columns)
        total_cols = len(df.columns)
        
        print(f"📊 {filepath}:")
        print(f"   Total columns: {total_cols}")
        print(f"   V6 features found: {matches}/19")
        
        if matches > best_count:
            best_count = matches
            best_match = filepath
        
        if matches == 19:
            print(f"   ✅ HAS ALL 19 V6 FEATURES!")
            # Check for injury features
            has_injury_shock = all(c in df.columns for c in ['injury_shock_diff', 'star_mismatch'])
            has_injury_advantage = 'injury_matchup_advantage' in df.columns
            print(f"   Injury shock components: {'✅' if has_injury_shock else '❌'}")
            print(f"   injury_matchup_advantage: {'✅' if has_injury_advantage else '❌'}")
        print()
    except Exception as e:
        print(f"❌ {filepath}: ERROR - {e}")
        print()

print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if best_count == 19:
    print(f"✅ Use: {best_match}")
    print(f"   Has all 19 v6 features")
    
    # Load and check if we need to add injury_matchup_advantage
    df = pd.read_csv(best_match, nrows=1)
    if 'injury_matchup_advantage' not in df.columns:
        print()
        print("⚠️  ACTION REQUIRED:")
        print("   injury_matchup_advantage is MISSING")
        print("   Need to calculate and add this feature to the dataset")
else:
    print(f"⚠️  PROBLEM: Best match only has {best_count}/19 features")
    print(f"   File: {best_match}")
    print()
    print("❌ No dataset has all 19 v6 features!")
    print("   Need to regenerate training data with feature_calculator_v5")
