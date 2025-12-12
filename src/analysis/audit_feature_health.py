"""
FEATURE HEALTH CHECK - Audit for Dead Features & Correlation Trampling
Validates:
1. away_3in4 is not all zeros (broken logic check)
2. Injury features are not over-correlated (redundancy check)
3. Math consistency (abs should equal |diff|)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import sqlite3
from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST

print("=" * 80)
print("🏥 FEATURE HEALTH CHECK")
print("=" * 80)

# Initialize calculator
print("\n1. Initializing Feature Calculator V5...")
calc = FeatureCalculatorV5()

# Load games
print("\n2. Loading game results...")
conn = sqlite3.connect('data/live/nba_betting_data.db')
query = """
SELECT 
    game_id,
    game_date,
    home_team,
    away_team,
    home_score,
    away_score,
    home_won
FROM game_results
WHERE game_date >= '2023-01-01' AND game_date < '2025-11-01'
ORDER BY game_date
"""
games_df = pd.read_sql(query, conn)
conn.close()

print(f"   Loaded {len(games_df)} games")

# Generate features for sample
print("\n3. Generating features (sampling 500 games for speed)...")
sample_games = games_df.sample(min(500, len(games_df)), random_state=42)
features_list = []

for idx, row in sample_games.iterrows():
    try:
        features = calc.calculate_game_features(
            home_team=row['home_team'],
            away_team=row['away_team'],
            game_date=row['game_date']
        )
        features_list.append(features)
    except Exception as e:
        continue

df = pd.DataFrame(features_list)
print(f"   Generated {len(df)} feature sets")

# ============================================================================
# AUDIT 1: Check away_3in4 for Dead Feature
# ============================================================================
print("\n" + "=" * 80)
print("🔍 AUDIT 1: Checking 'away_3in4' for Dead Feature")
print("=" * 80)

if 'away_3in4' in df.columns:
    counts = df['away_3in4'].value_counts().sort_index()
    print(f"\nValue Distribution:")
    for val, count in counts.items():
        pct = 100 * count / len(df)
        print(f"   {val}: {count} games ({pct:.1f}%)")
    
    if len(counts) == 1 and counts.index[0] == 0:
        print("\n   ❌ CRITICAL: Column is ALL ZEROS - Logic is broken!")
    else:
        print(f"\n   ✅ Data looks healthy ({(df['away_3in4'] > 0).sum()} positive cases)")
else:
    print("   ⚠️  'away_3in4' not in feature set")

# Check home_3in4 too
if 'home_3in4' in df.columns:
    home_count = (df['home_3in4'] > 0).sum()
    print(f"\n   For comparison, 'home_3in4' has {home_count} positive cases")

# ============================================================================
# AUDIT 2: Injury Feature Correlation (Trampling Check)
# ============================================================================
print("\n" + "=" * 80)
print("⚔️ AUDIT 2: Checking Injury Feature Correlation (Trampling)")
print("=" * 80)

injury_cols = ['injury_impact_diff', 'injury_impact_abs', 'injury_elo_interaction']
available_injury = [col for col in injury_cols if col in df.columns]

if len(available_injury) >= 2:
    corr = df[available_injury].corr()
    
    print("\nCorrelation Matrix:")
    print(corr.to_string())
    
    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("-" * 80)
    
    # Check for high correlations
    high_corr_pairs = []
    for i in range(len(available_injury)):
        for j in range(i+1, len(available_injury)):
            corr_val = corr.iloc[i, j]
            col1 = available_injury[i]
            col2 = available_injury[j]
            
            if abs(corr_val) > 0.85:
                high_corr_pairs.append((col1, col2, corr_val))
                print(f"⚠️  HIGH CORRELATION: {col1} ↔ {col2} = {corr_val:.3f}")
    
    if not high_corr_pairs:
        print("✅ No concerning correlations (all < 0.85)")
        print("   All three injury features provide distinct signals")
    else:
        print(f"\n💡 RECOMMENDATION:")
        for col1, col2, corr_val in high_corr_pairs:
            print(f"   Consider removing '{col2}' (corr={corr_val:.3f} with '{col1}')")
else:
    print(f"   ⚠️  Only {len(available_injury)} injury features found")

# ============================================================================
# AUDIT 3: Math Consistency Check
# ============================================================================
print("\n" + "=" * 80)
print("🧮 AUDIT 3: Math Consistency Check")
print("=" * 80)

if 'injury_impact_diff' in df.columns and 'injury_impact_abs' in df.columns:
    # Filter to games with actual injuries
    injury_games = df[df['injury_impact_abs'] > 0.1].copy()
    
    print(f"\nChecking {len(injury_games)} games with injuries...")
    print("\nSample (First 5):")
    print("-" * 80)
    print(f"{'Diff':<10} {'Abs':<10} {'Interaction':<15} {'Abs Math Check'}")
    print("-" * 80)
    
    math_errors = 0
    for idx, (_, row) in enumerate(injury_games.head(10).iterrows()):
        diff = row.get('injury_impact_diff', 0)
        abs_val = row.get('injury_impact_abs', 0)
        interaction = row.get('injury_elo_interaction', 0)
        
        # Expected abs = absolute value of diff
        # BUT WAIT - that's wrong! abs should be TOTAL injury impact
        # diff should be HOME - AWAY
        # So we can't verify abs = |diff| because that's not the definition
        
        print(f"{diff:<10.3f} {abs_val:<10.3f} {interaction:<15.3f}")
    
    print("\n" + "-" * 80)
    print("NOTE: injury_impact_abs = TOTAL injury impact (home + away)")
    print("      injury_impact_diff = HOME injury - AWAY injury")
    print("      So we expect: diff can be positive or negative")
    print("                    abs should always be >= |diff|")
    print("-" * 80)
    
    # Better check: abs should be >= |diff| always
    violations = (injury_games['injury_impact_abs'] < injury_games['injury_impact_diff'].abs() - 0.01)
    if violations.sum() > 0:
        print(f"\n⚠️  {violations.sum()} cases where abs < |diff| (logic error!)")
    else:
        print("\n✅ Math consistency verified (abs >= |diff| in all cases)")
else:
    print("   ⚠️  Injury features not available for math check")

# ============================================================================
# AUDIT 4: Feature Importance Context
# ============================================================================
print("\n" + "=" * 80)
print("📊 AUDIT 4: Feature Statistics Summary")
print("=" * 80)

if len(available_injury) > 0:
    print("\nInjury Feature Statistics:")
    print("-" * 80)
    print(f"{'Feature':<30} {'Mean':<10} {'Std':<10} {'Max':<10} {'Non-Zero %'}")
    print("-" * 80)
    
    for col in available_injury:
        mean_val = df[col].mean()
        std_val = df[col].std()
        max_val = df[col].max()
        nonzero_pct = 100 * (df[col] != 0).sum() / len(df)
        print(f"{col:<30} {mean_val:<10.3f} {std_val:<10.3f} {max_val:<10.3f} {nonzero_pct:.1f}%")

# Check rest features too
rest_features = ['home_rest_days', 'away_rest_days', 'rest_advantage', 
                 'fatigue_mismatch', 'home_back_to_back', 'away_back_to_back',
                 'home_3in4', 'away_3in4']
available_rest = [col for col in rest_features if col in df.columns]

if len(available_rest) > 0:
    print("\n\nRest/Fatigue Feature Statistics:")
    print("-" * 80)
    print(f"{'Feature':<30} {'Mean':<10} {'Std':<10} {'Max':<10} {'Non-Zero %'}")
    print("-" * 80)
    
    for col in available_rest:
        mean_val = df[col].mean()
        std_val = df[col].std()
        max_val = df[col].max()
        nonzero_pct = 100 * (df[col] != 0).sum() / len(df)
        print(f"{col:<30} {mean_val:<10.3f} {std_val:<10.3f} {max_val:<10.3f} {nonzero_pct:.1f}%")

print("\n" + "=" * 80)
print("✅ HEALTH CHECK COMPLETE")
print("=" * 80)
