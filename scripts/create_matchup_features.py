"""
Create Schematic Matchup Features from AVAILABLE Stats
Uses the existing ELO and EWMA features to create style clash indicators.

Strategy: The dataset has ELO (offensive/defensive) and EWMA diffs (EFG, pace, TOV, ORB, 3P).
We can create matchup features showing when team strengths/weaknesses collide.
"""

import pandas as pd
import numpy as np

print("="*70)
print("CREATING SCHEMATIC MATCHUP FEATURES FROM AVAILABLE STATS")
print("="*70)

# Load cleaned original dataset
print("\n1. Loading cleaned original dataset...")
df = pd.read_csv("data/training_data_with_features_cleaned.csv")
print(f"   Games: {len(df):,}")

print("\n2. Creating matchup features...")

# MATCHUP 1: Offensive Power vs Defensive Weakness
# When offensive ELO diff is large AND defensive ELO diff favors offense
# This captures "elite offense vs weak defense" scenarios
df['off_vs_def_mismatch'] = df['off_elo_diff'] * (-df['def_elo_diff'])  # Negative def_elo_diff = weak defense
print(f"   ✓ off_vs_def_mismatch: Captures offensive power exploiting defensive weakness")
print(f"     Mean={df['off_vs_def_mismatch'].mean():.1f}, Std={df['off_vs_def_mismatch'].std():.1f}")

# MATCHUP 2: Pace Clash 
# Difference in preferred pace can create tactical advantage
# Home team fast pace + away team slow = home edge (dictate tempo)
home_fast = (df['ewma_pace_diff'] > 0).astype(float)  # Home prefers faster
away_slow = (df['ewma_pace_diff'] > 0).astype(float)  # Same = away is slower
df['pace_control_advantage'] = home_fast * df['ewma_pace_diff'].abs()
print(f"   ✓ pace_control_advantage: Home team dictates tempo")
print(f"     Mean={df['pace_control_advantage'].mean():.2f}, Std={df['pace_control_advantage'].std():.2f}")

# MATCHUP 3: Three-Point Game Style Clash
# High 3P% team vs low 3P% opponent = style advantage
# Uses ewma_vol_3p_diff (volume difference) and away/home 3P%
if 'away_ewma_3p_pct' in df.columns and 'home_ewma_3p_pct' in df.columns:
    away_good_3pt = (df['away_ewma_3p_pct'] > 0.36).astype(float)  # Good 3PT shooter
    home_bad_3pt = (df['home_ewma_3p_pct'] < 0.34).astype(float)  # Poor 3PT defense proxy
    df['three_point_exploit'] = away_good_3pt * home_bad_3pt * df['ewma_vol_3p_diff'].abs()
    print(f"   ✓ three_point_exploit: Good 3PT team vs opponent weakness")
    print(f"     Mean={df['three_point_exploit'].mean():.3f}, Std={df['three_point_exploit'].std():.3f}")
else:
    df['three_point_exploit'] = 0.0
    print(f"   ⚠️ three_point_exploit: Placeholder (no 3P% columns)")

# MATCHUP 4: Turnover Battle
# Low TOV team (careful) vs high TOV opponent (forces turnovers) = clash
# Negative ewma_tov_diff = away turns it over more
df['turnover_pressure'] = df['ewma_tov_diff'].abs() * 10  # Scale up for visibility
print(f"   ✓ turnover_pressure: Turnover rate mismatch")
print(f"     Mean={df['turnover_pressure'].mean():.2f}, Std={df['turnover_pressure'].std():.2f}")

# MATCHUP 5: Rebounding Battle
# Strong offensive rebounding vs weak defensive rebounding
# Positive ewma_orb_diff = home gets more offensive boards
df['rebounding_clash'] = df['ewma_orb_diff'].abs() * 5  # Scale for visibility
print(f"   ✓ rebounding_clash: Offensive rebounding mismatch")
print(f"     Mean={df['rebounding_clash'].mean():.2f}, Std={df['rebounding_clash'].std():.2f}")

print("\n3. Validating matchup features...")
matchup_features = ['off_vs_def_mismatch', 'pace_control_advantage', 'three_point_exploit', 
                     'turnover_pressure', 'rebounding_clash']

for feat in matchup_features:
    non_zero_pct = (df[feat] != 0).sum() / len(df) * 100
    print(f"   {feat:25s}: {df[feat].min():8.2f} to {df[feat].max():8.2f} ({non_zero_pct:5.1f}% non-zero)")

# Calculate correlations with target
print("\n4. Correlations with target (spread cover)...")
for feat in matchup_features:
    corr = df[feat].corr(df['target_spread_cover'])
    print(f"   {feat:25s}: r={corr:+.4f}")

# Save
output_path = "data/training_data_with_matchups.csv"
df.to_csv(output_path, index=False)

total_features = len([c for c in df.columns if c not in ['date','game_id','home_team','away_team','season','target_spread','target_spread_cover','target_moneyline_win','target_game_total','target_over_under','target_home_cover','target_over']])

print(f"\n5. Saved: {output_path}")
print(f"   Total features: {total_features} (36 original + 5 matchup = 41)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("CREATED 5 SCHEMATIC MATCHUP FEATURES:")
print("  1. off_vs_def_mismatch - Offensive power exploiting defensive weakness")
print("  2. pace_control_advantage - Home team dictating tempo")
print("  3. three_point_exploit - Good 3PT shooter vs weak opponent")
print("  4. turnover_pressure - Turnover rate mismatch")
print("  5. rebounding_clash - Offensive/defensive rebounding battle")
print("\nThese leverage existing ELO + EWMA stats to capture style clashes.")
print("\nNEXT STEPS:")
print("  1. Train model with 41 features (original 36 + matchups 5)")
print("  2. Compare AUC vs 36-feature baseline")
print("  3. Check if matchup features get importance >2%")
print("="*70)
