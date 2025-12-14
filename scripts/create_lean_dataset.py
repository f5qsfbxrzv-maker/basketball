"""
Create lean dataset based on feature importance audit
- Remove 18 noisy features (correlation < 0.015)
- Remove 3 redundant features (perfect multicollinearity)
- Add 3 new interaction features
- Final: 18 features (vs 36) - 50% reduction
"""

import pandas as pd
import numpy as np

print("="*70)
print("CREATING LEAN DATASET - KILL THE NOISE")
print("="*70)

# Load original data
print("\n1. Loading original 36-feature dataset...")
df = pd.read_csv("data/training_data_with_features.csv")
print(f"   Total games: {len(df):,}")
print(f"   Original features: 36")

# Define metadata columns to preserve
metadata_cols = ['date', 'game_id', 'home_team', 'away_team', 'season']
target_cols = ['target_spread', 'target_spread_cover', 'target_moneyline_win', 
               'target_game_total', 'target_over_under']

# THE KILL LIST - Features with r < 0.015 or redundant
print("\n2. Removing noisy features...")
kill_list = [
    # REST FALLACY (r < 0.005)
    'rest_advantage',           # r=0.0015
    'home_rest_days',          # r=0.0053
    'away_rest_days',          # r=0.0047 (also redundant with home_rest_days)
    'fatigue_mismatch',        # r=0.0149
    'home_3in4',               # r=0.0130
    
    # INJURY MIRAGE (r < 0.012)
    'injury_impact_abs',       # r=0.0023
    'star_mismatch',           # r=0.0089 (also redundant with injury_impact_diff)
    'injury_impact_diff',      # r=0.0118
    'away_star_missing',       # r=0.0082
    'home_star_missing',       # r=0.0030
    
    # FOUL NOISE (r < 0.015)
    'total_foul_environment',  # r=0.0118
    'ewma_foul_synergy_home',  # r=0.0131
    'ewma_foul_synergy_away',  # r=0.0047
    'away_ewma_fta_rate',      # r=0.0047 (PERFECT correlation with ewma_foul_synergy_away)
    
    # CHAOS NOISE
    'ewma_net_chaos',          # r=0.0095
    
    # ORB WEAK SIGNAL
    'home_orb',                # r=0.0344 - marginal
    'away_orb',                # r=0.0260 - weak
    
    # TOV WEAK
    'away_ewma_tov_pct',       # r=0.0403 - replaced by interaction
]

print(f"   Features to remove: {len(kill_list)}")
for feat in kill_list:
    print(f"     ✗ {feat}")

# Features to KEEP (top 18 by importance + correlation)
keep_features = [
    'ewma_efg_diff',           # #1 importance (4.35%), r=0.098
    'off_elo_diff',            # #2 importance (3.86%), r=0.077
    'def_elo_diff',            # #3 importance (3.24%), r=0.049
    'altitude_game',           # #4 importance (3.18%), r=0.042
    'away_back_to_back',       # #5 importance (3.17%), r=0.053
    'ewma_pace_diff',          # #7 importance (3.12%), r=0.046
    'home_composite_elo',      # #8 importance (3.02%)
    'ewma_tov_diff',           # #9 importance (2.94%), r=0.051
    'home_ewma_3p_pct',        # #10 importance (2.92%), r=0.055
    'injury_shock_diff',       # #11 importance (2.91%), r=0.045
    'away_3in4',               # #12 importance (2.88%), r=0.041
    'ewma_vol_3p_diff',        # #13 importance (2.80%)
    'injury_shock_away',       # #14 importance (2.79%), r=0.036
    'ewma_chaos_home',         # #15 importance (2.76%)
    'home_back_to_back',       # #16 importance (2.76%)
    'ewma_orb_diff',           # #18 importance (2.70%), r=0.044
    'away_ewma_3p_pct',        # #26 but r=0.044 (useful for diversity)
    'injury_shock_home',       # #21 importance (2.63%)
]

print(f"\n3. Keeping core features: {len(keep_features)}")
for i, feat in enumerate(keep_features, 1):
    print(f"     ✓ {i:2d}. {feat}")

# Create lean dataframe
print("\n4. Creating lean dataframe...")
lean_df = df[metadata_cols + keep_features + target_cols].copy()

# Add 3 NEW INTERACTION FEATURES
print("\n5. Engineering interaction features...")

# Interaction 1: efficiency_x_pace (blow-out potential)
lean_df['efficiency_x_pace'] = df['ewma_efg_diff'] * df['ewma_pace_diff']
print("     ✓ efficiency_x_pace = ewma_efg_diff * ewma_pace_diff")
print("       (Efficient teams playing fast = blowouts)")

# Interaction 2: tired_altitude (schedule loss)
lean_df['tired_altitude'] = df['altitude_game'] * df['away_back_to_back']
print("     ✓ tired_altitude = altitude_game * away_back_to_back")
print("       (Altitude + tired away team = schedule loss)")

# Interaction 3: form_x_defense (defense travels)
lean_df['form_x_defense'] = df['def_elo_diff'] * df['ewma_tov_diff']
print("     ✓ form_x_defense = def_elo_diff * ewma_tov_diff")
print("       (Defensive advantage + forcing turnovers)")

# Final feature count
final_features = keep_features + ['efficiency_x_pace', 'tired_altitude', 'form_x_defense']
print(f"\n6. Final feature count: {len(final_features)}")
print(f"   - Core features: {len(keep_features)}")
print(f"   - Interaction features: 3")
print(f"   - Reduction: {36 - len(final_features)} features removed ({(36 - len(final_features))/36*100:.0f}%)")

# Verify no missing values
print("\n7. Data quality check...")
missing_counts = lean_df[final_features].isnull().sum()
if missing_counts.sum() == 0:
    print("     ✓ No missing values")
else:
    print("     ⚠ Missing values detected:")
    print(missing_counts[missing_counts > 0])

# Save lean dataset
output_path = "data/training_data_lean.csv"
lean_df.to_csv(output_path, index=False)
print(f"\n8. Saved lean dataset to: {output_path}")
print(f"   Size: {len(lean_df):,} games x {len(final_features)} features")

# Print feature list for reference
print("\n" + "="*70)
print("LEAN FEATURE SET (21 features)")
print("="*70)
print("\nCORE FEATURES (18):")
for i, feat in enumerate(keep_features, 1):
    print(f"  {i:2d}. {feat}")

print("\nINTERACTION FEATURES (3):")
print("  19. efficiency_x_pace")
print("  20. tired_altitude")
print("  21. form_x_defense")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original features:  36")
print(f"Removed (noise):    {len(kill_list)}")
print(f"Kept (signal):      {len(keep_features)}")
print(f"Added (interactions): 3")
print(f"Final features:     {len(final_features)}")
print(f"Speed improvement:  ~{(36 - len(final_features))/36*100:.0f}%")
print("\nNEXT STEPS:")
print("  1. Train model on lean dataset")
print("  2. Compare AUC: lean vs full (expect similar or better)")
print("  3. If AUC maintained, deploy lean model for production")
print("="*70)
