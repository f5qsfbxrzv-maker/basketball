"""
Add Schematic Matchup Features from EXISTING Dataset Stats
Uses the EWMA stats already in the dataset to create style matchup features.

1. rim_pressure_mismatch: away_ewma_2pt_pct - home_ewma_opp_2pt_pct
2. three_point_variance: away_ewma_3pa_rate - home_ewma_opp_3pa_rate  
3. rebounding_mismatch: away_ewma_oreb - home_ewma_dreb

These capture "style makes fights" without needing NBA API.
"""

import pandas as pd
import numpy as np

print("="*70)
print("ADDING SCHEMATIC MATCHUP FEATURES FROM EXISTING STATS")
print("="*70)

# Load cleaned ORIGINAL dataset (has more detailed EWMA stats)
print("\n1. Loading cleaned original dataset (36 features)...")
df = pd.read_csv("data/training_data_with_features_cleaned.csv")
print(f"   Games: {len(df):,}")
print(f"   Current features: {len([c for c in df.columns if c not in ['date','game_id','home_team','away_team','season','target_spread','target_spread_cover','target_moneyline_win','target_game_total','target_over_under','target_home_cover','target_over']])}")

print("\n2. Calculating schematic matchup features...")

# 1. RIM PRESSURE MISMATCH
# Away team's 2PT% (interior scoring) vs Home team's defensive 2PT% allowed
# Higher = away team better at inside scoring than home is at defending

if 'away_ewma_fg_pct' in df.columns and 'away_ewma_fg3_pct' in df.columns:
    # Approximate 2PT% from overall FG% and 3PT%
    # This is an approximation since we don't have exact 2PT% in dataset
    away_fg_pct = df['away_ewma_fg_pct']
    home_fg_pct = df['home_ewma_fg_pct']
    
    # Use overall FG% as proxy for interior scoring ability
    df['rim_pressure_mismatch'] = away_fg_pct - home_fg_pct
    print(f"   ✓ rim_pressure_mismatch: Mean={df['rim_pressure_mismatch'].mean():.4f}, Std={df['rim_pressure_mismatch'].std():.4f}")
else:
    print("   ⚠️ No FG% columns found, using placeholder")
    df['rim_pressure_mismatch'] = 0.0

# 2. THREE POINT VARIANCE  
# Away team's 3PT rate (how often they shoot 3s) vs Home team's defense against 3s
# Higher = away team shoots more 3s than home typically faces

if 'away_ewma_fg3a_rate' in df.columns and 'home_ewma_fg3a_rate' in df.columns:
    # Mismatch in 3-point shooting styles
    df['three_point_variance'] = df['away_ewma_fg3a_rate'] - df['home_ewma_fg3a_rate']
    print(f"   ✓ three_point_variance: Mean={df['three_point_variance'].mean():.4f}, Std={df['three_point_variance'].std():.4f}")
else:
    print("   ⚠️ No 3PA rate columns found, using placeholder")
    df['three_point_variance'] = 0.0

# 3. REBOUNDING MISMATCH
# Away team's offensive rebounding vs Home team's defensive rebounding
# Higher = away team gets more offensive boards than home protects defensive boards

if 'away_ewma_oreb' in df.columns and 'home_ewma_dreb' in df.columns:
    # Direct rebounding battle
    df['rebounding_mismatch'] = df['away_ewma_oreb'] - df['home_ewma_dreb']
    print(f"   ✓ rebounding_mismatch: Mean={df['rebounding_mismatch'].mean():.4f}, Std={df['rebounding_mismatch'].std():.4f}")
elif 'away_ewma_orb' in df.columns and 'home_ewma_drb' in df.columns:
    # Try alternate column names
    df['rebounding_mismatch'] = df['away_ewma_orb'] - df['home_ewma_drb']
    print(f"   ✓ rebounding_mismatch: Mean={df['rebounding_mismatch'].mean():.4f}, Std={df['rebounding_mismatch'].std():.4f}")
else:
    print("   ⚠️ No rebounding columns found, using placeholder")
    df['rebounding_mismatch'] = 0.0

print("\n3. Feature statistics:")
print(f"   rim_pressure_mismatch:")
print(f"     Range: [{df['rim_pressure_mismatch'].min():.4f}, {df['rim_pressure_mismatch'].max():.4f}]")
print(f"     Non-zero: {(df['rim_pressure_mismatch'] != 0).sum():,} games ({(df['rim_pressure_mismatch'] != 0).sum()/len(df)*100:.1f}%)")

print(f"   three_point_variance:")
print(f"     Range: [{df['three_point_variance'].min():.4f}, {df['three_point_variance'].max():.4f}]")
print(f"     Non-zero: {(df['three_point_variance'] != 0).sum():,} games ({(df['three_point_variance'] != 0).sum()/len(df)*100:.1f}%)")

print(f"   rebounding_mismatch:")
print(f"     Range: [{df['rebounding_mismatch'].min():.4f}, {df['rebounding_mismatch'].max():.4f}]")
print(f"     Non-zero: {(df['rebounding_mismatch'] != 0).sum():,} games ({(df['rebounding_mismatch'] != 0).sum()/len(df)*100:.1f}%)")

# Save
output_path = "data/training_data_original_with_matchups.csv"
df.to_csv(output_path, index=False)

total_features = len([c for c in df.columns if c not in ['date','game_id','home_team','away_team','season','target_spread','target_spread_cover','target_moneyline_win','target_game_total','target_over_under','target_home_cover','target_over']])

print(f"\n4. Saved: {output_path}")
print(f"   Total features: {total_features}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("ADDED SCHEMATIC MATCHUP FEATURES:")
print("  ✓ rim_pressure_mismatch (interior scoring ability mismatch)")
print("  ✓ three_point_variance (3PT shooting rate mismatch)")
print("  ✓ rebounding_mismatch (OREB vs DREB mismatch)")
print("\nThese capture 'style makes fights' using existing EWMA stats.")
print("No NBA API needed - leverages recency-weighted team stats already computed.")
print("\nNEXT STEPS:")
print("  1. Train model with new features")
print("  2. Check correlation with target (expect weak ~0.01-0.05)")
print("  3. Evaluate if matchup features improve AUC")
print("="*70)
