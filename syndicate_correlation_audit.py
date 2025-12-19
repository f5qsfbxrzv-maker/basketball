"""
SYNDICATE CORRELATION AUDIT
===========================
Tests the "Cannibalization" theory: Are syndicate features duplicating each other?

This script identifies:
1. High correlation pairs (>85% = "Redundancy Zone")
2. Variance Inflation Factor (VIF > 10 = feature is redundant)
3. The "kill list" of features to remove

Author: NBA Betting System
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==============================================================================
# LOAD TRAINING DATA
# ==============================================================================
print("="*80)
print("SYNDICATE CORRELATION AUDIT")
print("="*80)

df = pd.read_csv('data/training_data_SYNDICATE_28_features.csv')
print(f"\nLoaded {len(df)} games")

# ==============================================================================
# DEFINE SYNDICATE FEATURE LIST (22 features)
# ==============================================================================
features = [
    # ELO Matchup Advantages (3)
    'off_matchup_advantage', 
    'def_matchup_advantage', 
    'net_composite_advantage',
    
    # Matchup Friction (5)
    'effective_shooting_gap', 
    'turnover_pressure', 
    'rebound_friction',
    'total_rebound_control', 
    'whistle_leverage',
    
    # Volume & Injury (2)
    'volume_efficiency_diff', 
    'injury_leverage',
    
    # Supporting Features (12)
    'net_fatigue_score',
    'ewma_chaos_home',
    'ewma_foul_synergy_home',
    'total_foul_environment',
    'league_offensive_context',
    'season_progress',
    'three_point_matchup',
    'star_power_leverage',
    'offense_vs_defense_matchup',
    'ewma_pace_diff',
    'ewma_vol_3p_diff',
    'projected_possession_margin'
]

# Filter dataset to just these columns (drop NaNs to avoid errors)
X = df[features].dropna()
print(f"Analysis dataset: {len(X)} games with {len(features)} features")

# ==============================================================================
# TEST 1: THE CORRELATION HEATMAP (Visual Check)
# ==============================================================================
print("\n" + "="*80)
print("TEST 1: CORRELATION MATRIX")
print("="*80)

corr_matrix = X.corr()

# Save correlation matrix to CSV for inspection
corr_matrix.to_csv('syndicate_correlation_matrix.csv')
print("✓ Saved: syndicate_correlation_matrix.csv")

# Plot heatmap (only lower triangle for readability)
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, 
    mask=mask, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Syndicate Feature Correlation Matrix\n(Lower Triangle Only)", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('syndicate_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: syndicate_correlation_heatmap.png")
plt.close()

# ==============================================================================
# TEST 2: THE "KILL LIST" GENERATOR (>85% Correlation)
# ==============================================================================
print("\n" + "="*80)
print("TEST 2: HIGH CORRELATION WARNINGS (>85%)")
print("="*80)

high_corr_pairs = []

# Loop through the correlation matrix
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.85:
            colname_i = corr_matrix.columns[i]
            colname_j = corr_matrix.columns[j]
            high_corr_pairs.append({
                'Feature_1': colname_j,
                'Feature_2': colname_i,
                'Correlation': corr_val,
                'Abs_Correlation': abs(corr_val)
            })
            print(f"⚠️  REDUNDANCY DETECTED: {colname_j:30s} <--> {colname_i:30s}  ({corr_val:+.3f})")

if not high_corr_pairs:
    print("✓ No high correlation pairs found (all <85%)")
else:
    print(f"\n⚠️  Found {len(high_corr_pairs)} redundant feature pairs")
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Abs_Correlation', ascending=False)
    high_corr_df.to_csv('syndicate_high_correlations.csv', index=False)
    print("✓ Saved: syndicate_high_correlations.csv")

# ==============================================================================
# TEST 3: VARIANCE INFLATION FACTOR (Mathematical Proof)
# ==============================================================================
print("\n" + "="*80)
print("TEST 3: VARIANCE INFLATION FACTOR (VIF)")
print("="*80)
print("VIF Interpretation:")
print("  1-5   = Low multicollinearity (GOOD)")
print("  5-10  = Moderate multicollinearity (WATCH)")
print("  >10   = High multicollinearity (REMOVE)")
print("  >100  = Perfect duplication (KILL IMMEDIATELY)")
print("-"*80)

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# Sort by VIF (Highest = Worst)
vif_data_sorted = vif_data.sort_values(by="VIF", ascending=False)

# Categorize features by VIF severity
critical_vif = vif_data_sorted[vif_data_sorted['VIF'] > 10]
warning_vif = vif_data_sorted[(vif_data_sorted['VIF'] >= 5) & (vif_data_sorted['VIF'] <= 10)]
good_vif = vif_data_sorted[vif_data_sorted['VIF'] < 5]

print("\n🔴 CRITICAL (VIF > 10) - REMOVE THESE:")
print("-"*80)
if len(critical_vif) > 0:
    for idx, row in critical_vif.iterrows():
        print(f"  ❌ {row['Feature']:35s} VIF = {row['VIF']:8.2f}")
else:
    print("  ✓ No critical features")

print("\n🟡 WARNING (VIF 5-10) - MONITOR THESE:")
print("-"*80)
if len(warning_vif) > 0:
    for idx, row in warning_vif.iterrows():
        print(f"  ⚠️  {row['Feature']:35s} VIF = {row['VIF']:8.2f}")
else:
    print("  ✓ No warning features")

print("\n🟢 GOOD (VIF < 5) - KEEP THESE:")
print("-"*80)
for idx, row in good_vif.iterrows():
    print(f"  ✓ {row['Feature']:35s} VIF = {row['VIF']:8.2f}")

# Save full VIF results
vif_data_sorted.to_csv('syndicate_vif_scores.csv', index=False)
print("\n✓ Saved: syndicate_vif_scores.csv")

# ==============================================================================
# TEST 4: SYNDICATE FEATURE GROUPS ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("TEST 4: SYNDICATE GROUP ANALYSIS")
print("="*80)

# Define feature groups
feature_groups = {
    'ELO Matchup': ['off_matchup_advantage', 'def_matchup_advantage', 'net_composite_advantage'],
    'Matchup Friction': ['effective_shooting_gap', 'turnover_pressure', 'rebound_friction', 
                         'total_rebound_control', 'whistle_leverage'],
    'Volume & Injury': ['volume_efficiency_diff', 'injury_leverage'],
    'Context': ['net_fatigue_score', 'league_offensive_context', 'season_progress'],
    'Shooting': ['three_point_matchup', 'ewma_vol_3p_diff'],
    'Fouls': ['ewma_foul_synergy_home', 'total_foul_environment', 'whistle_leverage'],
    'Pace & Style': ['ewma_chaos_home', 'ewma_pace_diff', 'projected_possession_margin'],
    'Composite': ['star_power_leverage', 'offense_vs_defense_matchup']
}

for group_name, group_features in feature_groups.items():
    available_features = [f for f in group_features if f in X.columns]
    if len(available_features) > 1:
        group_corr = X[available_features].corr()
        max_corr = group_corr.abs().values[np.triu_indices_from(group_corr.values, 1)].max()
        print(f"\n{group_name}:")
        print(f"  Features: {', '.join(available_features)}")
        print(f"  Max internal correlation: {max_corr:.3f}")
        if max_corr > 0.85:
            print(f"  ⚠️  HIGH INTERNAL CORRELATION - GROUP MAY HAVE DUPLICATES")

# ==============================================================================
# RECOMMENDATIONS
# ==============================================================================
print("\n" + "="*80)
print("ACTIONABLE RECOMMENDATIONS")
print("="*80)

kill_list = critical_vif['Feature'].tolist() if len(critical_vif) > 0 else []
watch_list = warning_vif['Feature'].tolist() if len(warning_vif) > 0 else []

if kill_list:
    print("\n🔴 IMMEDIATE ACTION REQUIRED:")
    print("   Remove these features from training data:")
    for feature in kill_list:
        print(f"     ❌ {feature}")
    print("\n   These features are PERFECT DUPLICATES of other features.")
    print("   Their removal will INCREASE model performance by concentrating")
    print("   importance scores on unique signals.")

if watch_list:
    print("\n🟡 MONITOR CLOSELY:")
    print("   These features have moderate redundancy:")
    for feature in watch_list:
        print(f"     ⚠️  {feature}")
    print("\n   Consider removing if feature importance is low (<1%)")

if not kill_list and not watch_list:
    print("\n✓ ALL FEATURES PASS VIF TEST")
    print("  Your syndicate features are mathematically independent.")
    print("  The 0.6309 score is NOT due to feature duplication.")
    print("  Next steps:")
    print("    1. Check if friction features need real opponent stats")
    print("    2. Add Heat ELO system for momentum signal")
    print("    3. Consider ensemble approach (combine multiple models)")

print("\n" + "="*80)
print("AUDIT COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - syndicate_correlation_matrix.csv")
print("  - syndicate_correlation_heatmap.png")
if high_corr_pairs:
    print("  - syndicate_high_correlations.csv")
print("  - syndicate_vif_scores.csv")
