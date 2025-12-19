"""
Complete feature correlation analysis
"""

import pandas as pd
import numpy as np

print("\n" + "="*70)
print("COMPLETE FEATURE CORRELATION ANALYSIS")
print("="*70)

# Load data
df = pd.read_csv('data/training_data_with_temporal_features.csv')

exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win', 
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols]

print(f"\nAnalyzing {len(feature_cols)} features...")

# Compute correlation matrix
corr_matrix = X.corr()

# Get all pairs with their correlations
all_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        all_pairs.append({
            'feature1': corr_matrix.columns[i],
            'feature2': corr_matrix.columns[j],
            'correlation': corr_matrix.iloc[i, j]
        })

# Sort by absolute correlation
all_pairs_df = pd.DataFrame(all_pairs)
all_pairs_df['abs_corr'] = all_pairs_df['correlation'].abs()
all_pairs_df = all_pairs_df.sort_values('abs_corr', ascending=False).reset_index(drop=True)

# Save full correlation pairs
all_pairs_df.to_csv('models/all_feature_correlations.csv', index=False)
print(f"Saved: models/all_feature_correlations.csv")

# Display by correlation strength
print("\n" + "="*80)
print("HIGH CORRELATIONS (|r| > 0.8)")
print("="*80)
high = all_pairs_df[all_pairs_df['abs_corr'] > 0.8]
print(f"\n{len(high)} pairs found:\n")
print(f"{'Feature 1':<35} {'Feature 2':<35} {'Correlation':>12}")
print("-"*80)
for _, row in high.iterrows():
    print(f"{row['feature1']:<35} {row['feature2']:<35} {row['correlation']:>12.4f}")

print("\n" + "="*80)
print("MODERATE-HIGH CORRELATIONS (0.6 < |r| <= 0.8)")
print("="*80)
mod_high = all_pairs_df[(all_pairs_df['abs_corr'] > 0.6) & (all_pairs_df['abs_corr'] <= 0.8)]
print(f"\n{len(mod_high)} pairs found:\n")
print(f"{'Feature 1':<35} {'Feature 2':<35} {'Correlation':>12}")
print("-"*80)
for _, row in mod_high.iterrows():
    print(f"{row['feature1']:<35} {row['feature2']:<35} {row['correlation']:>12.4f}")

print("\n" + "="*80)
print("MODERATE CORRELATIONS (0.4 < |r| <= 0.6)")
print("="*80)
mod = all_pairs_df[(all_pairs_df['abs_corr'] > 0.4) & (all_pairs_df['abs_corr'] <= 0.6)]
print(f"\n{len(mod)} pairs found:\n")
print(f"{'Feature 1':<35} {'Feature 2':<35} {'Correlation':>12}")
print("-"*80)
for _, row in mod.iterrows():
    print(f"{row['feature1']:<35} {row['feature2']:<35} {row['correlation']:>12.4f}")

print("\n" + "="*80)
print("CORRELATION SUMMARY")
print("="*80)
print(f"\nTotal feature pairs: {len(all_pairs_df):,}")
print(f"  Very High (|r| > 0.8):  {len(high)}")
print(f"  High (0.6-0.8):         {len(mod_high)}")
print(f"  Moderate (0.4-0.6):     {len(mod)}")
print(f"  Low (< 0.4):            {len(all_pairs_df) - len(high) - len(mod_high) - len(mod)}")

print("\n" + "="*80)
