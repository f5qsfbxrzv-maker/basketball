"""
Create cleaned syndicate dataset by removing redundant features identified by VIF analysis.

REMOVAL LOGIC (based on correlation audit):
- VIF = inf (perfect duplicates): turnover_pressure, projected_possession_margin, rebound_friction
- VIF > 100 (severe redundancy): total_foul_environment (289.13), ewma_foul_synergy_home (169.53), net_composite_advantage (145.62)
- VIF > 10 (high multicollinearity): off_matchup_advantage (72.37), ewma_chaos_home (28.15)

KEEP LOGIC:
- VIF < 10: 13 features with low/moderate multicollinearity
- Also removing: total_rebound_control (0% importance, not in VIF due to missing data)

Result: 22 features → 14 features (36% reduction)
"""

import pandas as pd
import os

# Define features to remove based on VIF > 10 analysis
REDUNDANT_FEATURES = [
    # VIF = inf (perfect duplicates)
    'turnover_pressure',
    'projected_possession_margin', 
    'rebound_friction',
    
    # VIF > 100 (severe redundancy)
    'total_foul_environment',
    'ewma_foul_synergy_home',
    'net_composite_advantage',
    
    # VIF > 10 (high multicollinearity)
    'off_matchup_advantage',
    'ewma_chaos_home',
    
    # 0% importance + likely redundant
    'total_rebound_control'
]

# Load original syndicate dataset
print("="*80)
print("CREATING CLEANED SYNDICATE DATASET")
print("="*80)

input_file = 'data/training_data_SYNDICATE_28_features.csv'
output_file = 'data/training_data_SYNDICATE_CLEANED_14_features.csv'

if not os.path.exists(input_file):
    print(f"❌ ERROR: {input_file} not found!")
    exit(1)

df = pd.read_csv(input_file)
print(f"\n✓ Loaded {len(df)} games from {input_file}")
print(f"  Original features: {len(df.columns)} total columns")

# Identify feature columns (exclude metadata)
metadata_cols = ['game_id', 'game_date', 'season', 'home_team', 'away_team', 
                 'home_score', 'away_score', 'home_won', 'playoffs', 'target_home_spread']
feature_cols = [col for col in df.columns if col not in metadata_cols]
print(f"  Feature columns: {len(feature_cols)}")

# Check which redundant features actually exist
features_to_remove = [f for f in REDUNDANT_FEATURES if f in df.columns]
missing_features = [f for f in REDUNDANT_FEATURES if f not in df.columns]

print(f"\n🔴 REMOVING {len(features_to_remove)} REDUNDANT FEATURES:")
for feature in features_to_remove:
    print(f"  ❌ {feature}")

if missing_features:
    print(f"\n⚠️  {len(missing_features)} features not found in dataset (already removed?):")
    for feature in missing_features:
        print(f"  ? {feature}")

# Create cleaned dataset
df_cleaned = df.drop(columns=features_to_remove)
cleaned_feature_cols = [col for col in df_cleaned.columns if col not in metadata_cols]

print(f"\n🟢 KEEPING {len(cleaned_feature_cols)} CLEAN FEATURES:")
for feature in sorted(cleaned_feature_cols):
    print(f"  ✓ {feature}")

# Verify no NaN/inf values
nan_counts = df_cleaned[cleaned_feature_cols].isna().sum()
inf_counts = df_cleaned[cleaned_feature_cols].apply(lambda x: (x == float('inf')).sum() + (x == float('-inf')).sum())

if nan_counts.sum() > 0:
    print(f"\n⚠️  WARNING: Found {nan_counts.sum()} NaN values:")
    print(nan_counts[nan_counts > 0])
    
if inf_counts.sum() > 0:
    print(f"\n⚠️  WARNING: Found {inf_counts.sum()} inf values:")
    print(inf_counts[inf_counts > 0])

# Save cleaned dataset
df_cleaned.to_csv(output_file, index=False)
print(f"\n✓ Saved cleaned dataset: {output_file}")
print(f"  Games: {len(df_cleaned)}")
print(f"  Features: {len(cleaned_feature_cols)}")
print(f"  Total columns: {len(df_cleaned.columns)} (including {len(metadata_cols)} metadata)")

print("\n" + "="*80)
print("FEATURE REDUCTION SUMMARY")
print("="*80)
print(f"Original: {len(feature_cols)} features")
print(f"Removed:  {len(features_to_remove)} redundant features ({100*len(features_to_remove)/len(feature_cols):.1f}%)")
print(f"Cleaned:  {len(cleaned_feature_cols)} features")
print(f"\nExpected benefit: Importance consolidation from split features")
print(f"  Example: off_matchup_advantage (30.7%) + net_composite_advantage (26.4%)")
print(f"           → Should consolidate into def_matchup_advantage or other features")
print("="*80)
