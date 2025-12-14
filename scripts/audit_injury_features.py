"""
Forensic audit of injury feature data quality
Checks for sparse data, merge failures, and correlation with outcomes
"""
import pandas as pd
import numpy as np

print("="*80)
print("INJURY DATA FORENSICS")
print("="*80)

# Load cached training data
df = pd.read_csv('data/processed/training_features_30.csv')

print(f"\nDataset: {len(df)} games from {df['game_date'].min()} to {df['game_date'].max()}")

# Injury columns in our whitelist
injury_cols = ['injury_impact_diff', 'injury_impact_abs']

print("\n" + "="*80)
print("INJURY FEATURE AUDIT")
print("="*80)

for col in injury_cols:
    if col in df.columns:
        print(f"\n📊 Feature: {col}")
        print("-" * 60)
        
        # 1. Non-zero count (signal presence)
        non_zeros = (df[col] != 0.0).sum()
        pct_impacted = (non_zeros / len(df)) * 100
        
        # 2. Unique values (check for variance)
        unique_vals = df[col].nunique()
        
        # 3. Statistics
        print(f"   Games with Impact:      {non_zeros:,} / {len(df):,} ({pct_impacted:.1f}%)")
        print(f"   Unique Values:          {unique_vals:,}")
        print(f"   Mean Value:             {df[col].mean():.4f}")
        print(f"   Std Deviation:          {df[col].std():.4f}")
        print(f"   Min/Max:                {df[col].min():.4f} / {df[col].max():.4f}")
        
        # 4. Distribution percentiles
        print(f"   25th Percentile:        {df[col].quantile(0.25):.4f}")
        print(f"   50th Percentile:        {df[col].quantile(0.50):.4f}")
        print(f"   75th Percentile:        {df[col].quantile(0.75):.4f}")
        print(f"   95th Percentile:        {df[col].quantile(0.95):.4f}")
        
        # 5. Correlation with winning
        corr = df[col].corr(df['home_won'])
        print(f"   Correlation w/ Win:     {corr:.4f}")
        
        # 6. Value range breakdown
        print(f"\n   Value Distribution:")
        if col == 'injury_impact_diff':
            # Differential feature (can be negative)
            bins = [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf]
            labels = ['Major Away Adv', 'Minor Away Adv', 'Neutral', 'Minor Home Adv', 'Major Home Adv']
        else:
            # Absolute feature (always positive)
            bins = [0, 0.01, 0.1, 0.3, 0.5, np.inf]
            labels = ['No Impact', 'Minor', 'Moderate', 'Significant', 'Severe']
        
        df['temp_bin'] = pd.cut(df[col], bins=bins, labels=labels)
        for label in labels:
            count = (df['temp_bin'] == label).sum()
            pct = (count / len(df)) * 100
            print(f"      {label:20s}: {count:5d} ({pct:5.1f}%)")
        
        # Critical checks
        print(f"\n   🔍 VALIDITY CHECKS:")
        if unique_vals <= 1:
            print(f"      🚨 CRITICAL: No variance! Feature is flat.")
        elif pct_impacted < 5:
            print(f"      ⚠️  WARNING: <5% non-zero values. Possible merge failure.")
        elif pct_impacted < 15:
            print(f"      ⚠️  CAUTION: <15% impact rate. Check if realistic.")
        else:
            print(f"      ✅ Variance confirmed ({pct_impacted:.1f}% impact rate)")
        
        if abs(corr) < 0.01:
            print(f"      ⚠️  WARNING: Near-zero correlation with target")
        else:
            print(f"      ✅ Correlation detected: {corr:.4f}")
            
    else:
        print(f"\n❌ MISSING: {col} not found in dataset")

# Check specific high-injury games
print("\n" + "="*80)
print("SAMPLE HIGH-IMPACT GAMES")
print("="*80)

if 'injury_impact_abs' in df.columns:
    # Top 10 most impacted games
    top_injured = df.nlargest(10, 'injury_impact_abs')[
        ['game_date', 'home_team', 'away_team', 'injury_impact_abs', 'injury_impact_diff', 'home_won']
    ]
    
    print("\nTop 10 Most Injury-Impacted Games:")
    print("-" * 80)
    for idx, row in top_injured.iterrows():
        result = "HOME WIN" if row['home_won'] == 1 else "AWAY WIN"
        print(f"{row['game_date'][:10]} | {row['home_team']:3s} vs {row['away_team']:3s} | "
              f"Abs: {row['injury_impact_abs']:6.3f} | Diff: {row['injury_impact_diff']:7.3f} | {result}")

# Overall feature importance context
print("\n" + "="*80)
print("FEATURE IMPORTANCE CONTEXT")
print("="*80)

import joblib
model = joblib.load('models/xgboost_optuna_uncalibrated.pkl')
feature_importance = pd.DataFrame({
    'feature': model.feature_names_in_,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
for i, (idx, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    marker = "🏥" if 'inj' in row['feature'].lower() else "  "
    print(f"{i:2d}. {marker} {row['feature']:30s} {row['importance']:.4f}")

print("\nInjury Feature Rankings:")
injury_ranks = feature_importance[feature_importance['feature'].str.contains('inj', case=False)]
for idx, row in injury_ranks.iterrows():
    rank = feature_importance.index.get_loc(idx) + 1
    print(f"   #{rank:2d}. {row['feature']:30s} {row['importance']:.4f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Final diagnosis
injury_impact_pct = (df['injury_impact_abs'] != 0).sum() / len(df) * 100 if 'injury_impact_abs' in df.columns else 0

if injury_impact_pct < 5:
    print("\n🚨 CRITICAL FAILURE: Injury data has <5% non-zero values")
    print("   Likely cause: Name matching failure in merge")
    print("   Action: Check player name normalization in injury pipeline")
elif injury_impact_pct < 15:
    print("\n⚠️  LOW SIGNAL: Injury data sparse (<15% impact)")
    print("   Possible causes:")
    print("   1. Name matching issues (some players not merged)")
    print("   2. Injury data source incomplete")
    print("   3. Definition of 'significant injury' too narrow")
elif injury_impact_pct < 40:
    print("\n✅ REASONABLE: 15-40% of games have injury impact")
    print("   This is realistic for NBA (injuries are common but not universal)")
    print("   Model ranking at #21-23 suggests other factors dominate")
    print("   (Rest/fatigue features may capture some injury-adjacent signal)")
else:
    print("\n✅ HIGH SIGNAL: >40% of games show injury impact")
    print("   Data quality appears good")
    print("   Low ranking suggests model finds other signals more predictive")

print("\n" + "="*80)
