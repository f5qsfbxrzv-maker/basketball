"""
Comprehensive feature importance analysis for the pruned model
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

# Load feature importance
df = pd.read_csv("output/feature_importance_pruned.csv")

print("="*70)
print("FEATURE IMPORTANCE ANALYSIS - PRUNED MODEL (31 FEATURES)")
print("="*70)

# Overall stats
print(f"\n📊 OVERALL STATS:")
print(f"   Total features: {len(df)}")
print(f"   Mean importance: {df['importance'].mean():.4f}")
print(f"   Std importance: {df['importance'].std():.4f}")
print(f"   Max importance: {df['importance'].max():.4f}")
print(f"   Min importance: {df['importance'].min():.4f}")

# Top 10 features
print(f"\n🏆 TOP 10 FEATURES:")
top10 = df.nlargest(10, 'importance')
for idx, (_, row) in enumerate(top10.iterrows(), 1):
    print(f"   {idx:2d}. {row['feature']:30s} {row['importance']:.4f}")

# Bottom 5 features
print(f"\n⚠️ BOTTOM 5 FEATURES:")
bottom5 = df.nsmallest(5, 'importance')
for idx, (_, row) in enumerate(bottom5.iterrows(), 1):
    print(f"   {idx:2d}. {row['feature']:30s} {row['importance']:.4f}")

# Categorize features
def categorize_feature(name):
    if 'injury' in name:
        return 'Injury Context'
    elif any(x in name for x in ['rest', 'fatigue', 'back_to_back', '3in4']):
        return 'Rest/Fatigue'
    elif 'elo' in name:
        return 'ELO Engine'
    elif 'ewma' in name and 'diff' in name:
        return 'EWMA Diffs'
    elif 'ewma' in name and ('3p' in name or 'tov' in name or 'fta' in name):
        return 'EWMA Baselines'
    elif 'orb' in name:
        return 'Rebounding'
    elif 'foul' in name:
        return 'Foul Synergy'
    elif 'chaos' in name:
        return 'Chaos Metrics'
    elif 'altitude' in name:
        return 'Altitude'
    else:
        return 'Other'

df['category'] = df['feature'].apply(categorize_feature)

# Category-wise importance
print(f"\n📈 IMPORTANCE BY CATEGORY:")
category_importance = df.groupby('category').agg({
    'importance': ['sum', 'mean', 'count']
}).round(4)
category_importance.columns = ['Total', 'Avg', 'Count']
category_importance = category_importance.sort_values('Total', ascending=False)
print(category_importance.to_string())

# Injury features analysis
print(f"\n🏥 INJURY FEATURES ANALYSIS:")
injury_features = df[df['category'] == 'Injury Context']
if len(injury_features) > 0:
    print(f"   Total injury features: {len(injury_features)}")
    print(f"   Combined importance: {injury_features['importance'].sum():.4f}")
    print(f"   Average importance: {injury_features['importance'].mean():.4f}")
    print(f"   Rank in top features:")
    for _, row in injury_features.iterrows():
        rank = (df['importance'] > row['importance']).sum() + 1
        print(f"      {row['feature']:30s} Rank #{rank:2d} ({row['importance']:.4f})")
    
    # Compare to other categories
    injury_total = injury_features['importance'].sum()
    total_importance = df['importance'].sum()
    injury_pct = (injury_total / total_importance) * 100
    print(f"\n   Injury contribution: {injury_pct:.1f}% of total importance")
    
    if injury_pct > 15:
        print(f"   ✅ STRONG SIGNAL - Injury features are highly influential")
    elif injury_pct > 8:
        print(f"   ⚠️ MODERATE SIGNAL - Injury features contribute meaningfully")
    else:
        print(f"   ❌ WEAK SIGNAL - Injury features have limited impact")
else:
    print("   ⚠️ No injury features found in top features")

# EWMA vs Static comparison
print(f"\n📊 EWMA DIFFS vs BASELINES:")
ewma_diffs = df[df['category'] == 'EWMA Diffs']
ewma_baselines = df[df['category'] == 'EWMA Baselines']
print(f"   EWMA Diffs: {ewma_diffs['importance'].sum():.4f} (avg: {ewma_diffs['importance'].mean():.4f})")
print(f"   EWMA Baselines: {ewma_baselines['importance'].sum():.4f} (avg: {ewma_baselines['importance'].mean():.4f})")

# Rest/Fatigue analysis
print(f"\n😴 REST/FATIGUE FEATURES:")
rest_features = df[df['category'] == 'Rest/Fatigue']
print(f"   Total features: {len(rest_features)}")
print(f"   Combined importance: {rest_features['importance'].sum():.4f}")
print(f"   Top rest/fatigue feature: {rest_features.nlargest(1, 'importance')['feature'].values[0]}")

# ELO analysis
print(f"\n🎯 ELO ENGINE:")
elo_features = df[df['category'] == 'ELO Engine']
print(f"   Total features: {len(elo_features)}")
print(f"   Combined importance: {elo_features['importance'].sum():.4f}")
for _, row in elo_features.iterrows():
    rank = (df['importance'] > row['importance']).sum() + 1
    print(f"      {row['feature']:30s} Rank #{rank:2d} ({row['importance']:.4f})")

print("\n" + "="*70)
