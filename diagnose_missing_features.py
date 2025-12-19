"""
CRITICAL DIAGNOSTIC: Why are 5 important features showing 0.0000 importance?

Features with -100% change (completely unused by new model):
1. star_mismatch
2. ewma_tov_diff
3. ewma_foul_synergy_home
4. season_progress
5. star_power_leverage

This script will:
1. Check for NaNs in both old and new training data
2. Compare distributions of problematic features
3. Verify data consistency
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("CRITICAL FEATURE DIAGNOSTIC")
print("=" * 80)
print()

# The 5 features that went to 0.0000 importance
PROBLEM_FEATURES = [
    'star_mismatch',
    'ewma_tov_diff',
    'ewma_foul_synergy_home',
    'season_progress',
    'star_power_leverage'
]

# All 25 features for context
ALL_FEATURES = [
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'net_fatigue_score', 'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff', 
    'ewma_orb_diff', 'ewma_vol_3p_diff', 'injury_impact_diff', 'injury_shock_diff',
    'star_mismatch', 'ewma_chaos_home', 'ewma_foul_synergy_home', 
    'total_foul_environment', 'league_offensive_context', 'season_progress',
    'pace_efficiency_interaction', 'projected_possession_margin',
    'three_point_matchup', 'net_free_throw_advantage', 'star_power_leverage',
    'offense_vs_defense_matchup', 'injury_matchup_advantage'
]

print("Loading datasets...")
print()

# Load OLD training data (K=32 ELO)
old_df = pd.read_csv('data/training_data_matchup_with_injury_advantage_FIXED.csv')
print(f"OLD DATA (K=32 ELO): {len(old_df)} games")

# Load NEW training data (K=15 Gold ELO)
new_df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
print(f"NEW DATA (K=15 Gold ELO): {len(new_df)} games")
print()

print("=" * 80)
print("STEP 1: NULL VALUE ANALYSIS")
print("=" * 80)
print()

# Check for NaNs in problem features
print("PROBLEM FEATURES - NULL COUNT:")
print("-" * 80)
print(f"{'Feature':<30} {'Old NaNs':<12} {'New NaNs':<12} {'Status':<20}")
print("-" * 80)

for feature in PROBLEM_FEATURES:
    old_nulls = old_df[feature].isnull().sum() if feature in old_df.columns else "MISSING"
    new_nulls = new_df[feature].isnull().sum() if feature in new_df.columns else "MISSING"
    
    old_pct = f"{old_nulls/len(old_df)*100:.1f}%" if isinstance(old_nulls, int) else "N/A"
    new_pct = f"{new_nulls/len(new_df)*100:.1f}%" if isinstance(new_nulls, int) else "N/A"
    
    status = "OK"
    if new_nulls == "MISSING":
        status = "⚠️ MISSING COLUMN"
    elif isinstance(new_nulls, int) and new_nulls > len(new_df) * 0.5:
        status = "🔴 >50% NULL"
    elif isinstance(new_nulls, int) and new_nulls > len(new_df) * 0.1:
        status = "🟡 >10% NULL"
    
    print(f"{feature:<30} {str(old_nulls):<5} ({old_pct:<6}) {str(new_nulls):<5} ({new_pct:<6}) {status}")

print()
print()
print("ALL FEATURES - NULL COUNT:")
print("-" * 80)
print(f"{'Feature':<30} {'Old NaNs':<12} {'New NaNs':<12} {'Status':<20}")
print("-" * 80)

critical_issues = []
for feature in ALL_FEATURES:
    old_nulls = old_df[feature].isnull().sum() if feature in old_df.columns else "MISSING"
    new_nulls = new_df[feature].isnull().sum() if feature in new_df.columns else "MISSING"
    
    old_pct = f"{old_nulls/len(old_df)*100:.1f}%" if isinstance(old_nulls, int) else "N/A"
    new_pct = f"{new_nulls/len(new_df)*100:.1f}%" if isinstance(new_nulls, int) else "N/A"
    
    status = "OK"
    if new_nulls == "MISSING":
        status = "⚠️ MISSING COLUMN"
        critical_issues.append((feature, "Missing from new data"))
    elif isinstance(new_nulls, int) and new_nulls > len(new_df) * 0.5:
        status = "🔴 >50% NULL"
        critical_issues.append((feature, f"{new_nulls/len(new_df)*100:.1f}% NULL"))
    elif isinstance(new_nulls, int) and new_nulls > len(new_df) * 0.1:
        status = "🟡 >10% NULL"
    
    if status != "OK":
        print(f"{feature:<30} {str(old_nulls):<5} ({old_pct:<6}) {str(new_nulls):<5} ({new_pct:<6}) {status}")

print()
print()
print("=" * 80)
print("STEP 2: VALUE DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

print("PROBLEM FEATURES - STATISTICS:")
print("-" * 80)
print(f"{'Feature':<30} {'Dataset':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Unique':<8}")
print("-" * 80)

for feature in PROBLEM_FEATURES:
    if feature in old_df.columns:
        old_data = old_df[feature].dropna()
        if len(old_data) > 0:
            print(f"{feature:<30} {'OLD':<8} {old_data.mean():<10.4f} {old_data.std():<10.4f} {old_data.min():<10.4f} {old_data.max():<10.4f} {old_data.nunique():<8}")
    
    if feature in new_df.columns:
        new_data = new_df[feature].dropna()
        if len(new_data) > 0:
            print(f"{feature:<30} {'NEW':<8} {new_data.mean():<10.4f} {new_data.std():<10.4f} {new_data.min():<10.4f} {new_data.max():<10.4f} {new_data.nunique():<8}")
        else:
            print(f"{feature:<30} {'NEW':<8} {'NO DATA':>10} {'NO DATA':>10} {'NO DATA':>10} {'NO DATA':>10} {0:<8}")
    
    print("-" * 80)

print()
print()
print("=" * 80)
print("STEP 3: ZERO VARIANCE CHECK")
print("=" * 80)
print()

print("Features with no variance (constant values) in NEW data:")
print("-" * 80)
zero_variance = []
for feature in ALL_FEATURES:
    if feature in new_df.columns:
        new_data = new_df[feature].dropna()
        if len(new_data) > 0:
            if new_data.std() < 1e-10:  # Essentially zero variance
                print(f"  ⚠️ {feature:<30} Std: {new_data.std():.10f}  (Constant value: {new_data.mean():.4f})")
                zero_variance.append(feature)

if len(zero_variance) == 0:
    print("  ✓ No zero-variance features found")

print()
print()
print("=" * 80)
print("STEP 4: DISTRIBUTION COMPARISON (PROBLEM FEATURES)")
print("=" * 80)
print()

# Create distribution plots for problem features
fig, axes = plt.subplots(len(PROBLEM_FEATURES), 1, figsize=(12, len(PROBLEM_FEATURES) * 3))
if len(PROBLEM_FEATURES) == 1:
    axes = [axes]

for idx, feature in enumerate(PROBLEM_FEATURES):
    ax = axes[idx]
    
    if feature in old_df.columns and feature in new_df.columns:
        old_data = old_df[feature].dropna()
        new_data = new_df[feature].dropna()
        
        if len(old_data) > 0 and len(new_data) > 0:
            # Plot histograms
            ax.hist(old_data, bins=50, alpha=0.5, label=f'OLD (n={len(old_data)})', density=True, color='blue')
            ax.hist(new_data, bins=50, alpha=0.5, label=f'NEW (n={len(new_data)})', density=True, color='red')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'{feature} - Distribution Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'NO DATA AVAILABLE', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feature} - NO DATA')
    else:
        ax.text(0.5, 0.5, 'FEATURE NOT FOUND', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{feature} - MISSING')

plt.tight_layout()
plt.savefig('problem_features_distributions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: problem_features_distributions.png")

print()
print()
print("=" * 80)
print("STEP 5: ELO FEATURE COMPARISON")
print("=" * 80)
print()

print("Comparing ELO features between OLD and NEW datasets:")
print("-" * 80)
elo_features = ['home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff']
print(f"{'Feature':<25} {'Dataset':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-" * 80)
for feature in elo_features:
    if feature in old_df.columns:
        old_data = old_df[feature].dropna()
        print(f"{feature:<25} {'OLD':<8} {old_data.mean():<12.2f} {old_data.std():<12.2f} {old_data.min():<12.2f} {old_data.max():<12.2f}")
    
    if feature in new_df.columns:
        new_data = new_df[feature].dropna()
        print(f"{feature:<25} {'NEW':<8} {new_data.mean():<12.2f} {new_data.std():<12.2f} {new_data.min():<12.2f} {new_data.max():<12.2f}")
    
    print()

print()
print("=" * 80)
print("SUMMARY & DIAGNOSIS")
print("=" * 80)
print()

if len(critical_issues) > 0:
    print("🔴 CRITICAL ISSUES FOUND:")
    print("-" * 80)
    for feature, issue in critical_issues:
        print(f"  • {feature}: {issue}")
    print()
    print("LIKELY CAUSE:")
    print("  The create_proper_training_data.py script only replaced ELO features")
    print("  but didn't copy the other 21 features from the old training data!")
    print()
    print("SOLUTION:")
    print("  Re-run create_proper_training_data.py to copy ALL features from old data")
    print("  and only replace the 4 ELO features with Gold Standard values.")
else:
    print("✓ No critical issues found in NULL counts")

if len(zero_variance) > 0:
    print()
    print("🟡 ZERO VARIANCE FEATURES:")
    print("-" * 80)
    for feature in zero_variance:
        print(f"  • {feature}")
    print()
    print("LIKELY CAUSE:")
    print("  These features are constant in the new data (no variation)")
    print("  XGBoost automatically ignores features with zero variance")

print()
print("=" * 80)
