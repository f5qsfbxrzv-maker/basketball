"""
Quick check of current optimization's feature usage
Loads the study from memory if available, or checks saved model
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# Load training data
df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')

# Expected features
FEATURES = [
    'home_composite_elo', 'away_composite_elo',
    'home_off_elo', 'away_off_elo', 'home_def_elo', 'away_def_elo',
    'off_elo_diff', 'def_elo_diff',
    'home_recent_performance_ema', 'away_recent_performance_ema',
    'ewma_fga_diff', 'ewma_fg3a_diff', 'ewma_fta_diff', 'ewma_orb_diff', 'ewma_ast_diff',
    'ewma_stl_diff', 'ewma_blk_diff', 'ewma_tov_diff', 'ewma_pf_diff',
    'star_power_diff', 'star_power_leverage',
    'star_mismatch',
    'ewma_foul_synergy_home', 'season_progress',
    'hca_strength'
]

BLOCKED_FEATURES = [
    'star_mismatch',
    'ewma_tov_diff',
    'ewma_foul_synergy_home',
    'season_progress',
    'star_power_leverage'
]

# Prepare data
X = df[FEATURES].values
y = df['home_win'].values

# Trial 106 parameters (best so far at 0.6311)
best_params = {
    'max_depth': 3,
    'learning_rate': 0.012038551721886878,
    'n_estimators': 639,
    'min_child_weight': 34,
    'gamma': 3.636210584316306,
    'subsample': 0.7124020952317704,
    'colsample_bytree': 0.969671365806581,
    'colsample_bylevel': 0.863676057184114,
    'reg_alpha': 4.662850115359488,
    'reg_lambda': 4.740093653229383,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42
}

print("=" * 80)
print("FEATURE USAGE CHECK - TRIAL 106 (BEST: 0.6311)")
print("=" * 80)
print(f"\nParameters:")
print(f"  gamma: {best_params['gamma']:.3f}")
print(f"  max_depth: {best_params['max_depth']}")
print(f"  min_child_weight: {best_params['min_child_weight']}")
print(f"  learning_rate: {best_params['learning_rate']:.4f}")
print(f"  n_estimators: {best_params['n_estimators']}")

# Train model with these parameters
print(f"\nTraining model with Trial 106 parameters...")
model = xgb.XGBClassifier(**best_params)
model.fit(X, y, verbose=False)

# Get feature importance
importance = model.feature_importances_
importance_dict = dict(zip(FEATURES, importance))

# Sort by importance
sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Show all features
print("\nALL 25 FEATURES (sorted by importance):")
print(f"{'Rank':<6}{'Feature':<30}{'Importance':<12}{'Status'}")
print("-" * 80)

elo_features = ['home_composite_elo', 'away_composite_elo', 'home_off_elo', 'away_off_elo', 
                'home_def_elo', 'away_def_elo', 'off_elo_diff', 'def_elo_diff']
total_elo_importance = 0
zero_features = []

for rank, (feature, imp) in enumerate(sorted_features, 1):
    pct = imp * 100
    
    # Determine status
    status = ""
    if imp == 0:
        status = "⚠️ UNUSED"
        zero_features.append(feature)
    elif feature in elo_features:
        status = "🎯 ELO"
        total_elo_importance += imp
    elif feature in BLOCKED_FEATURES:
        status = "✓ RECOVERED" if imp > 0 else "❌ BLOCKED"
    
    print(f"{rank:<6}{feature:<30}{pct:>6.2f}%     {status}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal ELO importance: {total_elo_importance * 100:.2f}%")
print(f"Zero-importance features: {len(zero_features)}")

if len(zero_features) > 0:
    print(f"\nUNUSED FEATURES ({len(zero_features)}):")
    for feat in zero_features:
        blocked_status = " (was blocked in Trial 340)" if feat in BLOCKED_FEATURES else ""
        print(f"  - {feat}{blocked_status}")
else:
    print(f"\n✅ ALL 25 FEATURES ARE BEING USED!")

# Check blocked features specifically
print("\n" + "=" * 80)
print("BLOCKED FEATURES STATUS (from Trial 340)")
print("=" * 80)
recovered = []
still_blocked = []
for feat in BLOCKED_FEATURES:
    imp = importance_dict[feat]
    if imp > 0:
        recovered.append((feat, imp * 100))
    else:
        still_blocked.append(feat)

if len(recovered) > 0:
    print(f"\n✅ RECOVERED ({len(recovered)}/{len(BLOCKED_FEATURES)}):")
    for feat, pct in recovered:
        print(f"  - {feat}: {pct:.2f}%")

if len(still_blocked) > 0:
    print(f"\n❌ STILL BLOCKED ({len(still_blocked)}/{len(BLOCKED_FEATURES)}):")
    for feat in still_blocked:
        print(f"  - {feat}")

# Final assessment
print("\n" + "=" * 80)
print("ASSESSMENT")
print("=" * 80)
print(f"\nTrial 106 log-loss: 0.6311")
print(f"Target (Trial 1306): 0.6222")
print(f"Difference: {0.6311 - 0.6222:.4f} (+{(0.6311/0.6222 - 1) * 100:.2f}%)")
print(f"\nStatus: {'✅ BEATING TARGET!' if 0.6311 < 0.6222 else '⚠️ Not beating target yet'}")
print(f"Gamma: {best_params['gamma']:.3f} (constraint: 0.1-4.0)")

if len(recovered) == len(BLOCKED_FEATURES):
    print("\n🎉 All 5 blocked features have RECOVERED!")
elif len(recovered) > 0:
    print(f"\n✓ {len(recovered)}/{len(BLOCKED_FEATURES)} blocked features recovered")
else:
    print(f"\n⚠️ No blocked features recovered yet (gamma may still be too high)")
