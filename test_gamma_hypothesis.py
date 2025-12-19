"""
DEEPER DIAGNOSTIC: Why does XGBoost ignore 5 features with the new hyperparameters?

The data is identical between old and new training sets.
The issue must be in how the new hyperparameters interact with feature selection.

Hypothesis: With Trial 340's hyperparameters (max_depth=5, colsample_bytree=0.946, etc.),
XGBoost is finding that ELO features (def_elo_diff, off_elo_diff) provide such strong
signal that it never needs to split on the other features.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

# Trial 340 hyperparameters (best from Gold ELO optimization)
TRIAL_340_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.178,
    'n_estimators': 281,
    'min_child_weight': 10,
    'subsample': 0.84,
    'colsample_bytree': 0.946,
    'colsample_bylevel': 0.978,
    'gamma': 9.537,
    'reg_alpha': 1.887,
    'reg_lambda': 1.327,
    'random_state': 42,
    'tree_method': 'hist',
    'verbosity': 0
}

# All features
FEATURES = [
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'net_fatigue_score', 'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff', 
    'ewma_orb_diff', 'ewma_vol_3p_diff', 'injury_impact_diff', 'injury_shock_diff',
    'star_mismatch', 'ewma_chaos_home', 'ewma_foul_synergy_home', 
    'total_foul_environment', 'league_offensive_context', 'season_progress',
    'pace_efficiency_interaction', 'projected_possession_margin',
    'three_point_matchup', 'net_free_throw_advantage', 'star_power_leverage',
    'offense_vs_defense_matchup', 'injury_matchup_advantage'
]

print("=" * 80)
print("FEATURE SELECTION DIAGNOSTIC")
print("=" * 80)
print()

# Load data
df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
X = df[FEATURES].copy()
y = df['target_moneyline_win']

print(f"Training data: {len(df)} games, {len(FEATURES)} features")
print()

# Check for any features that are perfectly correlated
print("FEATURE CORRELATION WITH TARGET:")
print("-" * 80)
correlations = []
for feature in FEATURES:
    corr = X[feature].corr(y)
    correlations.append((feature, abs(corr)))

correlations.sort(key=lambda x: x[1], reverse=True)
print(f"{'Feature':<30} {'|Correlation|':<15}")
print("-" * 80)
for feature, corr in correlations[:10]:
    print(f"{feature:<30} {corr:<15.6f}")

print()
print()

# Check feature correlations with each other
print("FEATURE INTER-CORRELATION (checking for redundancy):")
print("-" * 80)
corr_matrix = X.corr()

# Find features highly correlated with ELO features
elo_features = ['home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff']
print("Features highly correlated with ELO features (|r| > 0.5):")
print()
for elo_feat in elo_features:
    high_corr = []
    for feature in FEATURES:
        if feature != elo_feat:
            corr = corr_matrix.loc[elo_feat, feature]
            if abs(corr) > 0.5:
                high_corr.append((feature, corr))
    
    if len(high_corr) > 0:
        print(f"{elo_feat}:")
        for feat, corr in sorted(high_corr, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feat:<40} {corr:>7.4f}")
        print()

print()
print("=" * 80)
print("XGBOOST FEATURE SELECTION WITH DIFFERENT PARAMETERS")
print("=" * 80)
print()

# Train model with Trial 340 params
tscv = TimeSeriesSplit(n_splits=5)
fold_idx = 0
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    fold_idx += 1
    if fold_idx == 5:  # Use last fold
        break

print(f"Training with Trial 340 hyperparameters...")
print(f"  max_depth={TRIAL_340_PARAMS['max_depth']}")
print(f"  n_estimators={TRIAL_340_PARAMS['n_estimators']}")
print(f"  learning_rate={TRIAL_340_PARAMS['learning_rate']}")
print(f"  colsample_bytree={TRIAL_340_PARAMS['colsample_bytree']}")
print(f"  gamma={TRIAL_340_PARAMS['gamma']}")
print()

model = xgb.XGBClassifier(**TRIAL_340_PARAMS)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Get feature importance
importance = model.feature_importances_
feature_importance = list(zip(FEATURES, importance))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("FEATURE IMPORTANCE RANKING:")
print("-" * 80)
print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'% of Total':<12}")
print("-" * 80)

total_importance = sum(importance)
cumulative = 0
for rank, (feature, imp) in enumerate(feature_importance, 1):
    pct = imp / total_importance * 100
    cumulative += pct
    marker = ""
    if imp == 0:
        marker = " ⚠️ ZERO"
    elif imp < 0.001:
        marker = " ⚠️ <0.001"
    print(f"{rank:<6} {feature:<35} {imp:<12.6f} {pct:<11.2f}% {marker}")
    if cumulative >= 90:
        print(f"       ... (90% of importance reached)")
        break

print()
print()
print("ZERO-IMPORTANCE FEATURES:")
print("-" * 80)
zero_features = [f for f, imp in feature_importance if imp == 0]
if len(zero_features) > 0:
    for feature in zero_features:
        print(f"  • {feature}")
else:
    print("  None")

print()
print()

# Get predictions and check accuracy
y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)
accuracy = (y_pred == y_val).mean()
logloss = log_loss(y_val, y_pred_proba)

print(f"Model Performance on validation fold:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Log-Loss: {logloss:.4f}")

print()
print()
print("=" * 80)
print("HYPOTHESIS TESTING")
print("=" * 80)
print()

print("Testing: Does gamma=9.537 prevent splits on low-importance features?")
print("-" * 80)
print()
print("gamma (min_split_loss) = 9.537 means:")
print("  - Each split must reduce loss by at least 9.537 to be added")
print("  - With such high gamma, only the strongest features (ELO) are worth splitting on")
print("  - Weaker features (star_mismatch, etc.) don't provide enough gain")
print()

# Try training with gamma=0 to see if features reappear
TRIAL_340_NO_GAMMA = TRIAL_340_PARAMS.copy()
TRIAL_340_NO_GAMMA['gamma'] = 0

print("Re-training with gamma=0 (remove split penalty)...")
model_no_gamma = xgb.XGBClassifier(**TRIAL_340_NO_GAMMA)
model_no_gamma.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

importance_no_gamma = model_no_gamma.feature_importances_
feature_importance_no_gamma = list(zip(FEATURES, importance_no_gamma))
feature_importance_no_gamma.sort(key=lambda x: x[1], reverse=True)

print()
print("FEATURE IMPORTANCE WITH GAMMA=0:")
print("-" * 80)
print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Change from gamma=9.537':<20}")
print("-" * 80)

for rank, (feature, imp_no_gamma) in enumerate(feature_importance_no_gamma, 1):
    # Find original importance
    imp_original = next((imp for f, imp in feature_importance if f == feature), 0)
    change = imp_no_gamma - imp_original
    marker = ""
    if imp_original == 0 and imp_no_gamma > 0:
        marker = " ✓ RECOVERED"
    
    print(f"{rank:<6} {feature:<35} {imp_no_gamma:<12.6f} {change:+.6f}{marker}")
    if rank >= 15:
        break

print()
print()
print("ZERO-IMPORTANCE FEATURES WITH GAMMA=0:")
print("-" * 80)
zero_features_no_gamma = [f for f, imp in feature_importance_no_gamma if imp == 0]
if len(zero_features_no_gamma) > 0:
    for feature in zero_features_no_gamma:
        print(f"  • {feature}")
else:
    print("  ✓ All features have non-zero importance!")

print()
print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

recovered = len(zero_features) - len(zero_features_no_gamma)
if recovered > 0:
    print(f"✓ Setting gamma=0 recovered {recovered} features!")
    print()
    print("ROOT CAUSE:")
    print("  gamma=9.537 is TOO HIGH - it prevents splits on weaker features")
    print("  With clean Gold Standard ELO, the ELO features are so strong that")
    print("  splits on other features don't meet the gamma threshold")
    print()
    print("SOLUTION:")
    print("  1. Re-run optimization with gamma range 0-5 (not 0-10)")
    print("  2. Or accept that clean ELO makes other features less important")
    print("  3. The model may actually be CORRECT - it doesn't need those features")
else:
    print("⚠️ Features still zero even with gamma=0")
    print("  This suggests features are genuinely not useful with Gold Standard ELO")
    print("  OR there's a data issue we haven't found yet")

print()
print("=" * 80)
