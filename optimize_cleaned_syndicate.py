"""
Optimize XGBoost on CLEANED syndicate dataset (13 features, redundancy removed).

Previous Results:
- Baseline (25 features, old ELO): 0.6222 log-loss
- Syndicate (22 features, high VIF): 0.6309 log-loss (1.4% worse)
- Redundancy identified: 9 features with VIF > 10 or 0% importance

Hypothesis: Removing redundant features will consolidate importance
- Example: off_matchup_advantage (30.7%) + net_composite_advantage (26.4%) = 57.1%
- Expected: This 57.1% should redistribute to remaining features
- Target: Beat 0.6222 baseline

Cleaned Features (13):
1. def_matchup_advantage - Only ELO feature kept (VIF=1.61)
2. effective_shooting_gap, three_point_matchup - Shooting matchups
3. volume_efficiency_diff - Offensive style
4. net_fatigue_score, injury_leverage - Context factors
5. ewma_pace_diff, ewma_vol_3p_diff - Recent form
6. star_power_leverage, offense_vs_defense_matchup - Composite advantages
7. season_progress, league_offensive_context - Seasonal adjustments
8. whistle_leverage - Foul advantage (VIF=9.78, borderline)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'data/training_data_SYNDICATE_CLEANED_14_features.csv'
N_TRIALS = 500
N_FOLDS = 5
RANDOM_STATE = 42
TARGET = 'target_spread_cover'

# Features to use (13 clean features)
FEATURES = [
    'def_matchup_advantage',
    'effective_shooting_gap', 
    'three_point_matchup',
    'volume_efficiency_diff',
    'net_fatigue_score',
    'injury_leverage',
    'ewma_pace_diff',
    'ewma_vol_3p_diff',
    'star_power_leverage',
    'offense_vs_defense_matchup',
    'season_progress',
    'league_offensive_context',
    'whistle_leverage'
]

print("="*80)
print("OPTIMIZING CLEANED SYNDICATE MODEL (13 Features)")
print("="*80)
print(f"Target: Beat 0.6222 baseline log-loss")
print(f"Trials: {N_TRIALS}")
print(f"CV Folds: {N_FOLDS}")
print(f"Random State: {RANDOM_STATE}")
print("="*80)

# Load data
df = pd.read_csv(DATA_FILE)
print(f"\n✓ Loaded {len(df)} games")

# Verify all features exist
missing_features = [f for f in FEATURES if f not in df.columns]
if missing_features:
    print(f"\n❌ ERROR: Missing features: {missing_features}")
    exit(1)

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Check for NaN/inf
if X.isna().sum().sum() > 0:
    print(f"⚠️  WARNING: {X.isna().sum().sum()} NaN values found, filling with 0")
    X = X.fillna(0)
    
if np.isinf(X.values).sum() > 0:
    print(f"⚠️  WARNING: {np.isinf(X.values).sum()} inf values found, clipping")
    X = X.replace([np.inf, -np.inf], 0)

print(f"✓ Features: {len(FEATURES)}")
print(f"✓ Target distribution: {y.value_counts().to_dict()}")
print(f"✓ Class balance: {y.mean():.3f} (1=home covers)")

# Define objective function
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'verbosity': 0,
        
        # Hyperparameters to optimize
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 0, 4.0),  # Constrained range from previous optimization
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
    }
    
    # Cross-validation
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=1)
    
    return -scores.mean()  # Optuna minimizes, we want to minimize log-loss

# Create study
study = optuna.create_study(direction='minimize', study_name='cleaned_syndicate')
print(f"\n🚀 Starting optimization at {datetime.now().strftime('%H:%M:%S')}")
print("-"*80)

# Optimize
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# Results
print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)

best_trial = study.best_trial
print(f"\n🏆 BEST TRIAL: #{best_trial.number}")
print(f"   Log-loss: {best_trial.value:.4f}")
print(f"   Baseline: 0.6222")
print(f"   Delta: {((best_trial.value/0.6222 - 1)*100):+.2f}%")

if best_trial.value < 0.6222:
    print(f"   ✅ BEATS BASELINE by {((1 - best_trial.value/0.6222)*100):.2f}%")
else:
    print(f"   ❌ Still {((best_trial.value/0.6222 - 1)*100):.2f}% worse than baseline")

print(f"\n📊 BEST HYPERPARAMETERS:")
for key, value in best_trial.params.items():
    print(f"   {key:20s}: {value}")

# Train final model with best params
print(f"\n🔧 Training final model...")
final_model = xgb.XGBClassifier(**best_trial.params, random_state=RANDOM_STATE)
final_model.fit(X, y)

# Feature importance
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\n📈 FEATURE IMPORTANCE (Top 10):")
print("-"*80)
for idx, row in feature_importance_df.head(10).iterrows():
    pct = row['importance'] * 100
    bar = '█' * int(pct / 2)
    print(f"  {row['feature']:30s}: {pct:5.1f}% {bar}")

# Check if importance consolidated
top_feature_pct = feature_importance_df.iloc[0]['importance'] * 100
print(f"\n🎯 TOP FEATURE IMPORTANCE: {top_feature_pct:.1f}%")
if top_feature_pct > 35:
    print(f"   ✅ HIGH CONSOLIDATION (expected from removing redundancy)")
elif top_feature_pct > 25:
    print(f"   ⚠️  MODERATE CONSOLIDATION")
else:
    print(f"   ❌ LOW CONSOLIDATION (redundancy may not have been issue)")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_file = f'optimization_results_cleaned_{timestamp}.txt'

with open(results_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLEANED SYNDICATE OPTIMIZATION RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Trials: {len(study.trials)}\n")
    f.write(f"Best Trial: #{best_trial.number}\n")
    f.write(f"Best Log-loss: {best_trial.value:.4f}\n")
    f.write(f"Baseline: 0.6222\n")
    f.write(f"Delta: {((best_trial.value/0.6222 - 1)*100):+.2f}%\n\n")
    
    f.write("BEST HYPERPARAMETERS:\n")
    f.write("-"*80 + "\n")
    for key, value in best_trial.params.items():
        f.write(f"{key:20s}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("FEATURE IMPORTANCE\n")
    f.write("="*80 + "\n")
    for idx, row in feature_importance_df.iterrows():
        f.write(f"{row['feature']:30s}: {row['importance']*100:5.1f}%\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("ALL TRIALS\n")
    f.write("="*80 + "\n")
    for trial in study.trials:
        f.write(f"Trial {trial.number}: {trial.value:.4f}\n")

print(f"\n✓ Saved results to: {results_file}")

# Save model
model_file = f'cleaned_syndicate_model_{timestamp}.json'
final_model.save_model(model_file)
print(f"✓ Saved model to: {model_file}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
if best_trial.value < 0.6222:
    print("✅ SUCCESS! Model beats baseline.")
    print("   → Run walk-forward backtest on 2024-2025 season")
    print("   → Calculate ROI and unit returns")
    print("   → Compare vs baseline model profitability")
else:
    print("❌ Still worse than baseline. Options:")
    print("   1. Inspect if removed features had unique signal")
    print("   2. Try different gamma range (lower to force usage)")
    print("   3. Investigate if Gold Standard ELO is actually worse than noisy K=32 ELO")
    print("   4. Re-calculate friction features from raw game logs (currently placeholders)")
print("="*80)
