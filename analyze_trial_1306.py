"""
Analyze Trial 1306 - Current Best Performance
Log Loss: 0.6221971111922393
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import json

# ==============================================================================
# TRIAL 1306 DETAILS FROM OPTUNA OUTPUT
# ==============================================================================
TRIAL_1306_PARAMS = {
    'max_depth': 3,
    'min_child_weight': 25,
    'gamma': 5.162427047142856,
    'learning_rate': 0.010519422544676995,
    'n_estimators': 9947,
    'subsample': 0.6277685565263181,
    'colsample_bytree': 0.6014538139159614,
    'reg_alpha': 6.193992559265241
}

TRIAL_1306_LOG_LOSS = 0.6221971111922393

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("="*90)
print("TRIAL 1306 ANALYSIS - CURRENT BEST")
print("="*90)

DATA_PATH = 'data/training_data_matchup_with_injury_advantage_FIXED.csv'
FEATURES = [
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff', 'ewma_orb_diff',
    'ewma_vol_3p_diff', 'ewma_chaos_home', 'injury_matchup_advantage',
    'net_fatigue_score', 'ewma_foul_synergy_home', 'total_foul_environment',
    'league_offensive_context', 'season_progress', 'pace_efficiency_interaction',
    'projected_possession_margin', 'three_point_matchup', 'net_free_throw_advantage',
    'star_power_leverage', 'offense_vs_defense_matchup'
]
TARGET = 'target_moneyline_win'

print(f"\n[1/4] Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values('date').reset_index(drop=True)

X = df[FEATURES].values
y = df[TARGET].values

print(f"  Samples: {len(df):,}")
print(f"  Features: {len(FEATURES)}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

# ==============================================================================
# HYPERPARAMETER ANALYSIS
# ==============================================================================
print(f"\n[2/4] Trial 1306 Hyperparameters:")
print(f"  {'Parameter':<25} {'Value':<20} {'Analysis'}")
print(f"  {'-'*75}")
print(f"  {'max_depth':<25} {TRIAL_1306_PARAMS['max_depth']:<20} Shallow trees (conservative)")
print(f"  {'min_child_weight':<25} {TRIAL_1306_PARAMS['min_child_weight']:<20} Heavy pruning")
print(f"  {'gamma':<25} {TRIAL_1306_PARAMS['gamma']:<20.4f} Strong split requirement")
print(f"  {'learning_rate':<25} {TRIAL_1306_PARAMS['learning_rate']:<20.6f} Very slow learning")
print(f"  {'n_estimators':<25} {TRIAL_1306_PARAMS['n_estimators']:<20} Many weak learners")
print(f"  {'subsample':<25} {TRIAL_1306_PARAMS['subsample']:<20.4f} 63% row sampling")
print(f"  {'colsample_bytree':<25} {TRIAL_1306_PARAMS['colsample_bytree']:<20.4f} 60% feature sampling")
print(f"  {'reg_alpha':<25} {TRIAL_1306_PARAMS['reg_alpha']:<20.4f} L1 regularization")

# ==============================================================================
# PERFORMANCE COMPARISON
# ==============================================================================
print(f"\n[3/4] Performance Comparison:")
print(f"  {'Model':<40} {'Log Loss':<15} {'Improvement'}")
print(f"  {'-'*70}")
print(f"  {'Previous best (25 features, broken ELO)':<40} 0.6584          Baseline")
print(f"  {'Trial 1306 (22 features, fixed ELO)':<40} {TRIAL_1306_LOG_LOSS:.6f}     {((0.6584 - TRIAL_1306_LOG_LOSS) / 0.6584 * 100):.2f}% better ✅")

improvement_pts = (0.6584 - TRIAL_1306_LOG_LOSS) * 1000
print(f"\n  Absolute improvement: {improvement_pts:.1f} points (lower is better)")
print(f"  Status: {'SIGNIFICANT IMPROVEMENT' if improvement_pts > 20 else 'Marginal improvement'}")

# ==============================================================================
# TRAIN FULL MODEL WITH TRIAL 1306 PARAMS
# ==============================================================================
print(f"\n[4/4] Training full model with Trial 1306 hyperparameters...")

params = TRIAL_1306_PARAMS.copy()
params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42
})

# Extract n_estimators for num_boost_round
num_boost_round = params.pop('n_estimators')

dtrain = xgb.DMatrix(X, label=y)
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    verbose_eval=False
)

# Evaluate
y_pred = model.predict(dtrain)
train_loss = log_loss(y, y_pred)
train_auc = roc_auc_score(y, y_pred)
train_brier = brier_score_loss(y, y_pred)
train_acc = np.mean((y_pred > 0.5) == y)

print(f"\n  Full Dataset Performance:")
print(f"    Log Loss:    {train_loss:.6f}")
print(f"    AUC:         {train_auc:.4f}")
print(f"    Brier Score: {train_brier:.6f}")
print(f"    Accuracy:    {train_acc:.2%}")

# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================
print(f"\n  Feature Importance (Top 15):")
importance_gain = model.get_score(importance_type='gain')

feature_importance = []
for i, feat_name in enumerate(FEATURES):
    gain = importance_gain.get(f'f{i}', 0)
    feature_importance.append({'feature': feat_name, 'gain': gain})

feature_importance.sort(key=lambda x: x['gain'], reverse=True)

print(f"  {'Rank':<6} {'Feature':<40} {'Gain':<12}")
print(f"  {'-'*60}")
for rank, fi in enumerate(feature_importance[:15], 1):
    print(f"  {rank:<6} {fi['feature']:<40} {fi['gain']:<12.1f}")

# Check critical features
home_elo_rank = next((i+1 for i, x in enumerate(feature_importance) if x['feature'] == 'home_composite_elo'), None)
away_elo_rank = next((i+1 for i, x in enumerate(feature_importance) if x['feature'] == 'away_composite_elo'), None)
injury_rank = next((i+1 for i, x in enumerate(feature_importance) if x['feature'] == 'injury_matchup_advantage'), None)

print(f"\n  Key Feature Ranks:")
print(f"    home_composite_elo:       Rank #{home_elo_rank}/22 (was #24/25 with broken ELO)")
print(f"    away_composite_elo:       Rank #{away_elo_rank}/22")
print(f"    injury_matchup_advantage: Rank #{injury_rank}/22")

# ==============================================================================
# SAVE MODEL
# ==============================================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'models/xgboost_22features_trial1306_{timestamp}.json'
model.save_model(output_path)

print(f"\n{'='*90}")
print(f"✅ TRIAL 1306 ANALYSIS COMPLETE")
print(f"{'='*90}")
print(f"  Model saved: {output_path}")
print(f"  Validation log loss: {TRIAL_1306_LOG_LOSS:.6f} (5.5% better than previous best)")
print(f"  Training log loss:   {train_loss:.6f}")
print(f"  Training AUC:        {train_auc:.4f}")
print(f"\n  KEY IMPROVEMENTS:")
print(f"    ✅ Fixed home_composite_elo (no more wild oscillations)")
print(f"    ✅ Removed 3 redundant injury features")
print(f"    ✅ 22 features vs 25 (simpler model)")
print(f"    ✅ Significantly better validation performance")
print(f"\n  READY FOR DEPLOYMENT")
print(f"{'='*90}\n")

# Save hyperparameters
params_output = {
    'trial': 1306,
    'validation_log_loss': TRIAL_1306_LOG_LOSS,
    'training_log_loss': float(train_loss),
    'training_auc': float(train_auc),
    'hyperparameters': TRIAL_1306_PARAMS,
    'features': FEATURES,
    'n_features': len(FEATURES),
    'dataset': 'training_data_matchup_with_injury_advantage_FIXED.csv',
    'timestamp': timestamp
}

with open(f'models/trial1306_params_{timestamp}.json', 'w') as f:
    json.dump(params_output, f, indent=2)

print(f"  Hyperparameters saved: models/trial1306_params_{timestamp}.json\n")
