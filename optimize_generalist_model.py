"""
SYNDICATE MODEL OPTIMIZATION - 22 Features with Matchup Advantages

Strategy: Use pre-computed matchup differentials and friction features
so XGBoost doesn't waste splits learning subtraction.

Target: Beat Trial 1306's 0.6222 log-loss
Baseline: 0.6308 (Trial 144 with Gold ELO but hobbyist features)

Key Syndicate Upgrades:
- ELO matchup advantages (off vs def cross-matchups)
- Matchup friction (team strength vs opponent weakness)
- Volume-adjusted efficiency (eFG% × Possessions)
- Consolidated injury leverage
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import optuna
from optuna.pruners import MedianPruner
import json
from datetime import datetime

# 22 SYNDICATE features
FEATURES = [
    # Tier 1: ELO Matchup Advantages (3)
    'off_matchup_advantage', 'def_matchup_advantage', 'net_composite_advantage',
    
    # Tier 2: Matchup Friction (5)
    'effective_shooting_gap', 'turnover_pressure', 'rebound_friction',
    'total_rebound_control', 'whistle_leverage',
    
    # Tier 3: Volume & Injury (2)
    'volume_efficiency_diff', 'injury_leverage',
    
    # Tier 4: Context & Supporting (12)
    'net_fatigue_score', 'ewma_chaos_home', 'ewma_foul_synergy_home',
    'total_foul_environment', 'league_offensive_context', 'season_progress',
    'three_point_matchup', 'star_power_leverage', 'offense_vs_defense_matchup',
    'ewma_pace_diff', 'ewma_vol_3p_diff', 'projected_possession_margin'
]

# Features to watch (new syndicate features)
NEW_SYNDICATE_FEATURES = [
    'off_matchup_advantage', 'def_matchup_advantage', 'net_composite_advantage',
    'effective_shooting_gap', 'turnover_pressure', 'rebound_friction',
    'total_rebound_control', 'whistle_leverage', 'volume_efficiency_diff'
]

print("=" * 80)
print("SYNDICATE MODEL OPTIMIZATION")
print("=" * 80)
print()
print("Objective: Beat Trial 1306's 0.6222 log-loss")
print("Strategy: Matchup advantages + friction features (22 features)")
print()
print("Key Constraints:")
print("  gamma: 0.1-4.0 (lower ceiling to unlock weak features)")
print("  max_depth: 3-7 (controlled depth)")
print("  min_child_weight: 10-40 (noise control)")
print("  learning_rate: 0.01-0.1 log scale (slower for weak features)")
print()

# Load data
print("Loading Syndicate training data...")
df = pd.read_csv('data/training_data_SYNDICATE_28_features.csv')
X = df[FEATURES].copy()
y = df['target_moneyline_win']

print(f"Total games: {len(df)}")
print(f"Features: {len(FEATURES)}")
print(f"Home win rate: {y.mean():.3f}")
print()

# Feature Audit Callback
class FeatureAuditCallback:
    """Monitor syndicate feature usage during optimization"""
    
    def __init__(self, features, syndicate_features, check_every=50):
        self.features = features
        self.syndicate_features = syndicate_features
        self.check_every = check_every
        self.last_check = 0
        
    def __call__(self, study, trial):
        # Only check every N trials
        if trial.number - self.last_check < self.check_every:
            return
            
        self.last_check = trial.number
        
        # Get best trial so far
        best_trial = study.best_trial
        
        # Extract feature importance from user attributes if available
        if 'feature_importance' in best_trial.user_attrs:
            importance_dict = best_trial.user_attrs['feature_importance']
            
            # Check syndicate feature usage
            syndicate_usage = {f: importance_dict.get(f, 0) for f in self.syndicate_features if f in self.features}
            zero_syndicate = [f for f, imp in syndicate_usage.items() if imp == 0]
            
            print()
            print(f"⚠️  SYNDICATE AUDIT (Trial {trial.number}):")
            print(f"   Best trial: #{best_trial.number} (log-loss: {best_trial.value:.4f})")
            print(f"   Syndicate features active: {len(syndicate_usage) - len(zero_syndicate)}/{len(syndicate_usage)}")
            
            if len(zero_syndicate) > 0:
                print(f"   ✗ Unused syndicate features: {', '.join(zero_syndicate)}")
            else:
                print(f"   ✓ All syndicate features active!")
                
            # Show top syndicate features
            active_syndicate = [(f, imp) for f, imp in syndicate_usage.items() if imp > 0]
            if active_syndicate:
                active_syndicate.sort(key=lambda x: x[1], reverse=True)
                print(f"   Top syndicate features:")
                for f, imp in active_syndicate[:5]:
                    print(f"     {f}: {imp:.1f}%")
            print()

# Objective function
def objective(trial):
    # REVISED HYPERPARAMETER RANGES FOR GENERALIST MODEL
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42,
        
        # Core parameters - FOCUSED RANGES
        'max_depth': trial.suggest_int('max_depth', 3, 7),  # Controlled depth
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  # Slower learning
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),  # More trees with slower learning
        
        # Regularization - USE min_child_weight INSTEAD OF gamma FOR NOISE CONTROL
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 40),  # Strong noise control
        'gamma': trial.suggest_float('gamma', 0.1, 4.0),  # LOWER CEILING - key change!
        
        # Sampling
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
        
        # L1/L2 regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0)
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Evaluate
        y_pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, y_pred)
        fold_scores.append(ll)
        
        # Report intermediate value for pruning
        trial.report(ll, fold)
        
        # Prune if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Calculate mean log-loss
    mean_ll = np.mean(fold_scores)
    
    # Store feature importance from final model for audit
    importance_dict = dict(zip(FEATURES, model.feature_importances_))
    trial.set_user_attr('feature_importance', importance_dict)
    
    # Store individual fold scores
    trial.set_user_attr('fold_scores', fold_scores)
    trial.set_user_attr('fold_std', np.std(fold_scores))
    
    return mean_ll

# Create study with conservative pruning
print("Creating Optuna study...")
print("  Pruner: MedianPruner(n_startup_trials=100, n_warmup_steps=3)")
print("  Trials: 500")
print("  Estimated time: 3-4 hours")
print()

study = optuna.create_study(
    direction='minimize',
    study_name='generalist_gold_elo',
    pruner=MedianPruner(n_startup_trials=100, n_warmup_steps=3)
)

# Add feature audit callback
feature_audit = FeatureAuditCallback(FEATURES, NEW_SYNDICATE_FEATURES, check_every=50)

print("Starting optimization...")
print("=" * 80)
print()

try:
    study.optimize(
        objective, 
        n_trials=2000, 
        callbacks=[feature_audit],
        show_progress_bar=True
    )
except KeyboardInterrupt:
    print("\n\nOptimization interrupted by user")

print()
print("=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
print()

# Results
best_trial = study.best_trial
print(f"Best Trial: #{best_trial.number}")
print(f"Best Log-Loss: {best_trial.value:.6f}")
print()

print("Best Hyperparameters:")
print("-" * 80)
for key, value in best_trial.params.items():
    print(f"  {key:<25} {value}")

print()
print("Performance Details:")
print("-" * 80)
fold_scores = best_trial.user_attrs.get('fold_scores', [])
fold_std = best_trial.user_attrs.get('fold_std', 0)
print(f"  5-Fold CV Mean:  {best_trial.value:.6f}")
print(f"  5-Fold CV Std:   {fold_std:.6f}")
print(f"  Individual folds: {[f'{x:.4f}' for x in fold_scores]}")

print()
print("Comparison to Baselines:")
print("-" * 80)
print(f"  Trial 1306 (OLD ELO):    0.6222")
print(f"  Trial 340 (high gamma):  0.6297")
print(f"  This trial:              {best_trial.value:.6f}")

improvement_vs_1306 = (0.6222 - best_trial.value) / 0.6222 * 100
improvement_vs_340 = (0.6297 - best_trial.value) / 0.6297 * 100

if best_trial.value < 0.6222:
    print(f"  ✓ BEAT Trial 1306 by {improvement_vs_1306:.2f}%!")
elif best_trial.value < 0.6297:
    print(f"  ✓ Better than Trial 340 by {improvement_vs_340:.2f}%")
else:
    print(f"  ✗ Still worse than baselines")

print()
print("Feature Importance Analysis:")
print("-" * 80)
importance_dict = best_trial.user_attrs.get('feature_importance', {})
feature_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Status':<15}")
print("-" * 80)

elo_importance = 0
blocked_recovered = 0
for rank, (feature, imp) in enumerate(feature_importance, 1):
    status = ""
    if feature in ['home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff']:
        elo_importance += imp
        status = "ELO"
    elif feature in BLOCKED_FEATURES:
        if imp > 0:
            status = "✓ RECOVERED"
            blocked_recovered += 1
        else:
            status = "✗ BLOCKED"
    
    print(f"{rank:<6} {feature:<35} {imp:<12.6f} {status:<15}")

print()
print(f"Total syndicate importance: {syndicate_importance:.4f} ({syndicate_importance*100:.1f}%)")
print(f"Syndicate features active: {sum(1 for f in NEW_SYNDICATE_FEATURES if importance_dict.get(f, 0) > 0)}/{len(NEW_SYNDICATE_FEATURES)}")

# Save results
print()
print("Saving results...")
results = {
    'best_trial_number': best_trial.number,
    'best_log_loss': best_trial.value,
    'best_params': best_trial.params,
    'fold_scores': fold_scores,
    'fold_std': fold_std,
    'feature_importance': importance_dict,
    'timestamp': datetime.now().isoformat(),
    'total_trials': len(study.trials),
    'comparison': {
        'trial_1306': 0.6222,
        'trial_340': 0.6297,
        'this_trial': best_trial.value,
        'beat_1306': best_trial.value < 0.6222,
        'beat_340': best_trial.value < 0.6297
    }
}

with open('models/generalist_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Saved: models/generalist_model_results.json")

# Train final model with best params
print()
print("Training final model with best hyperparameters...")
final_model = xgb.XGBClassifier(**best_trial.params)
final_model.fit(X, y, verbose=False)

# Save model
final_model.save_model('models/generalist_gold_elo_model.json')
print("✓ Saved: models/generalist_gold_elo_model.json")

print()
print("=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)
print()
print(f"Target:   Beat 0.6222 (Trial 1306 on OLD ELO)")
print(f"Result:   {best_trial.value:.6f}")
print(f"Status:   {'✓ SUCCESS' if best_trial.value < 0.6222 else '✗ Need more optimization'}")
print()
print(f"ELO Signal:     {elo_importance*100:.1f}% of importance")
print(f"Features Used:  {sum(1 for _, imp in feature_importance if imp > 0)}/{len(FEATURES)}")
print(f"Blocked:        {len(BLOCKED_FEATURES) - blocked_recovered}")
print()
print("=" * 80)
