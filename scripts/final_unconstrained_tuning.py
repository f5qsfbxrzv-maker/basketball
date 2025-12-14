"""
UNCONSTRAINED HYPERTUNING - 43 Features (Clean + Temporal)
No constraints - let model learn complex nonlinear relationships.

Strategy:
1. Clean data (bugs fixed) + temporal features (era signals explicit)
2. NO constraints (basketball relationships are complex)
3. Aggressive search space (lower reg, deeper trees)
4. 250 trials with 5-fold CV

Baseline: 0.5570 AUC unconstrained
Target: 0.570+ AUC with tuning
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import json
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

print("="*70)
print("UNCONSTRAINED HYPERTUNING - 43 FEATURES")
print("="*70)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season',
             'target_spread', 'target_spread_cover', 'target_moneyline_win',
             'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

features = [c for c in df.columns if c not in drop_cols]
X = df[features]
y = df['target_spread_cover']

print(f"   Games: {len(df):,}")
print(f"   Features: {len(features)}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

# Define objective
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'auc',
        
        # Aggressive hyperparameter ranges
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.12, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 800, 3000, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 35),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 5.0),
        'random_state': 42,
        'verbosity': 0
    }
    
    # 5-fold Time Series CV
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)

print("\n2. Running Optuna optimization...")
print("   Trials: 250")
print("   CV Folds: 5 (TimeSeriesSplit)")
print("   Strategy: Unconstrained - learn all patterns")
print("   Baseline: 0.5570 AUC")
print("   Target: 0.570+ AUC")
print("")
print("   ETA: ~80-100 minutes")
print("   Press Ctrl+C to stop early")
print("")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=250, show_progress_bar=True)

# Results
print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)
print(f"Best trial: #{study.best_trial.number}")
print(f"Best AUC: {study.best_value:.5f}")

print("\nBest parameters:")
for key, value in study.best_trial.params.items():
    if isinstance(value, float):
        print(f"  {key:20s}: {value:.4f}")
    else:
        print(f"  {key:20s}: {value}")

# Performance comparison
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"Dirty 36 features (bugs):           0.55080")
print(f"Clean 36 features:                  0.54070")
print(f"Clean 43 features (baseline):       0.55698")
print(f"Clean 43 features (TUNED):          {study.best_value:.5f}")
print("")
improvement = (study.best_value - 0.55698) * 100
print(f"Improvement from baseline:  {improvement:+.2f}%")
print(f"Target (0.570):             {'✓ ACHIEVED' if study.best_value >= 0.570 else f'Need +{(0.570 - study.best_value)*100:.2f}%'}")

# Save
output_path = "models/final_unconstrained_best_params.json"
with open(output_path, 'w') as f:
    json.dump({
        'best_trial': study.best_trial.number,
        'best_auc': study.best_value,
        'best_params': study.best_trial.params,
        'tuning_date': datetime.now().isoformat(),
        'dataset': 'training_data_with_temporal_features.csv',
        'n_features': len(features),
        'n_trials': 250,
        'cv_folds': 5,
        'strategy': 'Unconstrained - complex nonlinear relationships',
        'baselines': {
            'dirty_36': 0.55080,
            'clean_36': 0.54070,
            'clean_43_baseline': 0.55698
        }
    }, f, indent=2)

print(f"\n✓ Saved: {output_path}")

# Train final model
print("\n3. Training final model...")
model_final = xgb.XGBClassifier(
    **study.best_trial.params,
    objective='binary:logistic',
    tree_method='hist',
    eval_metric='auc',
    random_state=42,
    verbosity=0
)
model_final.fit(X, y)

preds_full = model_final.predict_proba(X)[:, 1]
auc_full = roc_auc_score(y, preds_full)
acc_full = accuracy_score(y, (preds_full > 0.5).astype(int))

print(f"   Full dataset AUC: {auc_full:.5f}")
print(f"   Full dataset Accuracy: {acc_full:.4f}")

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 15 features:")
for idx, row in importances.head(15).iterrows():
    print(f"     {row['feature']:30s}: {row['importance']*100:5.2f}%")

temporal_features = ['season_year', 'season_year_normalized', 'games_into_season',
                     'season_progress', 'is_season_opener', 'endgame_phase', 'season_month']
temporal_imp = importances[importances['feature'].isin(temporal_features)]
print(f"\n   Temporal features: {temporal_imp['importance'].sum()*100:.2f}%")

# Save model
model_path = "models/final_unconstrained_model.json"
model_final.save_model(model_path)
print(f"\n✓ Saved: {model_path}")

importances.to_csv("output/final_unconstrained_feature_importance.csv", index=False)
print(f"✓ Saved: output/final_unconstrained_feature_importance.csv")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Calibrate probabilities (isotonic/Platt)")
print("2. Walk-forward backtest on 2024-25")
print("3. Integrate with Kelly optimizer")
print("4. Deploy to dashboard")
print("="*70)
