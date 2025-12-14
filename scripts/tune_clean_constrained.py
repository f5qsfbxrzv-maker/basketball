"""
Constrained Hyperparameter Tuning on CLEAN 36-Feature Dataset
Uses monotonic constraints to enforce basketball logic during optimization.

Key Changes from Previous Tuning:
1. Monotonic constraints: off_elo_diff↑ = win↑, back_to_back↑ = win↓
2. Search space shift: Lower regularization (data is clean, trust the signal)
3. Clean data: 286-day rest bug fixed, ELO inflation corrected
4. Target: Break 0.56 AUC with interpretable, logical model

Why Constraints Matter:
- Previous model learned "286 days rest = advantage" (wrong!)
- Constraints force logical relationships: fatigue always hurts
- Prevents overfitting to noise in early folds (small sample)
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
print("CONSTRAINED HYPERPARAMETER TUNING - CLEAN 36 FEATURES")
print("="*70)

# ---------------------------------------------------------
# 1. LOAD CLEAN DATA
# ---------------------------------------------------------
print("\n1. Loading clean dataset...")
df = pd.read_csv('data/training_data_with_features_cleaned.csv')

# Sort by date for proper Time Series Split
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"   Games: {len(df):,}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

# Define Target and Features
target = 'target_spread_cover'
drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season',
             'target_spread', 'target_spread_cover', 'target_moneyline_win',
             'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

features = [c for c in df.columns if c not in drop_cols]
X = df[features]
y = df[target]

print(f"   Features: {len(features)}")
print(f"   Target: {y.value_counts().to_dict()}")

# ---------------------------------------------------------
# 2. DEFINE MONOTONIC CONSTRAINTS
# ---------------------------------------------------------
print("\n2. Setting up monotonic constraints...")
print("   Enforcing basketball logic:")

# 1 = Positive relationship (feature ↑ → win prob ↑)
# -1 = Negative relationship (feature ↑ → win prob ↓)
# 0 = No constraint (let model learn)

constraint_map = {
    # ELO: Higher is better
    'off_elo_diff': 1,           # Better offense → win
    'def_elo_diff': 1,           # Better defense → win
    'home_composite_elo': 1,     # Higher ELO → win
    
    # EWMA Stats: Higher shooting/efficiency is better
    'ewma_efg_diff': 1,          # Better shooting → win
    'ewma_pace_diff': 0,         # Pace is tactical, no constraint
    'ewma_tov_diff': -1,         # More turnovers → loss (home - away, so negative is bad)
    'ewma_orb_diff': 1,          # More offensive rebounds → win
    
    # Fatigue: Always hurts
    'away_back_to_back': -1,     # Fatigue → loss
    'home_back_to_back': -1,     # Fatigue → loss
    'away_3in4': -1,             # Fatigue → loss
    'home_3in4': 0,              # Rare, no constraint
    
    # Rest: More is better (but capped at reasonable values now)
    'away_rest_days': 0,         # Complex (too much = rust), no constraint
    'home_rest_days': 0,         # Complex (too much = rust), no constraint
    'rest_advantage': 1,         # More rest advantage → win
    
    # Injury: Negative impact
    'injury_shock_diff': 1,      # Home injury advantage → win (home - away)
    'injury_shock_home': -1,     # Home injury shock → loss
    'injury_shock_away': 0,      # Away injury (helps home), no direct constraint
    'injury_impact_abs': -1,     # Total injury impact → harder game
    'injury_impact_diff': 1,     # Home injury advantage → win
    'away_star_missing': 1,      # Away star missing → home win
    'home_star_missing': -1,     # Home star missing → home loss
    'star_mismatch': 1,          # Mismatch favoring home → win
    
    # Altitude
    'altitude_game': 0,          # Complex (home advantage in Denver), no constraint
    
    # 3-Point Shooting
    'away_ewma_3p_pct': 0,       # Away shooting (complex)
    'home_ewma_3p_pct': 1,       # Home shooting → win
    'ewma_vol_3p_diff': 0,       # Volume difference (tactical)
    
    # Rebounding
    'away_orb': 0,               # Away offensive boards (helps away)
    'home_orb': 1,               # Home offensive boards → win
    'away_drb': 0,               # Complex
    'home_drb': 1,               # Home defensive boards → win
    
    # Chaos/Volatility
    'ewma_chaos_home': 0,        # Unpredictability (tactical)
    'ewma_net_chaos': 0,         # Net chaos (tactical)
    
    # Fouls (complex relationship, removed from lean dataset)
    'away_ewma_fta_rate': 0,
    'home_ewma_fta_rate': 1,     # Getting to line → win
    'ewma_foul_synergy_away': 0,
    'ewma_foul_synergy_home': 1,
    
    # Fatigue Mismatch
    'fatigue_mismatch': 1,       # Fatigue advantage → win
    
    # Interaction Features (from lean dataset)
    'efficiency_x_pace': 1,      # Efficiency × pace → win
    'tired_altitude': -1,        # Tired at altitude → loss
    'form_x_defense': 1,         # Form × defense → win
}

# Create the tuple required by XGBoost (must match feature order)
constraints = tuple(constraint_map.get(col, 0) for col in features)

# Print applied constraints
constrained_features = [(feat, constraint_map.get(feat, 0)) for feat in features if constraint_map.get(feat, 0) != 0]
print(f"   Total constraints: {len([c for c in constraints if c != 0])}/{len(features)}")
print(f"   Positive (↑ = good): {len([c for c in constraints if c == 1])}")
print(f"   Negative (↑ = bad): {len([c for c in constraints if c == -1])}")

# ---------------------------------------------------------
# 3. DEFINE OPTUNA OBJECTIVE
# ---------------------------------------------------------
print("\n3. Defining optimization objective...")

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'monotone_constraints': constraints,  # ← THE GUARDRAILS
        
        # Hyperparameters to tune (shifted ranges for clean data)
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),  # Lower than before (10-100 → 5-50)
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),  # Lower max (5.0 → 3.0)
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),  # Lower max (5.0 → 3.0)
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'random_state': 42,
        'verbosity': 0
    }
    
    # Time Series Cross-Validation (3 splits for speed, robust validation)
    tscv = TimeSeriesSplit(n_splits=3)
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

# ---------------------------------------------------------
# 4. RUN OPTIMIZATION
# ---------------------------------------------------------
print("\n4. Running Optuna optimization...")
print("   Trials: 200 (recommend 200-300 for constrained search)")
print("   Validation: TimeSeriesSplit (3 folds)")
print("   Strategy: Constrained search with lower regularization")
print("")
print("   Press Ctrl+C to stop early (best params will be saved)")
print("")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, show_progress_bar=True)

# ---------------------------------------------------------
# 5. RESULTS
# ---------------------------------------------------------
print("\n" + "="*70)
print("OPTIMIZATION RESULTS")
print("="*70)
print(f"Best trial: #{study.best_trial.number}")
print(f"Best AUC: {study.best_value:.4f}")

print("\nBest parameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key:20s}: {value}")

# Compare to "toxic" old params (if available)
print("\nKey parameter shifts (clean vs dirty data expectations):")
print(f"  reg_alpha: {study.best_trial.params['reg_alpha']:.2f} (expect <2.0, was >5.0 on dirty)")
print(f"  min_child_weight: {study.best_trial.params['min_child_weight']} (expect <20, was ~23 on dirty)")
print(f"  gamma: {study.best_trial.params['gamma']:.2f} (expect <2.0, was >4.0 on dirty)")

# Save best params
output_path = "models/clean_constrained_best_params.json"
with open(output_path, 'w') as f:
    json.dump({
        'best_trial': study.best_trial.number,
        'best_auc': study.best_value,
        'best_params': study.best_trial.params,
        'tuning_date': datetime.now().isoformat(),
        'dataset': 'training_data_with_features_cleaned.csv',
        'n_trials': 200,
        'strategy': 'Constrained search with monotonic constraints',
        'constraints_applied': len([c for c in constraints if c != 0]),
        'data_fixes': ['286-day rest bug', 'ELO inflation normalized', 'Outliers clipped']
    }, f, indent=2)

print(f"\n✓ Saved best parameters: {output_path}")

# ---------------------------------------------------------
# 6. TRAIN FINAL MODEL
# ---------------------------------------------------------
print("\n5. Training final model with best parameters...")
model_final = xgb.XGBClassifier(
    **study.best_trial.params,
    objective='binary:logistic',
    tree_method='hist',
    eval_metric='auc',
    monotone_constraints=constraints,
    random_state=42,
    verbosity=0
)
model_final.fit(X, y)

# Validate on full dataset
preds_full = model_final.predict_proba(X)[:, 1]
auc_full = roc_auc_score(y, preds_full)
acc_full = accuracy_score(y, (preds_full > 0.5).astype(int))

print(f"   Full dataset AUC: {auc_full:.4f}")
print(f"   Full dataset Accuracy: {acc_full:.4f}")

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model_final.feature_importances_,
    'constraint': [constraint_map.get(f, 0) for f in X.columns]
}).sort_values('importance', ascending=False)

print("\n   Top 15 features:")
for idx, row in importances.head(15).iterrows():
    constraint_str = ''
    if row['constraint'] == 1:
        constraint_str = '↑'
    elif row['constraint'] == -1:
        constraint_str = '↓'
    print(f"     {row['feature']:30s}: {row['importance']*100:5.2f}% {constraint_str}")

# Save model
model_path = "models/clean_constrained_model.json"
model_final.save_model(model_path)
print(f"\n✓ Saved model: {model_path}")

# Save feature importance
importances.to_csv("output/clean_constrained_feature_importance.csv", index=False)
print(f"✓ Saved feature importance: output/clean_constrained_feature_importance.csv")

print("\n" + "="*70)
print("TUNING COMPLETE")
print("="*70)
print(f"Best Validation AUC: {study.best_value:.4f}")
print(f"Full Dataset AUC: {auc_full:.4f}")
print(f"Target (0.56): {'✓ ACHIEVED' if study.best_value >= 0.56 else f'Need +{(0.56 - study.best_value)*100:.2f}%'}")

print("\nNext steps:")
print("  1. Calibrate probabilities (isotonic/Platt)")
print("  2. Walk-forward backtest on 2024-25 season")
print("  3. Test Kelly sizing with calibrated probabilities")
print("  4. Monitor constraint violations (should be 0)")
print("="*70)
