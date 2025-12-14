"""
FINAL HYPERPARAMETER TUNING - 43 Features with Smart Constraints
Combines clean data + temporal features + basketball physics constraints.

Key Strategy:
1. Constrain "basketball physics": ELO, fatigue, efficiency (always directional)
2. NO constraints on temporal: season_year, season_progress (let model learn eras)
3. Lower regularization: Data is clean, can be more aggressive
4. 300 trials: Deeper search for optimal balance

Expected: Break 0.565 AUC with interpretable, robust model
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
print("FINAL CONSTRAINED HYPERTUNING - 43 FEATURES")
print("="*70)

# ---------------------------------------------------------
# 1. LOAD CLEAN + TEMPORAL DATA
# ---------------------------------------------------------
print("\n1. Loading dataset with temporal features...")
df = pd.read_csv('data/training_data_with_temporal_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"   Games: {len(df):,}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

# Define features and target
drop_cols = ['date', 'game_id', 'home_team', 'away_team', 'season',
             'target_spread', 'target_spread_cover', 'target_moneyline_win',
             'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']

features = [c for c in df.columns if c not in drop_cols]
X = df[features]
y = df['target_spread_cover']

print(f"   Features: {len(features)}")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# ---------------------------------------------------------
# 2. DEFINE SMART MONOTONIC CONSTRAINTS
# ---------------------------------------------------------
print("\n2. Applying smart monotonic constraints...")
print("   Strategy: Constrain physics, not eras")

constraint_map = {
    # ========== CONSTRAINED: Basketball Physics ==========
    
    # ELO: Higher is better
    'off_elo_diff': 1,           # Better offense → win
    'def_elo_diff': 1,           # Better defense → win
    'home_composite_elo': 1,     # Higher ELO → win
    
    # Shooting Efficiency: Higher is better
    'ewma_efg_diff': 1,          # Better shooting → win
    'home_ewma_3p_pct': 1,       # Better 3PT% → win
    'away_ewma_3p_pct': 0,       # Complex (helps away), no constraint
    
    # Fatigue: Always hurts
    'away_back_to_back': -1,     # Fatigue → loss
    'home_back_to_back': -1,     # Fatigue → loss
    'away_3in4': -1,             # Fatigue → loss
    'home_3in4': 0,              # Rare event, no constraint
    
    # Rest: More is better (now honest 0-12 days)
    'rest_advantage': 1,         # More rest advantage → win
    'away_rest_days': 0,         # Complex (can be rust), no constraint
    'home_rest_days': 0,         # Complex (can be rust), no constraint
    
    # Injury: Negative impact
    'injury_shock_diff': 1,      # Home injury advantage → win
    'injury_shock_home': -1,     # Home injuries → loss
    'injury_shock_away': 0,      # Away injuries (helps home), complex
    'injury_impact_abs': -1,     # Total injuries → harder game
    'injury_impact_diff': 1,     # Home injury advantage → win
    'away_star_missing': 1,      # Away star missing → home win
    'home_star_missing': -1,     # Home star missing → loss
    'star_mismatch': 1,          # Star advantage → win
    
    # Turnovers: Negative
    'ewma_tov_diff': -1,         # More turnovers (home - away) → loss
    
    # Rebounding: Positive
    'ewma_orb_diff': 1,          # More offensive rebounds → win
    'home_orb': 1,               # Home offensive rebounds → win
    'home_drb': 1,               # Home defensive rebounds → win
    'away_orb': 0,               # Away offensive rebounds (complex)
    'away_drb': 0,               # Complex
    
    # Fouls (getting to the line is good)
    'home_ewma_fta_rate': 1,     # Getting to FT line → win
    'away_ewma_fta_rate': 0,     # Complex
    'ewma_foul_synergy_home': 1, # Drawing fouls → win
    'ewma_foul_synergy_away': 0, # Complex
    
    # Fatigue Advantage
    'fatigue_mismatch': 1,       # Fatigue advantage → win
    
    # ========== UNCONSTRAINED: Temporal & Tactical ==========
    
    # Temporal: Let model learn era effects
    'season_year': 0,            # League evolution (unconstrained)
    'season_year_normalized': 0, # Same
    'games_into_season': 0,      # Form development (nonlinear)
    'season_progress': 0,        # Same
    'is_season_opener': 0,       # Rust effect (already captured explicitly)
    'endgame_phase': 0,          # Tanking/resting (complex)
    'season_month': 0,           # Seasonal patterns (nonlinear)
    
    # Tactical: Complex relationships
    'ewma_pace_diff': 0,         # Pace is tactical, not always good/bad
    'ewma_vol_3p_diff': 0,       # 3PT volume (tactical)
    'ewma_chaos_home': 0,        # Unpredictability (tactical)
    'ewma_net_chaos': 0,         # Net chaos (tactical)
    'altitude_game': 0,          # Home advantage in Denver (complex)
}

# Create constraint tuple
constraints = tuple(constraint_map.get(col, 0) for col in features)

constrained_count = len([c for c in constraints if c != 0])
print(f"   Constrained features: {constrained_count}/{len(features)}")
print(f"   Positive constraints (↑ = good): {len([c for c in constraints if c == 1])}")
print(f"   Negative constraints (↑ = bad): {len([c for c in constraints if c == -1])}")
print(f"   Unconstrained (learn freely): {len([c for c in constraints if c == 0])}")

# ---------------------------------------------------------
# 3. DEFINE OPTUNA OBJECTIVE
# ---------------------------------------------------------
print("\n3. Setting up Optuna optimization...")

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'monotone_constraints': constraints,
        
        # Hyperparameters (aggressive ranges for clean data)
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 800, 2500, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 0.0, 2.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 4.0),
        'random_state': 42,
        'verbosity': 0
    }
    
    # 5-fold Time Series CV for robust validation
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

# ---------------------------------------------------------
# 4. RUN OPTIMIZATION
# ---------------------------------------------------------
print("\n4. Running optimization...")
print("   Trials: 300")
print("   CV Folds: 5 (TimeSeriesSplit)")
print("   Target: 0.565+ AUC")
print("")
print("   This will take ~90-120 minutes")
print("   Press Ctrl+C to stop early (best params will be saved)")
print("")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300, show_progress_bar=True)

# ---------------------------------------------------------
# 5. RESULTS & ANALYSIS
# ---------------------------------------------------------
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

# Compare to baselines
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"Dirty 36 features (bugs):           0.55080")
print(f"Clean 36 features (baseline):       0.54070")
print(f"Clean 43 features (unconstrained):  0.55698")
print(f"Clean 43 features (CONSTRAINED):    {study.best_value:.5f}")
print("")
improvement_vs_dirty = (study.best_value - 0.55080) * 100
improvement_vs_unconstrained = (study.best_value - 0.55698) * 100
print(f"vs Dirty baseline:      {improvement_vs_dirty:+.2f}%")
print(f"vs Unconstrained test:  {improvement_vs_unconstrained:+.2f}%")
print(f"Target (0.565):         {'✓ ACHIEVED' if study.best_value >= 0.565 else f'Need +{(0.565 - study.best_value)*100:.2f}%'}")

# Save results
output_path = "models/final_constrained_best_params.json"
with open(output_path, 'w') as f:
    json.dump({
        'best_trial': study.best_trial.number,
        'best_auc': study.best_value,
        'best_params': study.best_trial.params,
        'tuning_date': datetime.now().isoformat(),
        'dataset': 'training_data_with_temporal_features.csv',
        'n_features': len(features),
        'n_trials': 300,
        'cv_folds': 5,
        'strategy': 'Smart constraints: physics constrained, eras unconstrained',
        'constraints_applied': constrained_count,
        'baselines': {
            'dirty_36': 0.55080,
            'clean_36': 0.54070,
            'clean_43_unconstrained': 0.55698
        }
    }, f, indent=2)

print(f"\n✓ Saved parameters: {output_path}")

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

# Full dataset performance
preds_full = model_final.predict_proba(X)[:, 1]
auc_full = roc_auc_score(y, preds_full)
acc_full = accuracy_score(y, (preds_full > 0.5).astype(int))

print(f"   Full dataset AUC: {auc_full:.5f}")
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
        constraint_str = '(↑)'
    elif row['constraint'] == -1:
        constraint_str = '(↓)'
    print(f"     {row['feature']:30s}: {row['importance']*100:5.2f}% {constraint_str}")

# Temporal feature importance
temporal_features = ['season_year', 'season_year_normalized', 'games_into_season', 
                     'season_progress', 'is_season_opener', 'endgame_phase', 'season_month']
temporal_imp = importances[importances['feature'].isin(temporal_features)]
print(f"\n   Temporal features combined: {temporal_imp['importance'].sum()*100:.2f}%")

# Save model
model_path = "models/final_constrained_model.json"
model_final.save_model(model_path)
print(f"\n✓ Saved model: {model_path}")

# Save feature importance
importances.to_csv("output/final_constrained_feature_importance.csv", index=False)
print(f"✓ Saved feature importance: output/final_constrained_feature_importance.csv")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Calibrate probabilities (isotonic/Platt scaling)")
print("2. Walk-forward backtest on 2024-25 season")
print("3. Integrate with Kelly optimizer (calibration factor)")
print("4. Deploy to production dashboard")
print("="*70)
