import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import sys
import joblib

# CONFIG
DATA_PATH = 'data/training_data_MDP_with_margins.csv'
N_TRIALS = 50  # 50 trials is sufficient to find the new "sweet spot"

# FEATURES (19 Variant D features available in MDP dataset)
ACTIVE_FEATURES = [
    'off_elo_diff', 'def_elo_diff', 'home_composite_elo',           
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',          
    'injury_matchup_advantage', 'injury_shock_diff', 'star_power_leverage',
    'season_progress', 'league_offensive_context',     
    'total_foul_environment', 'net_free_throw_advantage',             
    'offense_vs_defense_matchup', 'pace_efficiency_interaction', 'star_mismatch'
]

def objective(trial):
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # Validation: Ensure Margin Target Exists
    if 'margin_target' not in df.columns:
        # Fallback if user forgot to rebuild
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['margin_target'] = df['home_score'] - df['away_score']
        else:
            raise ValueError("Data missing 'margin_target'. Run build_mdp_training_data.py first.")
        
    X = df[ACTIVE_FEATURES]
    y = df['margin_target']

    # 2. Suggest Hyperparameters for REGRESSION
    param = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror', # Optimizing for Spread Accuracy
        'eval_metric': 'rmse',
        'n_jobs': -1,
        'random_state': 42,
        
        # Structure (Likely needs more depth than classifier)
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
        
        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000, step=100),
        
        # Regularization (Critical for Regression outliers)
        'gamma': trial.suggest_float('gamma', 0.1, 5.0),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
    }

    # 3. K-Fold Cross Validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42) # 3 splits for speed
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train
        model = xgb.train(param, dtrain, num_boost_round=param['n_estimators'], verbose_eval=False)
        
        # Predict
        preds = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)

if __name__ == "__main__":
    print("🚀 TUNING NEW MARGIN ARCHITECTURE...")
    print(f"   Target: Minimize RMSE (Point Spread Error)")
    print(f"   Data: {DATA_PATH}")
    print(f"   Trials: {N_TRIALS}")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "="*60)
    print("🏆 NEW BEST PARAMS (REGRESSOR)")
    print("="*60)
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    print(f"\nBest RMSE: {study.best_value:.4f}")
    
    # Save Best Params
    joblib.dump(study.best_params, 'best_params_margin.joblib')
    print("💾 Saved to best_params_margin.joblib")
