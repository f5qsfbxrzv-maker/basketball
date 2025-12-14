"""
Optuna hyperparameter tuning for 36-feature model.
Optimizes XGBoost parameters using TimeSeriesSplit validation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
import optuna
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def objective(trial):
    """Optuna objective function."""
    
    # Load data
    df = pd.read_csv("data/training_data_with_features.csv")
    
    # Features from whitelist
    feature_cols = [c for c in df.columns if c not in [
        'game_id', 'date', 'home_team', 'away_team', 'season',
        'target_spread', 'target_spread_cover', 'target_moneyline_win',
        'target_game_total', 'target_over_under'
    ]]
    
    X = df[feature_cols]
    y = df['target_moneyline_win']
    
    # Suggest hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 20.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }
    
    # Time series cross-validation (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, y_pred_proba)
        scores.append(score)
    
    return np.mean(scores)

def main():
    logger.info("Starting Optuna hyperparameter optimization...")
    logger.info("This will take 1-2 hours for 100 trials\n")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    logger.info(f"\nBest trial:")
    logger.info(f"  Value (log loss): {study.best_trial.value:.4f}")
    logger.info(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save best params
    joblib.dump(study.best_params, "models/optuna_best_params_36features.pkl")
    logger.info("\n✅ Saved best params to models/optuna_best_params_36features.pkl")

if __name__ == "__main__":
    main()
