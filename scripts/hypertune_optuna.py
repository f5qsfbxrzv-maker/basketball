"""
Hyperparameter tuning with Optuna.
1000 trials with 5-fold TimeSeriesSplit cross-validation.
Deep hyperparameter search for optimal model performance.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def objective(trial):
    """Optuna objective function"""
    
    # Suggest hyperparameters (expanded search space)
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.15, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 8000),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 50),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 10, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.3),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }
    
    # Load data
    df = pd.read_csv("data/training_data_with_features.csv")
    
    # Get all feature columns (exclude metadata and targets)
    exclude_cols = ['date', 'game_id', 'home_team', 'away_team', 'season', 
                   'target_spread', 'target_spread_cover', 'target_moneyline_win', 
                   'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target_moneyline_win']
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        scores.append(score)
    
    return np.mean(scores)

def main():
    logger.info("="*60)
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info("="*60)
    logger.info("Trials: 100")
    logger.info("Cross-validation: 5-fold TimeSeriesSplit")
    logger.info("Metric: ROC AUC")
    logger.info("")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_nba_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization (1000 trials for deeper search)
    study.optimize(objective, n_trials=1000, show_progress_bar=True)
    
    # Save study
    joblib.dump(study, "output/optuna_study.pkl")
    logger.info(f"\nStudy saved to: output/optuna_study.pkl")
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info("BEST HYPERPARAMETERS")
    logger.info("="*60)
    for key, value in study.best_params.items():
        logger.info(f"  {key:20s} {value}")
    
    logger.info(f"\nBest AUC: {study.best_value:.4f}")
    
    # Save best params to file for next script
    import json
    with open("output/optuna_best_params_1000trials.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info("Best params saved to: output/optuna_best_params_1000trials.json")
    
    # Also save trial history
    df_history = study.trials_dataframe()
    df_history.to_csv("output/optuna_study_history_1000trials.csv", index=False)
    logger.info(f"Trial history saved ({len(df_history)} trials)")

if __name__ == "__main__":
    main()
