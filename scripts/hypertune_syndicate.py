"""
"Syndicate" Hyperparameter Tuning - Constrained Search
Forces exploration in proven parameter regions:
- Max depth: 6-12 (not 4-5)
- Gamma: 0-3 (not 4.5+)
- Colsample: 0.5-0.8 (feature selection)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def objective(trial):
    """Constrained objective forcing smarter configurations"""
    
    # CONSTRAINED search space based on manual success
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'max_depth': trial.suggest_int('max_depth', 6, 12),  # FORCED DEEPER
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 25),
        'gamma': trial.suggest_float('gamma', 0, 3.0),  # CAPPED at 3.0
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),  # FOCUSED RANGE
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.8),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 7),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 5, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.9, 1.1),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }
    
    # Load data
    df = pd.read_csv("data/training_data_with_features.csv")
    
    # Get features
    exclude_cols = ['date', 'game_id', 'home_team', 'away_team', 'season', 
                   'target_spread', 'target_spread_cover', 'target_moneyline_win', 
                   'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target_moneyline_win']
    
    # Time series cross-validation (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        auc_scores.append(auc)
    
    return np.mean(auc_scores)

def main():
    logger.info("="*60)
    logger.info("SYNDICATE HYPERPARAMETER TUNING (Constrained Search)")
    logger.info("="*60)
    logger.info("Trials: 1500")
    logger.info("Constraints:")
    logger.info("  - Max depth: 6-12 (forced deeper)")
    logger.info("  - Gamma: 0-3.0 (capped to prevent conservatism)")
    logger.info("  - Colsample: 0.5-0.8 (feature selection focus)")
    logger.info("")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_syndicate_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization (1500 trials - faster due to constraints)
    study.optimize(objective, n_trials=1500, show_progress_bar=True)
    
    # Save study
    joblib.dump(study, "output/syndicate_study.pkl")
    logger.info(f"\nStudy saved to: output/syndicate_study.pkl")
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info("BEST HYPERPARAMETERS (SYNDICATE)")
    logger.info("="*60)
    for key, value in study.best_params.items():
        logger.info(f"  {key:20s} {value}")
    
    logger.info(f"\nBest AUC: {study.best_value:.4f}")
    
    # Compare to unconstrained best
    logger.info(f"\nUNCONSTRAINED BEST (Trial 360): 0.5571")
    logger.info(f"SYNDICATE BEST:                  {study.best_value:.4f}")
    logger.info(f"Improvement: {(study.best_value - 0.5571)*100:.2f} percentage points")
    
    # Save best params
    import json
    with open("output/syndicate_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info("\nBest params saved to: output/syndicate_best_params.json")
    
    # Save trial history
    df_history = study.trials_dataframe()
    df_history.to_csv("output/syndicate_study_history.csv", index=False)
    logger.info(f"Trial history saved ({len(df_history)} trials)")

if __name__ == "__main__":
    main()
