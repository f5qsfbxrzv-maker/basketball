"""
CONSERVATIVE OPTUNA TUNING - 3000 TRIALS
=========================================

Hyperparameter constraints designed for maximum generalization:
- Shallow trees (max_depth 3-5): Prevent memorization
- Heavy pruning (min_child_weight 25-75, gamma 2-10): Kill noise
- Slow learning (lr 0.001-0.02, n_estimators 5000-12000): Stable convergence
- High randomness (subsample/colsample 0.5-0.7): Anti-overfitting
- Strong regularization (reg_alpha 5-20): Force feature selection

This is designed to train a model that:
1. Generalizes to unseen data
2. Doesn't memorize specific game scenarios
3. Relies on robust statistical patterns only
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path("data/training_data_with_injury_shock.csv")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
STUDY_NAME = f"conservative_3000trials_{datetime.now().strftime('%Y%m%d_%H%M')}"
DB_PATH = OUTPUT_DIR / f"{STUDY_NAME}.db"

def objective(trial):
    """
    Conservative objective function with strict anti-overfitting constraints
    """
    
    # CONSERVATIVE HYPERPARAMETER RANGES
    params = {
        # Tree Structure (SHALLOW - prevents memorization)
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        
        # Pruning & Noise Filtering (AGGRESSIVE)
        'min_child_weight': trial.suggest_int('min_child_weight', 25, 75),
        'gamma': trial.suggest_float('gamma', 2.0, 10.0),
        
        # Learning Speed (SLOW & STEADY)
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.02, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 5000, 12000),
        
        # Randomness (ANTI-MEMORIZATION)
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7),
        
        # Regularization (THE STRAIGHTJACKET)
        'reg_alpha': trial.suggest_float('reg_alpha', 5.0, 20.0),  # L1 - force feature selection
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),  # L2 - weight shrinkage
        
        # Fixed parameters
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist',
        'enable_categorical': False,
    }
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Calculate injury_matchup_advantage (the new feature)
    df['injury_matchup_advantage'] = (
        0.008127 * df['injury_impact_diff']
      - 0.023904 * df['injury_shock_diff']
      + 0.031316 * df['star_mismatch']
    )
    
    # Feature set: Use features from training_data_with_injury_shock.csv
    # This dataset has the 42 EWMA-based features from the 43-feature model
    feature_cols = [
        # EWMA Four Factors & Shooting
        'ewma_efg_diff', 'ewma_tov_diff', 'ewma_orb_diff', 'ewma_pace_diff',
        'ewma_vol_3p_diff', 'home_ewma_3p_pct', 'away_ewma_3p_pct', 'away_ewma_tov_pct',
        
        # Rebounding & Free Throws
        'home_orb', 'away_orb', 'away_ewma_fta_rate',
        
        # Foul Environment & Chaos
        'ewma_foul_synergy_home', 'ewma_foul_synergy_away', 'total_foul_environment',
        'ewma_chaos_home', 'ewma_net_chaos',
        
        # Rest & Fatigue
        'home_rest_days', 'away_rest_days', 'rest_advantage', 'fatigue_mismatch',
        'home_back_to_back', 'away_back_to_back', 'home_3in4', 'away_3in4',
        
        # ELO Ratings
        'home_composite_elo', 'off_elo_diff', 'def_elo_diff',
        
        # Altitude
        'altitude_game',
        
        # Existing Injury Features (keep for backward compatibility)
        'injury_impact_diff', 'injury_impact_abs',
        'injury_shock_home', 'injury_shock_away', 'injury_shock_diff',
        'home_star_missing', 'away_star_missing', 'star_mismatch',
        
        # NEW OPTIMIZED INJURY FEATURE
        'injury_matchup_advantage'
    ]
    
    # Verify all features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        return 0.0
    
    X = df[feature_cols]
    y = df['target_moneyline_win']
    
    # Time series cross-validation (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        logloss = log_loss(y_val, y_pred_proba)
        
        fold_scores.append(auc)
        
        # Report intermediate values for pruning
        trial.report(auc, fold)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return mean AUC across folds
    mean_auc = np.mean(fold_scores)
    return mean_auc

def main():
    """Run conservative hyperparameter tuning"""
    
    logger.info("=" * 80)
    logger.info("CONSERVATIVE OPTUNA TUNING - 3000 TRIALS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Data:        {DATA_PATH}")
    logger.info(f"  Trials:      3000")
    logger.info(f"  CV Folds:    5 (TimeSeriesSplit)")
    logger.info(f"  Metric:      ROC AUC")
    logger.info(f"  Database:    {DB_PATH}")
    logger.info("")
    logger.info("Hyperparameter Constraints (CONSERVATIVE):")
    logger.info("  max_depth:         3-5    (shallow trees)")
    logger.info("  min_child_weight:  25-75  (aggressive pruning)")
    logger.info("  gamma:             2-10   (split penalty)")
    logger.info("  learning_rate:     0.001-0.02 (slow learning)")
    logger.info("  n_estimators:      5000-12000 (many weak learners)")
    logger.info("  subsample:         0.5-0.7 (high randomness)")
    logger.info("  colsample_bytree:  0.5-0.7 (feature randomness)")
    logger.info("  reg_alpha:         5-20  (L1 regularization)")
    logger.info("  reg_lambda:        1-10  (L2 regularization)")
    logger.info("")
    
    # Create study with pruning
    study = optuna.create_study(
        direction='maximize',
        study_name=STUDY_NAME,
        storage=f'sqlite:///{DB_PATH}',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=50, n_warmup_steps=2)
    )
    
    logger.info("Starting optimization...")
    logger.info("=" * 80)
    logger.info("")
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=3000,
        show_progress_bar=True,
        n_jobs=1  # Sequential for stability
    )
    
    # Save results
    results_path = OUTPUT_DIR / f"{STUDY_NAME}_results.json"
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best AUC:   {study.best_value:.6f}")
    logger.info("")
    logger.info("BEST HYPERPARAMETERS:")
    logger.info("-" * 40)
    for key, value in study.best_params.items():
        logger.info(f"  {key:20s}: {value}")
    logger.info("")
    
    # Save study
    joblib.dump(study, OUTPUT_DIR / f"{STUDY_NAME}_study.pkl")
    logger.info(f"Study saved:   {OUTPUT_DIR / f'{STUDY_NAME}_study.pkl'}")
    logger.info(f"Database:      {DB_PATH}")
    logger.info("")
    
    # Train final model with best params
    logger.info("Training final model with best parameters...")
    df = pd.read_csv(DATA_PATH)
    
    # Add injury feature
    df['injury_matchup_advantage'] = (
        0.008127 * df['injury_impact_diff']
      - 0.023904 * df['injury_shock_diff']
      + 0.031316 * df['star_mismatch']
    )
    
    feature_cols = [
        'vs_efg_diff', 'vs_tov', 'vs_reb_diff', 'vs_ftr_diff', 'vs_net_rating',
        'expected_pace', 'rest_days_diff', 'is_b2b_diff', 'h2h_win_rate_l3y',
        'elo_diff', 'off_elo_diff', 'def_elo_diff', 'composite_elo_diff',
        'h_off_rating', 'h_def_rating', 'a_off_rating', 'a_def_rating',
        'sos_diff', 'injury_impact_diff',
        'injury_matchup_advantage'
    ]
    
    X = df[feature_cols]
    y = df['target_moneyline_win']
    
    # Train on full dataset
    final_model = xgb.XGBClassifier(**study.best_params)
    final_model.fit(X, y, verbose=False)
    
    # Save model
    model_path = OUTPUT_DIR / f"{STUDY_NAME}_model.json"
    final_model.save_model(model_path)
    logger.info(f"Model saved:   {model_path}")
    logger.info("")
    
    # Feature importance
    logger.info("FEATURE IMPORTANCE (Top 10):")
    logger.info("-" * 40)
    importance = final_model.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_importance[:10], 1):
        logger.info(f"  {i:2d}. {feat:30s} {score:8.1f}")
    logger.info("")
    
    # Check injury feature ranking
    injury_rank = next((i for i, (f, _) in enumerate(sorted_importance, 1) if f == 'injury_matchup_advantage'), None)
    if injury_rank:
        logger.info(f"✅ injury_matchup_advantage rank: {injury_rank} / {len(sorted_importance)}")
    else:
        logger.info("⚠️  injury_matchup_advantage not in top features")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 TUNING COMPLETE - CONSERVATIVE 20-FEATURE MODEL READY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
