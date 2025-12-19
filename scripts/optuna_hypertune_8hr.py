"""
8-Hour Optuna Hyperparameter Tuning with CORRECTED 44-Feature Set
CRITICAL: Uses HISTORICAL injuries for training, NOT live injury data
"""

import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import warnings
import sys
import os
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

warnings.filterwarnings('ignore')

print("="*80)
print("8-HOUR OPTUNA HYPERTUNING - NBA BETTING MODEL")
print("="*80)
print(f"Start Time: {datetime.now()}")
print("Features: 44 (including away_composite_elo)")
print("Strategy: Deep Learning Rate + Heavy Regularization")
print("Validation: TimeSeriesSplit (5 folds)")
print("="*80)

# --- 1. DATA LOADING WITH HISTORICAL INJURIES ---
def load_and_prep_data():
    """
    Load training data with HISTORICAL injury data.
    CRITICAL: Must use injury data as it existed at game time, not current injuries.
    """
    
    print("\n[1/4] Loading training data...")
    
    # Check if training data exists
    data_path = 'data/training_data_with_temporal_features.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            "Run data preparation script first to generate historical features."
        )
    
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} games")
    
    # Sort by date for time-series split
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Define target and drop columns
    drop_cols = [
        'date', 'game_id', 'home_team', 'away_team', 'season',
        'target_spread', 'target_spread_cover', 'target_moneyline_win',
        'target_game_total', 'target_over_under', 'target_home_cover', 'target_over'
    ]
    
    # Target variable
    if 'target_spread_cover' not in df.columns:
        raise ValueError("target_spread_cover not found in data")
    
    y = df['target_spread_cover']
    
    # Features
    features = [c for c in df.columns if c not in drop_cols]
    X = df[features]
    
    print(f"  Features: {len(features)}")
    print(f"  Target: target_spread_cover ({y.sum():,} covers, {(~y.astype(bool)).sum():,} non-covers)")
    
    # CRITICAL CHECK: Verify away_composite_elo is present
    if 'away_composite_elo' not in features:
        print("\n  WARNING: away_composite_elo NOT in features!")
        print("  This training data was generated with the OLD feature set.")
        print("  You must regenerate training data with the corrected feature calculator.")
        raise ValueError("Training data missing away_composite_elo - regenerate with updated feature_calculator_live.py")
    
    if 'home_composite_elo' not in features:
        print("\n  WARNING: home_composite_elo NOT in features!")
        raise ValueError("Training data missing home_composite_elo")
    
    print(f"  OK - home_composite_elo present")
    print(f"  OK - away_composite_elo present")
    
    # Verify no NaN values
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        print("\n  WARNING: NaN values detected:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"    {col}: {count} NaNs")
        
        # Fill NaN with feature-appropriate defaults
        print("  Filling NaN values...")
        for col in X.columns:
            if X[col].isna().any():
                if 'elo' in col.lower():
                    X[col].fillna(1500, inplace=True)
                elif 'injury' in col.lower():
                    X[col].fillna(0, inplace=True)
                else:
                    X[col].fillna(X[col].median(), inplace=True)
    
    print(f"\n[1/4] OK - Data loaded: {X.shape[0]:,} games x {X.shape[1]} features")
    
    return X, y, features

# Load data once (saves memory during optimization)
X, y, feature_names = load_and_prep_data()

# --- 2. OPTUNA OBJECTIVE FUNCTION ---
def objective(trial):
    """
    Train one model configuration and return validation loss.
    Optuna minimizes this score.
    """
    
    # --- A. Hyperparameter Search Space ---
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'random_state': 42,
        
        # DEEP STRATEGY: Low learning rate forces more trees (better generalization)
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05, log=True),
        
        # Tree Complexity (prevent overfitting)
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        
        # Regularization (L1 + L2)
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),  # L2
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),    # L1
        
        # Sampling (add randomness for robustness)
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 5),  # Min loss reduction
        
        # High cap, early stopping will control actual count
        'n_estimators': 10000,
        'n_jobs': -1,
    }
    
    # --- B. Time-Series Cross-Validation (5 Folds) ---
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold_scores = []
    fold_aucs = []
    
    for fold_idx, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Convert to DMatrix for speed
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train with early stopping
        model = xgb.train(
            param,
            dtrain,
            num_boost_round=param['n_estimators'],
            evals=[(dval, 'val')],
            early_stopping_rounds=100,  # Stop if no improvement for 100 rounds
            verbose_eval=False
        )
        
        # Predict on validation fold
        preds = model.predict(dval)
        
        # Calculate metrics
        fold_logloss = log_loss(y_val, preds)
        fold_auc = roc_auc_score(y_val, preds)
        
        fold_scores.append(fold_logloss)
        fold_aucs.append(fold_auc)
        
        # --- C. Pruning: Stop early if this configuration is clearly bad ---
        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Return average log loss across all folds
    avg_logloss = np.mean(fold_scores)
    avg_auc = np.mean(fold_aucs)
    
    # Store AUC as user attribute for later analysis
    trial.set_user_attr('avg_auc', avg_auc)
    
    return avg_logloss

# --- 3. RUN 8-HOUR OPTIMIZATION ---
if __name__ == "__main__":
    
    print("\n[2/4] Initializing Optuna study...")
    
    # SQLite storage allows pause/resume if interrupted
    storage_path = "sqlite:///models/nba_optuna_44features.db"
    study_name = "nba_44features_deep_v1"
    
    study = optuna.create_study(
        direction="minimize",
        storage=storage_path,
        study_name=study_name,
        load_if_exists=True,  # Resume if study exists
        pruner=optuna.pruners.MedianPruner(  # Prune bad trials early
            n_startup_trials=5,
            n_warmup_steps=2
        )
    )
    
    print(f"  Study: {study_name}")
    print(f"  Storage: {storage_path}")
    print(f"  Direction: minimize log_loss")
    print(f"  Pruning: MedianPruner")
    
    # Check if resuming
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_completed > 0:
        print(f"\n  Resuming from {n_completed} completed trials")
        print(f"  Best so far: {study.best_value:.6f}")
    
    print("\n[3/4] Starting 8-hour optimization...")
    print("  This will take ~8 hours. You can stop and resume anytime.")
    print("  Press Ctrl+C to stop early (results will be saved)\n")
    
    # Custom callback to print AUC after each trial
    def print_trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            auc = trial.user_attrs.get('avg_auc', None)
            auc_str = f"{auc:.5f}" if auc else "N/A"
            print(f"  Trial {trial.number:3d}: LogLoss={trial.value:.6f}, AUC={auc_str}")
            if trial.number == study.best_trial.number:
                print(f"    NEW BEST!")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"  Trial {trial.number:3d}: PRUNED")
    
    # 8 hours = 28,800 seconds
    try:
        study.optimize(objective, timeout=28800, n_jobs=1, callbacks=[print_trial_callback])
    except KeyboardInterrupt:
        print("\n\n  Optimization interrupted by user. Saving results...")
    
    print("\n" + "="*80)
    print("[4/4] OPTIMIZATION COMPLETE")
    print("="*80)
    
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS FOUND:")
    print("="*80)
    
    best_params = study.best_params
    best_logloss = study.best_value
    best_auc = study.best_trial.user_attrs.get('avg_auc', None)
    
    for param, value in best_params.items():
        print(f"  {param:20s} = {value}")
    
    print(f"\nBest Log Loss: {best_logloss:.6f}")
    if best_auc:
        print(f"Best AUC:      {best_auc:.5f}")
    
    # Interpret log loss
    print("\nLog Loss Interpretation:")
    if best_logloss < 0.62:
        print("  < 0.62 = VEGAS LEVEL (Elite)")
    elif best_logloss < 0.65:
        print("  < 0.65 = Very Strong")
    elif best_logloss < 0.68:
        print("  < 0.68 = Decent Baseline")
    else:
        print("  > 0.68 = Needs Improvement")
    
    # Save results
    results = {
        'best_params': best_params,
        'best_logloss': best_logloss,
        'best_auc': best_auc,
        'n_features': len(feature_names),
        'features': feature_names,
        'n_trials': len(study.trials),
        'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'optimization_date': datetime.now().isoformat(),
        'dataset_size': len(X),
        'study_name': study_name
    }
    
    output_path = 'models/optuna_best_params_44features.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nOK - Results saved to: {output_path}")
    print(f"OK - Study database: {storage_path}")
    print(f"\nEnd Time: {datetime.now()}")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the best parameters above")
    print("2. Train final model with these params:")
    print("     python scripts/train_final_model_44features.py")
    print("3. Run calibration on the trained model")
    print("4. Backtest on hold-out data to verify performance")
    print("="*80)
