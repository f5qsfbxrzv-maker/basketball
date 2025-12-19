"""
Optuna Hyperparameter Tuning for 25-Feature Model
Conservative tuning strategy with 3000 trials for moneyline prediction
Features: 24 matchup features + injury_matchup_advantage (optimized)
"""

import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(r"c:\Users\d76do\OneDrive\Documents\New Basketball Model")
DATA_PATH = PROJECT_ROOT / "data" / "training_data_matchup_with_injury_advantage.csv"
OUTPUT_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature list (25 total)
FEATURES = [
    'home_composite_elo',
    'away_composite_elo',
    'off_elo_diff',
    'def_elo_diff',
    'net_fatigue_score',
    'ewma_efg_diff',
    'ewma_pace_diff',
    'ewma_tov_diff',
    'ewma_orb_diff',
    'ewma_vol_3p_diff',
    'injury_impact_diff',
    'injury_shock_diff',
    'star_mismatch',
    'ewma_chaos_home',
    'ewma_foul_synergy_home',
    'total_foul_environment',
    'league_offensive_context',
    'season_progress',
    'pace_efficiency_interaction',
    'projected_possession_margin',
    'three_point_matchup',
    'net_free_throw_advantage',
    'star_power_leverage',
    'offense_vs_defense_matchup',
    'injury_matchup_advantage'  # NEW: Optimized injury metric
]

TARGET = 'target_moneyline_win'

# Conservative hyperparameter ranges
PARAM_RANGES = {
    'max_depth': (3, 5),  # Shallow trees
    'min_child_weight': (25, 75),  # Aggressive pruning
    'gamma': (2.0, 10.0),  # Strong split requirement
    'learning_rate': (0.001, 0.02),  # Very slow learning
    'n_estimators': (5000, 12000),  # Many weak learners
    'subsample': (0.5, 0.7),  # Bagging
    'colsample_bytree': (0.5, 0.7),  # Feature sampling
    'reg_alpha': (5.0, 20.0),  # L1 regularization (strong)
}

# Fixed parameters
FIXED_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0
}


def load_data():
    """Load training data with temporal ordering"""
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure temporal ordering
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} games")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Features: {len(FEATURES)}")
    print(f"Target: {TARGET} (balance: {df[TARGET].mean():.3f})")
    
    # Verify all features exist
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    X = df[FEATURES].values
    y = df[TARGET].values
    
    return X, y, df


def objective(trial, X, y):
    """Optuna objective function with time-series cross-validation"""
    
    # Sample hyperparameters from conservative ranges
    params = {
        'max_depth': trial.suggest_int('max_depth', *PARAM_RANGES['max_depth']),
        'min_child_weight': trial.suggest_int('min_child_weight', *PARAM_RANGES['min_child_weight']),
        'gamma': trial.suggest_float('gamma', *PARAM_RANGES['gamma']),
        'learning_rate': trial.suggest_float('learning_rate', *PARAM_RANGES['learning_rate'], log=True),
        'n_estimators': trial.suggest_int('n_estimators', *PARAM_RANGES['n_estimators']),
        'subsample': trial.suggest_float('subsample', *PARAM_RANGES['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *PARAM_RANGES['colsample_bytree']),
        'reg_alpha': trial.suggest_float('reg_alpha', *PARAM_RANGES['reg_alpha']),
        **FIXED_PARAMS
    }
    
    # Time-series split (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = xgb.XGBClassifier(**params, early_stopping_rounds=100)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict and score
        y_pred = model.predict_proba(X_val)[:, 1]
        logloss = log_loss(y_val, y_pred)
        cv_scores.append(logloss)
    
    # Return mean CV log loss
    mean_logloss = np.mean(cv_scores)
    
    return mean_logloss


def run_optimization(X, y, n_trials=3000):
    """Run Optuna optimization"""
    
    print(f"\n{'='*70}")
    print(f"STARTING OPTUNA OPTIMIZATION - {n_trials} TRIALS")
    print(f"{'='*70}")
    print(f"Strategy: Conservative (shallow trees, strong regularization)")
    print(f"Objective: Minimize log loss (time-series CV)")
    print(f"Folds: 5 (TimeSeriesSplit)")
    print()
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Use 1 for stability with XGBoost
    )
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best log loss: {study.best_value:.6f}")
    print(f"Best trial: #{study.best_trial.number}")
    print()
    print("Best hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param:20s}: {value}")
    
    return study


def train_final_model(X, y, best_params):
    """Train final model on all data with best hyperparameters"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING FINAL MODEL")
    print(f"{'='*70}")
    
    params = {**best_params, **FIXED_PARAMS}
    
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=True)
    
    # Evaluate on full training set
    y_pred = model.predict_proba(X)[:, 1]
    
    metrics = {
        'log_loss': log_loss(y, y_pred),
        'auc': roc_auc_score(y, y_pred),
        'brier_score': brier_score_loss(y, y_pred),
        'accuracy': (model.predict(X) == y).mean()
    }
    
    print("\nTraining Set Performance:")
    print(f"  Log Loss:    {metrics['log_loss']:.6f}")
    print(f"  AUC:         {metrics['auc']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.6f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    
    return model, metrics


def extract_feature_importance(model):
    """Extract and sort feature importance"""
    
    importance = model.feature_importances_
    feature_df = pd.DataFrame({
        'feature': FEATURES,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    for idx, row in feature_df.head(10).iterrows():
        print(f"  {row['feature']:35s}: {row['importance']:.1f}")
    
    # Check injury_matchup_advantage rank
    injury_rank = feature_df[feature_df['feature'] == 'injury_matchup_advantage'].index[0] + 1
    injury_importance = feature_df[feature_df['feature'] == 'injury_matchup_advantage']['importance'].values[0]
    print(f"\ninjury_matchup_advantage: Rank #{injury_rank}, Importance {injury_importance:.1f}")
    
    return feature_df


def save_results(study, model, metrics, feature_importance):
    """Save model, study, and results"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = OUTPUT_DIR / f"xgboost_25features_optuna_{timestamp}.json"
    model.save_model(model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save study
    study_path = OUTPUT_DIR / f"optuna_study_{timestamp}.pkl"
    optuna.study.save_study(study, study_path)
    print(f"Study saved: {study_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'n_trials': len(study.trials),
        'best_trial': study.best_trial.number,
        'best_logloss': study.best_value,
        'best_params': study.best_params,
        'final_metrics': metrics,
        'n_features': len(FEATURES),
        'features': FEATURES,
        'target': TARGET,
        'data_path': str(DATA_PATH),
        'param_ranges': {k: list(v) for k, v in PARAM_RANGES.items()},
    }
    
    metadata_path = OUTPUT_DIR / f"metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
    # Save feature importance
    importance_path = OUTPUT_DIR / f"feature_importance_{timestamp}.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved: {importance_path}")
    
    return {
        'model_path': model_path,
        'study_path': study_path,
        'metadata_path': metadata_path,
        'importance_path': importance_path
    }


def main():
    """Main execution"""
    
    print(f"\n{'='*70}")
    print(f"OPTUNA TUNING - 25 FEATURES + injury_matchup_advantage")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    X, y, df = load_data()
    
    # Run optimization (3000 trials)
    study = run_optimization(X, y, n_trials=3000)
    
    # Train final model with best params
    model, metrics = train_final_model(X, y, study.best_params)
    
    # Extract feature importance
    feature_importance = extract_feature_importance(model)
    
    # Save everything
    paths = save_results(study, model, metrics, feature_importance)
    
    print(f"\n{'='*70}")
    print(f"TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFiles created:")
    for name, path in paths.items():
        print(f"  {name:20s}: {path.name}")
    print()


if __name__ == "__main__":
    main()
