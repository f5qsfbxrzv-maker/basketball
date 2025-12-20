"""
EXPERIMENTAL MODEL TRAINER
Safe training script that ONLY outputs to models/experimental/
Prevents accidental overwriting of production models

Usage:
    python train_experimental_variant.py --variant a
    python train_experimental_variant.py --variant b1
    python train_experimental_variant.py --variant b2
    python train_experimental_variant.py --variant c
    python train_experimental_variant.py --variant d
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import xgboost as xgb

# Safety check - ensure experimental directory exists
EXPERIMENTAL_DIR = Path("models/experimental")
PRODUCTION_DIR = Path("models")

if not EXPERIMENTAL_DIR.exists():
    raise RuntimeError(f"Experimental directory {EXPERIMENTAL_DIR} does not exist!")

# Trial 1306 baseline hyperparameters
TRIAL1306_PARAMS = {
    'max_depth': 3,
    'min_child_weight': 25,
    'gamma': 5.162427047142856,
    'learning_rate': 0.010519422544676995,
    'n_estimators': 9947,
    'subsample': 0.6277685565263181,
    'colsample_bytree': 0.6014538139159614,
    'reg_alpha': 6.193992559265241,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42
}

# Trial 1306 full feature set (baseline)
TRIAL1306_FEATURES = [
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
    'injury_matchup_advantage',
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
    'offense_vs_defense_matchup'
]

# Variant feature sets
VARIANT_FEATURES = {
    'a': [f for f in TRIAL1306_FEATURES if f not in ['ewma_orb_diff', 'ewma_tov_diff']],
    'b1': [f for f in TRIAL1306_FEATURES if f != 'home_composite_elo'],
    'b2': [f for f in TRIAL1306_FEATURES if f != 'away_composite_elo'],
    'b3': [f for f in TRIAL1306_FEATURES if f not in ['home_composite_elo', 'away_composite_elo']],
    'c': [f for f in TRIAL1306_FEATURES if f not in ['ewma_foul_synergy_home', 'net_free_throw_advantage']],
    'd': [f for f in TRIAL1306_FEATURES if f not in [
        'ewma_orb_diff', 'ewma_tov_diff', 'away_composite_elo',
        'ewma_foul_synergy_home'
    ]]
}

VARIANT_DESCRIPTIONS = {
    'baseline': 'Trial 1306 baseline (22 features)',
    'a': 'Remove possession components (20 features: keep projected_possession_margin)',
    'b1': 'Remove home_composite_elo (21 features: keep away + diffs)',
    'b2': 'Remove away_composite_elo (21 features: keep home + diffs)',
    'b3': 'Remove both composite ELOs (20 features: diffs only)',
    'c': 'Consolidate foul features (20 features: keep total_foul_environment)',
    'd': 'Full pruning (18 features: keep net_free_throw_advantage for testing)'
}


def safe_output_path(variant: str, extension: str = 'json') -> Path:
    """Generate safe output path in experimental directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"xgboost_variant_{variant}_{timestamp}.{extension}"
    output_path = EXPERIMENTAL_DIR / filename
    
    # Triple check we're in experimental directory
    if not str(output_path).startswith(str(EXPERIMENTAL_DIR)):
        raise RuntimeError(f"SAFETY VIOLATION: Output path {output_path} not in experimental directory!")
    
    return output_path


def load_data():
    """Load training data"""
    data_path = Path("data/training_data_GOLD_ELO_22_features.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} rows")
    
    return df


def train_variant(variant: str, features: list, params: dict, df: pd.DataFrame):
    """Train model with specified feature set"""
    
    print("\n" + "="*70)
    print(f"TRAINING VARIANT: {variant.upper()}")
    print("="*70)
    print(f"Description: {VARIANT_DESCRIPTIONS.get(variant, 'Unknown')}")
    print(f"Features: {len(features)}")
    print(f"Removed from baseline: {set(TRIAL1306_FEATURES) - set(features)}")
    print("="*70)
    
    # Check all features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing}")
    
    # Prepare data
    X = df[features].copy()
    y = df['target_moneyline_win'].copy()
    
    # Remove NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"\n✓ Training samples: {len(X):,}")
    print(f"✓ Positive class: {y.sum():,} ({y.mean()*100:.1f}%)")
    
    # Time-series cross-validation (5 folds)
    print("\n" + "="*70)
    print("CROSS-VALIDATION (Time Series Split)")
    print("="*70)
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        fold_log_loss = log_loss(y_val, y_pred_proba)
        fold_accuracy = accuracy_score(y_val, y_pred)
        fold_auc = roc_auc_score(y_val, y_pred_proba)
        
        cv_results.append({
            'fold': fold,
            'log_loss': fold_log_loss,
            'accuracy': fold_accuracy,
            'auc': fold_auc,
            'train_size': len(X_train),
            'val_size': len(X_val)
        })
        
        print(f"Fold {fold}: Log Loss={fold_log_loss:.4f}, Accuracy={fold_accuracy:.4f}, AUC={fold_auc:.4f}")
    
    # Aggregate CV results
    cv_log_loss = np.mean([r['log_loss'] for r in cv_results])
    cv_accuracy = np.mean([r['accuracy'] for r in cv_results])
    cv_auc = np.mean([r['auc'] for r in cv_results])
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"Mean Log Loss: {cv_log_loss:.4f}")
    print(f"Mean Accuracy: {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")
    print(f"Mean AUC:      {cv_auc:.4f}")
    
    # Compare to baseline
    baseline_log_loss = 0.6330
    baseline_accuracy = 0.6389
    
    print("\n" + "="*70)
    print("COMPARISON TO TRIAL 1306 BASELINE")
    print("="*70)
    print(f"Log Loss:  {cv_log_loss:.4f} vs {baseline_log_loss:.4f} "
          f"({(cv_log_loss - baseline_log_loss)*1000:+.1f}e-3)")
    print(f"Accuracy:  {cv_accuracy:.4f} vs {baseline_accuracy:.4f} "
          f"({(cv_accuracy - baseline_accuracy)*100:+.2f}%)")
    
    delta_log_loss = cv_log_loss - baseline_log_loss
    delta_accuracy = cv_accuracy - baseline_accuracy
    
    if abs(delta_log_loss) < 0.005 and abs(delta_accuracy) < 0.01:
        print("\n✅ PERFORMANCE: Within acceptable range")
    elif delta_log_loss > 0.005 or delta_accuracy < -0.01:
        print("\n⚠️  PERFORMANCE: Degradation detected")
    else:
        print("\n✨ PERFORMANCE: Improvement detected")
    
    # Train final model on full data
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL (full dataset)")
    print("="*70)
    
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)
    
    # Full dataset metrics
    y_pred_proba_full = final_model.predict_proba(X)[:, 1]
    y_pred_full = (y_pred_proba_full >= 0.5).astype(int)
    
    train_log_loss = log_loss(y, y_pred_proba_full)
    train_accuracy = accuracy_score(y, y_pred_full)
    train_auc = roc_auc_score(y, y_pred_proba_full)
    
    print(f"Train Log Loss: {train_log_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train AUC:      {train_auc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 10 FEATURES")
    print("="*70)
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:<35} {row['importance']:.4f}")
    
    # Save model
    model_path = safe_output_path(variant, 'json')
    final_model.save_model(str(model_path))
    print(f"\n✓ Saved model: {model_path}")
    
    # Save results
    results = {
        'variant': variant,
        'description': VARIANT_DESCRIPTIONS.get(variant, 'Unknown'),
        'timestamp': datetime.now().isoformat(),
        'n_features': len(features),
        'features': features,
        'removed_features': list(set(TRIAL1306_FEATURES) - set(features)),
        'hyperparameters': params,
        'cv_log_loss': float(cv_log_loss),
        'cv_accuracy': float(cv_accuracy),
        'cv_auc': float(cv_auc),
        'train_log_loss': float(train_log_loss),
        'train_accuracy': float(train_accuracy),
        'train_auc': float(train_auc),
        'n_samples': len(X),
        'fold_results': cv_results,
        'feature_importance': feature_importance.to_dict('records'),
        'baseline_comparison': {
            'baseline_log_loss': baseline_log_loss,
            'baseline_accuracy': baseline_accuracy,
            'delta_log_loss': float(delta_log_loss),
            'delta_accuracy': float(delta_accuracy),
            'within_tolerance': bool(abs(delta_log_loss) < 0.005 and abs(delta_accuracy) < 0.01)
        }
    }
    
    results_path = safe_output_path(variant, 'json').with_name(
        safe_output_path(variant, 'json').stem + '_results.json'
    )
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results: {results_path}")
    
    # Save feature importance
    importance_path = safe_output_path(variant, 'csv').with_name(
        safe_output_path(variant, 'csv').stem + '_importance.csv'
    )
    feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Saved importance: {importance_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train experimental model variant')
    parser.add_argument('--variant', type=str, required=True,
                      choices=['baseline', 'a', 'b1', 'b2', 'b3', 'c', 'd'],
                      help='Variant to train')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EXPERIMENTAL MODEL TRAINER")
    print("="*70)
    print(f"Variant: {args.variant}")
    print(f"Output directory: {EXPERIMENTAL_DIR}")
    print(f"Safety check: ✓ Experimental directory exists")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Get feature set
    if args.variant == 'baseline':
        features = TRIAL1306_FEATURES
    else:
        features = VARIANT_FEATURES[args.variant]
    
    # Train model
    results = train_variant(args.variant, features, TRIAL1306_PARAMS, df)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"Model variant: {args.variant}")
    print(f"CV Log Loss: {results['cv_log_loss']:.4f}")
    print(f"CV Accuracy: {results['cv_accuracy']:.4f}")
    print(f"Files saved to: {EXPERIMENTAL_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
