"""
Retrain XGBoost model with NEW injury shock features.

This script:
1. Loads existing training data (already has baseline features)
2. Adds NEW injury shock features (injury_shock_*, home_star_missing, etc.)
3. Trains XGBoost with Optuna best params
4. Compares feature importance: OLD vs NEW
5. Saves model if injury features improved

Expected: Injury features move from #20-21 → Top 10
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import sys
import os
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.feature_calculator_v5 import FeatureCalculatorV5
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Optuna best hyperparameters (from walk-forward validation)
OPTUNA_BEST_PARAMS = {
    'learning_rate': 0.003,
    'n_estimators': 3731,
    'max_depth': 11,
    'min_child_weight': 3,
    'gamma': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 9.60,
    'reg_lambda': 1.5,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

def add_injury_shock_features(df, calculator):
    """Add injury shock features to existing dataframe."""
    logger.info("Adding injury shock features...")
    
    new_features = []
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            logger.info(f"  Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
        
        try:
            # Handle datetime
            if isinstance(row['date'], str):
                game_date = datetime.strptime(row['date'], '%Y-%m-%d')
            else:
                game_date = row['date']
            
            date_str = game_date.strftime('%Y-%m-%d')
            
            # Get existing injury impact
            home_injury = row.get('injury_impact_diff', 0) + row.get('injury_impact_abs', 0) / 2
            away_injury = row.get('injury_impact_abs', 0) / 2 - row.get('injury_impact_diff', 0) / 2
            
            # Calculate EWMA baseline
            home_ewma_inj = calculator._get_ewma_injury_impact(row['home_team'], date_str)
            away_ewma_inj = calculator._get_ewma_injury_impact(row['away_team'], date_str)
            
            # Shock features
            injury_shock_home = home_injury - home_ewma_inj
            injury_shock_away = away_injury - away_ewma_inj
            injury_shock_diff = injury_shock_home - injury_shock_away
            
            # Star binary flags
            STAR_THRESHOLD = 4.0
            home_star_missing = 1 if home_injury >= STAR_THRESHOLD else 0
            away_star_missing = 1 if away_injury >= STAR_THRESHOLD else 0
            star_mismatch = home_star_missing - away_star_missing
            
            new_features.append({
                'injury_shock_home': injury_shock_home,
                'injury_shock_away': injury_shock_away,
                'injury_shock_diff': injury_shock_diff,
                'home_star_missing': home_star_missing,
                'away_star_missing': away_star_missing,
                'star_mismatch': star_mismatch
            })
            
        except Exception as e:
            logger.debug(f"Error on row {idx}: {e}")
            new_features.append({
                'injury_shock_home': 0,
                'injury_shock_away': 0,
                'injury_shock_diff': 0,
                'home_star_missing': 0,
                'away_star_missing': 0,
                'star_mismatch': 0
            })
    
    # Add new features to dataframe
    new_features_df = pd.DataFrame(new_features)
    df = pd.concat([df.reset_index(drop=True), new_features_df], axis=1)
    
    logger.info(f"Added {len(new_features_df.columns)} injury shock features")
    return df

def train_and_compare():
    """Train models with and without injury shock features."""
    
    # Load training data
    logger.info("Loading training data...")
    df = pd.read_csv("data/training_data_with_features.csv")
    logger.info(f"  Loaded {len(df)} games")
    
    # Initialize calculator for EWMA injury calculation
    logger.info("Initializing FeatureCalculatorV5...")
    calculator = FeatureCalculatorV5()
    
    # Add injury shock features
    df = add_injury_shock_features(df, calculator)
    
    # Define feature sets
    baseline_features = [
        'vs_efg_diff', 'vs_tov', 'vs_reb_diff', 'vs_ftr_diff', 'vs_net_rating',
        'expected_pace', 'rest_days_diff', 'is_b2b_diff', 'h2h_win_rate_l3y',
        'injury_impact_diff', 'elo_diff', 'off_elo_diff', 'def_elo_diff',
        'composite_elo_diff', 'sos_diff', 'h_off_rating', 'h_def_rating',
        'a_off_rating', 'a_def_rating'
    ]
    
    injury_shock_features = [
        'injury_shock_home', 'injury_shock_away', 'injury_shock_diff',
        'home_star_missing', 'away_star_missing', 'star_mismatch'
    ]
    
    enhanced_features = baseline_features + injury_shock_features
    
    # Prepare data
    y = df['target_moneyline_win']
    
    # Train baseline model
    logger.info("\n" + "="*60)
    logger.info("BASELINE MODEL (without injury shock)")
    logger.info("="*60)
    
    X_baseline = df[baseline_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model_baseline = xgb.XGBClassifier(**OPTUNA_BEST_PARAMS)
    model_baseline.fit(X_train, y_train)
    
    y_pred_baseline = model_baseline.predict(X_test)
    y_proba_baseline = model_baseline.predict_proba(X_test)[:, 1]
    
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    auc_baseline = roc_auc_score(y_test, y_proba_baseline)
    
    logger.info(f"Accuracy: {acc_baseline:.4f}")
    logger.info(f"AUC: {auc_baseline:.4f}")
    
    # Feature importance baseline
    importance_baseline = pd.DataFrame({
        'feature': baseline_features,
        'importance': model_baseline.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Features (Baseline):")
    for idx, row in importance_baseline.head(10).iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    inj_rank_baseline = importance_baseline[
        importance_baseline['feature'] == 'injury_impact_diff'
    ].index[0] + 1
    logger.info(f"\nInjury feature rank: #{inj_rank_baseline}")
    
    # Train enhanced model
    logger.info("\n" + "="*60)
    logger.info("ENHANCED MODEL (with injury shock)")
    logger.info("="*60)
    
    X_enhanced = df[enhanced_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model_enhanced = xgb.XGBClassifier(**OPTUNA_BEST_PARAMS)
    model_enhanced.fit(X_train, y_train)
    
    y_pred_enhanced = model_enhanced.predict(X_test)
    y_proba_enhanced = model_enhanced.predict_proba(X_test)[:, 1]
    
    acc_enhanced = accuracy_score(y_test, y_pred_enhanced)
    auc_enhanced = roc_auc_score(y_test, y_proba_enhanced)
    
    logger.info(f"Accuracy: {acc_enhanced:.4f}")
    logger.info(f"AUC: {auc_enhanced:.4f}")
    
    # Feature importance enhanced
    importance_enhanced = pd.DataFrame({
        'feature': enhanced_features,
        'importance': model_enhanced.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Features (Enhanced):")
    for idx, row in importance_enhanced.head(15).iterrows():
        is_new = '🆕' if row['feature'] in injury_shock_features else '  '
        logger.info(f"{is_new} {row['feature']:30s} {row['importance']:.4f}")
    
    # Find injury feature ranks
    inj_features_in_top15 = importance_enhanced.head(15)['feature'].isin(injury_shock_features).sum()
    logger.info(f"\nInjury shock features in Top 15: {inj_features_in_top15}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"Baseline Accuracy:  {acc_baseline:.4f}")
    logger.info(f"Enhanced Accuracy:  {acc_enhanced:.4f}")
    logger.info(f"Improvement:        {(acc_enhanced - acc_baseline)*100:+.2f}%")
    logger.info(f"")
    logger.info(f"Baseline AUC:       {auc_baseline:.4f}")
    logger.info(f"Enhanced AUC:       {auc_enhanced:.4f}")
    logger.info(f"Improvement:        {(auc_enhanced - auc_baseline)*100:+.2f}%")
    
    # Save enhanced model if better
    if acc_enhanced > acc_baseline:
        logger.info("\n✅ Enhanced model is better! Saving...")
        model_path = "models/xgboost_with_injury_shock.pkl"
        joblib.dump(model_enhanced, model_path)
        logger.info(f"Saved to: {model_path}")
        
        # Save feature importance
        importance_enhanced.to_csv("output/feature_importance_injury_shock.csv", index=False)
        logger.info("Saved feature importance to: output/feature_importance_injury_shock.csv")
    else:
        logger.info("\n❌ Enhanced model not better. Keeping baseline.")
    
    return model_baseline, model_enhanced, importance_baseline, importance_enhanced

if __name__ == "__main__":
    train_and_compare()
