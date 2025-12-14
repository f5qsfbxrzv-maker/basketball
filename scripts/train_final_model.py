"""
Train final model with best hyperparameters from Optuna.
"""

import pandas as pd
import xgboost as xgb
import json
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("TRAINING FINAL MODEL")
    logger.info("="*60)
    
    # Load best params
    logger.info("\nLoading best hyperparameters...")
    with open("output/optuna_best_params.json", "r") as f:
        best_params = json.load(f)
    
    best_params['random_state'] = 42
    best_params['tree_method'] = 'hist'
    best_params['eval_metric'] = 'logloss'
    
    logger.info("Best parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key:20s} {value}")
    
    # Load data
    logger.info("\nLoading training data...")
    df = pd.read_csv("data/training_data_with_injury_shock.csv")
    logger.info(f"  Total games: {len(df):,}")
    
    # Get features
    exclude_cols = ['date', 'game_id', 'home_team', 'away_team', 'season', 
                   'target_spread', 'target_spread_cover', 'target_moneyline_win', 
                   'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    logger.info(f"  Features: {len(feature_cols)}")
    
    X = df[feature_cols]
    y = df['target_moneyline_win']
    
    # Split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"  Train: {len(X_train):,} games")
    logger.info(f"  Test:  {len(X_test):,} games")
    
    # Train model
    logger.info("\nTraining model...")
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=True)
    
    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"AUC:      {auc:.4f}")
    
    # Feature importance
    logger.info("\n" + "="*60)
    logger.info("TOP 15 FEATURES")
    logger.info("="*60)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    # Save model
    logger.info("\nSaving model...")
    joblib.dump(model, "models/xgboost_final.pkl")
    logger.info("Model saved to: models/xgboost_final.pkl")
    
    # Save feature importance
    importance_df.to_csv("output/feature_importance_final.csv", index=False)
    logger.info("Feature importance saved to: output/feature_importance_final.csv")

if __name__ == "__main__":
    main()
