"""
Train final XGBoost model with Optuna-tuned hyperparameters on full 36-feature dataset.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading training data...")
    df = pd.read_csv("data/training_data_with_features.csv")
    
    # Features
    feature_cols = [c for c in df.columns if c not in [
        'game_id', 'date', 'home_team', 'away_team', 'season',
        'target_spread', 'target_spread_cover', 'target_moneyline_win',
        'target_game_total', 'target_over_under'
    ]]
    
    logger.info(f"Training on {len(feature_cols)} features: {feature_cols[:10]}...")
    
    X = df[feature_cols]
    y = df['target_moneyline_win']
    
    # Load Optuna best params
    logger.info("Loading Optuna best parameters...")
    best_params = joblib.load("models/optuna_best_params_36features.pkl")
    best_params['random_state'] = 42
    best_params['tree_method'] = 'hist'
    best_params['eval_metric'] = 'logloss'
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"\nTraining XGBoost on {len(X_train)} games...")
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=True)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    
    logger.info(f"\n{'='*60}")
    logger.info("FINAL MODEL PERFORMANCE")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"AUC:       {auc:.4f}")
    logger.info(f"Log Loss:  {logloss:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 15 Features:")
    for idx, row in importance.head(15).iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    # Save model
    model_path = "models/xgboost_36features_tuned.pkl"
    joblib.dump(model, model_path)
    logger.info(f"\n✅ Model saved to {model_path}")
    
    # Save feature importance
    importance.to_csv("output/feature_importance_36features.csv", index=False)
    logger.info("✅ Feature importance saved to output/feature_importance_36features.csv")

if __name__ == "__main__":
    main()
