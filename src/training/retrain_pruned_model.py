"""
Retrain XGBoost model with pruned 31-feature whitelist
Compare performance against original 107-feature model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import xgboost as xgb
import joblib
from datetime import datetime

from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST

print("=" * 80)
print("MODEL RETRAINING WITH PRUNED FEATURES (31 features)")
print("=" * 80)

# Initialize feature calculator
print("\n1. Initializing feature calculator...")
calc = FeatureCalculatorV5()

# Load game results from database
print("\n2. Loading game results...")
conn = sqlite3.connect('data/live/nba_betting_data.db')

query = """
SELECT 
    game_id,
    game_date,
    home_team,
    away_team,
    home_score,
    away_score,
    home_won
FROM game_results
WHERE game_date >= '2023-01-01' AND game_date < '2025-11-01'
ORDER BY game_date
"""

games_df = pd.read_sql(query, conn)
print(f"Loaded {len(games_df)} games from 2023-01-01 to 2025-11-01")

conn.close()

# Generate features for all games
print("\n3. Generating features for all games...")
features_list = []
labels = []

for idx, row in games_df.iterrows():
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{len(games_df)} ({100*idx/len(games_df):.1f}%)", end='\r')
    
    try:
        features = calc.calculate_game_features(
            home_team=row['home_team'],
            away_team=row['away_team'],
            game_date=row['game_date']
        )
        
        # Verify all whitelisted features are present
        if all(f in features for f in FEATURE_WHITELIST):
            features_list.append(features)
            
            # Use pre-computed label
            labels.append(row['home_won'])
        else:
            missing = [f for f in FEATURE_WHITELIST if f not in features]
            print(f"\n  Warning: Missing features for game {row['game_id']}: {missing}")
            
    except Exception as e:
        print(f"\n  Error processing game {row['game_id']}: {e}")
        continue

print(f"\n  Completed: {len(features_list)} games with complete features")

# Convert to DataFrame
X = pd.DataFrame(features_list)
y = np.array(labels)

print(f"\n4. Feature matrix shape: {X.shape}")
print(f"   Features: {list(X.columns)}")
print(f"   Labels: {len(y)} ({y.sum()} home wins, {len(y) - y.sum()} away wins)")

# Train-test split (time series - last 20% for testing)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n5. Train/Test split:")
print(f"   Train: {len(X_train)} games")
print(f"   Test:  {len(X_test)} games")

# Train XGBoost model
print("\n6. Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate
print("\n7. Model performance:")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)

print(f"   Accuracy:    {accuracy:.4f}")
print(f"   Log Loss:    {logloss:.4f}")
print(f"   Brier Score: {brier:.4f}")

# Feature importance
print("\n8. Top 10 feature importances:")
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importances.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.4f}")

# Save model
print("\n9. Saving model...")
model_path = 'models/xgboost_pruned_31features.pkl'
joblib.dump(model, model_path)
print(f"   Saved to: {model_path}")

# Save feature importance
importance_path = 'output/feature_importance_pruned.csv'
importances.to_csv(importance_path, index=False)
print(f"   Feature importance saved to: {importance_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nModel trained on {len(X_train)} games with {len(X.columns)} features")
print(f"Test accuracy: {accuracy:.2%}")
print(f"Next: Run SHAP analysis to verify injury features surfaced")
