"""
Train and save MDP Production Regressor Model
Uses params from production_config_mdp.py and trains on full dataset
"""

import pandas as pd
import xgboost as xgb
from production_config_mdp import (
    ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS, DATA_PATH
)

print("="*80)
print("🚀 TRAINING MDP PRODUCTION REGRESSOR")
print("="*80)

# Load training data
print(f"\n📥 Loading data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   ✓ Loaded {len(df):,} games")

# Prepare features and target
X = df[ACTIVE_FEATURES]
y = df['margin_target']

print(f"\n🔧 Features: {len(ACTIVE_FEATURES)}")
for feat in ACTIVE_FEATURES:
    print(f"   - {feat}")

print(f"\n⚙️  XGBoost Params:")
for k, v in XGB_PARAMS.items():
    print(f"   - {k}: {v}")

# Train model
print(f"\n🏋️  Training regressor ({N_ESTIMATORS} boosting rounds)...")
dtrain = xgb.DMatrix(X, label=y, feature_names=ACTIVE_FEATURES)
model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=100)

# Evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_pred = model.predict(dtrain)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

print(f"\n📊 Training Metrics:")
print(f"   RMSE: {rmse:.2f} points")
print(f"   MAE:  {mae:.2f} points")

# Save model
output_path = 'models/nba_mdp_production_tuned.json'
model.save_model(output_path)
print(f"\n💾 Model saved: {output_path}")

print("\n✅ COMPLETE - MDP Regressor ready for production")
print("="*80)
