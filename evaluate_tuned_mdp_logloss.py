import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
import joblib

# CONFIG
DATA_PATH = 'data/training_data_MDP_with_margins.csv'
PARAMS_PATH = 'best_params_margin.joblib'
NBA_STD_DEV = 13.5

# FEATURES
ACTIVE_FEATURES = [
    'off_elo_diff', 'def_elo_diff', 'home_composite_elo',           
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',          
    'injury_matchup_advantage', 'injury_shock_diff', 'star_power_leverage',
    'season_progress', 'league_offensive_context',     
    'total_foul_environment', 'net_free_throw_advantage',             
    'offense_vs_defense_matchup', 'pace_efficiency_interaction', 'star_mismatch'
]

print("="*60)
print("🔍 EVALUATING TUNED MDP REGRESSOR LOG LOSS")
print("="*60)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Total games: {len(df):,}")

# Load best params
best_params = joblib.load(PARAMS_PATH)
print(f"\n📊 Best Hyperparameters:")
for key, value in best_params.items():
    print(f"   {key}: {value}")

# Prepare data
X = df[ACTIVE_FEATURES]
y_margin = df['margin_target']
y_binary = (y_margin > 0).astype(int)  # Home team wins

# Split by date for temporal validation
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
split_date = '2024-10-01'
train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train_margin, y_test_margin = y_margin[train_mask], y_margin[test_mask]
y_train_binary, y_test_binary = y_binary[train_mask], y_binary[test_mask]

print(f"\nTrain: {len(X_train):,} games (before {split_date})")
print(f"Test:  {len(X_test):,} games (2024-25 season)")

# Build optimal params for XGBoost
xgb_params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_jobs': -1,
    'random_state': 42,
    **best_params
}

# Train regressor
print("\n🚀 Training MDP Regressor with optimal params...")
dtrain = xgb.DMatrix(X_train, label=y_train_margin)
dtest = xgb.DMatrix(X_test, label=y_test_margin)

model = xgb.train(
    xgb_params, 
    dtrain, 
    num_boost_round=xgb_params['n_estimators'],
    verbose_eval=False
)

# Predict margins
pred_margins = model.predict(dtest)

# Convert margins to probabilities
pred_probs = norm.cdf(pred_margins / NBA_STD_DEV)

# Calculate metrics
test_log_loss = log_loss(y_test_binary, pred_probs)
test_brier = brier_score_loss(y_test_binary, pred_probs)
test_mae = np.mean(np.abs(pred_margins - y_test_margin))
test_rmse = np.sqrt(np.mean((pred_margins - y_test_margin)**2))

print("\n" + "="*60)
print("📈 TUNED MDP REGRESSION RESULTS (2024-25 TEST SET)")
print("="*60)
print(f"RMSE (Margin Error):        {test_rmse:.4f} points")
print(f"MAE (Margin Error):         {test_mae:.4f} points")
print(f"\nAfter margin → probability conversion:")
print(f"Log Loss:                   {test_log_loss:.5f}")
print(f"Brier Score:                {test_brier:.5f}")
print(f"\nPredicted Margin Stats:")
print(f"  Mean:  {pred_margins.mean():+.2f} points")
print(f"  Std:   {pred_margins.std():.2f} points")
print(f"  Range: {pred_margins.min():+.2f} to {pred_margins.max():+.2f}")
print(f"\nPredicted Probability Stats:")
print(f"  Mean:  {pred_probs.mean():.3f}")
print(f"  Std:   {pred_probs.std():.3f}")
print(f"  Range: {pred_probs.min():.3f} to {pred_probs.max():.3f}")

# Compare to baseline classifier performance
print("\n" + "="*60)
print("📊 COMPARISON TO PREVIOUS RESULTS")
print("="*60)
print("Binary Classifier (from previous test):")
print("  Log Loss: 0.72122")
print(f"\nTuned MDP Regressor:")
print(f"  Log Loss: {test_log_loss:.5f}")
print(f"\nImprovement: {((0.72122 - test_log_loss) / 0.72122 * 100):+.2f}%")

# Calibration analysis
print("\n" + "="*60)
print("🎯 CALIBRATION ANALYSIS")
print("="*60)
prob_bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
for low, high in prob_bins:
    mask = (pred_probs >= low) & (pred_probs < high)
    if mask.sum() > 0:
        actual_rate = y_test_binary[mask].mean()
        predicted_avg = pred_probs[mask].mean()
        count = mask.sum()
        error = actual_rate - predicted_avg
        print(f"{low:.0%}-{high:.0%}: Pred {predicted_avg:.1%}, Actual {actual_rate:.1%}, "
              f"Error {error:+.1%}, N={count}")

print("\n✅ Evaluation complete!")
