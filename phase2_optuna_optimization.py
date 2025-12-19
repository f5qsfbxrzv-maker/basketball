"""
Phase 2: Optuna Hyperparameter Optimization with Gold Standard ELO
Optimizes XGBoost/LightGBM hyperparameters now that ELO features use new scale.
Uses TimeSeriesSplit to prevent look-ahead bias.
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
import xgboost as xgb
import optuna
from datetime import datetime
import json

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\data\live\nba_betting_data.db"

print("\n" + "=" * 80)
print("PHASE 2: OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=" * 80)
print("\nObjective: Find optimal XGBoost hyperparameters for Gold Standard ELO features")
print("Metric: Log-Loss (rewards confident + correct predictions)\n")

# Load training data with NEW ELO features
print("Loading training data with Gold Standard ELO features...")

conn = sqlite3.connect(DB_PATH)

# First, get all games from game_logs
query_games = """
    SELECT 
        g1.GAME_DATE as game_date,
        g1.SEASON_ID as season,
        g1.TEAM_ABBREVIATION as home_team,
        g2.TEAM_ABBREVIATION as away_team,
        g1.PTS as home_score,
        g2.PTS as away_score,
        CASE WHEN g1.WL = 'W' THEN 1 ELSE 0 END as home_won
    FROM game_logs g1
    JOIN game_logs g2 ON g1.GAME_ID = g2.GAME_ID AND g1.TEAM_ABBREVIATION != g2.TEAM_ABBREVIATION
    WHERE g1.MATCHUP LIKE '%vs%'
    AND g1.SEASON_ID >= '2015-16'
    ORDER BY g1.GAME_DATE
"""

games_df = pd.read_sql_query(query_games, conn)

# Convert season format from "22015" to "2015-16"
def convert_season_format(season_id):
    """Convert from NBA API format (22015) to our format (2015-16)."""
    year = int(season_id) - 20000  # 22015 -> 2015
    return f"{year}-{str(year+1)[2:]}"  # 2015 -> "2015-16"

games_df['season'] = games_df['season'].apply(convert_season_format)

# Get all ELO ratings
elo_df = pd.read_sql_query("""
    SELECT team, season, game_date, composite_elo, off_elo, def_elo
    FROM elo_ratings
    ORDER BY game_date
""", conn)

conn.close()

print(f"   Loaded {len(games_df)} games from game_logs")
print(f"   Loaded {len(elo_df)} ELO ratings")

# Merge ELO ratings with games
print("   Merging ELO features with games...")

# Convert elo_df to dict for faster lookup
elo_dict = {}
for _, row in elo_df.iterrows():
    key = (row['team'], row['season'], row['game_date'])
    elo_dict[key] = {
        'composite_elo': row['composite_elo'],
        'off_elo': row['off_elo'],
        'def_elo': row['def_elo']
    }

def get_latest_elo(team, season, before_date):
    """Get the most recent ELO for a team before a given date."""
    # Find all dates for this team/season before the game date
    dates = [date for (t, s, date) in elo_dict.keys() if t == team and s == season and date < before_date]
    if dates:
        latest_date = max(dates)
        return elo_dict[(team, season, latest_date)]
    return None

# Add ELO features to games (now much faster with dict lookup)

# Add ELO features to games
home_elos = []
away_elos = []

for idx, row in games_df.iterrows():
    if idx % 2000 == 0 and idx > 0:
        print(f"      Progress: {idx}/{len(games_df)} games...")
    
    home_elo = get_latest_elo(row['home_team'], row['season'], row['game_date'])
    away_elo = get_latest_elo(row['away_team'], row['season'], row['game_date'])
    
    home_elos.append(home_elo)
    away_elos.append(away_elo)

# Add ELO columns
games_df['home_composite_elo'] = [e['composite_elo'] if e else None for e in home_elos]
games_df['home_off_elo'] = [e['off_elo'] if e else None for e in home_elos]
games_df['home_def_elo'] = [e['def_elo'] if e else None for e in home_elos]
games_df['away_composite_elo'] = [e['composite_elo'] if e else None for e in away_elos]
games_df['away_off_elo'] = [e['off_elo'] if e else None for e in away_elos]
games_df['away_def_elo'] = [e['def_elo'] if e else None for e in away_elos]

# Drop games without ELO data (first few games of each season)
df = games_df.dropna(subset=['home_composite_elo', 'away_composite_elo']).copy()

print(f"   Loaded {len(df)} games with complete ELO features")

print(f"   Loaded {len(df)} games with ELO features")
print(f"   Date range: {df['game_date'].min()} to {df['game_date'].max()}")

# Feature engineering
df['elo_diff'] = df['home_composite_elo'] - df['away_composite_elo']
df['off_elo_diff'] = df['home_off_elo'] - df['away_off_elo']
df['def_elo_diff'] = df['home_def_elo'] - df['away_def_elo']
df['total_score'] = df['home_score'] + df['away_score']

# Features and target
feature_cols = [
    'home_composite_elo', 'away_composite_elo', 'elo_diff',
    'home_off_elo', 'home_def_elo', 'away_off_elo', 'away_def_elo',
    'off_elo_diff', 'def_elo_diff'
]

X = df[feature_cols].values
y = df['home_won'].values
dates = pd.to_datetime(df['game_date'])

print(f"\nFeatures: {len(feature_cols)} ELO-based features")
print(f"Target: Home team win (1) or loss (0)")
print(f"Positive class rate: {y.mean():.3f}")

# Define Optuna objective
def objective(trial):
    """Optuna objective function for XGBoost hyperparameter tuning."""
    
    # Suggest hyperparameters with wider search ranges for deep optimization
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'random_state': 42
    }
    
    # Time series cross-validation (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    log_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate log-loss
        ll = log_loss(y_val, y_pred_proba)
        log_losses.append(ll)
        
        # Report intermediate value for pruning
        trial.report(ll, fold)
        
        # Handle pruning based on intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return average log-loss across folds
    return np.mean(log_losses)

# Run Optuna optimization
print("\nStarting deep Optuna optimization (1000 trials - 8 hour overnight run)...")
print("Using MedianPruner to skip obviously poor trials early")
print("Estimated completion: ~8 hours\n")

# Create study with pruning for efficiency
study = optuna.create_study(
    direction='minimize', 
    study_name='xgboost_gold_elo_deep',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
)
study.optimize(objective, n_trials=1000, show_progress_bar=True)

print("\n✅ Optimization complete!")

# Best parameters
best_params = study.best_params
best_log_loss = study.best_value

print("\n" + "=" * 80)
print("OPTIMAL HYPERPARAMETERS")
print("=" * 80)
for param, value in best_params.items():
    print(f"   {param}: {value}")

print(f"\n📊 Best Log-Loss (CV): {best_log_loss:.4f}")
print(f"   Completed trials: {len(study.trials)}")
print(f"   Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"   Optimization time: {(datetime.now() - datetime.fromisoformat(study.trials[0].datetime_start.isoformat())).total_seconds() / 3600:.2f} hours")

# Train final model with best parameters
print("\n🎯 Training final model with optimal hyperparameters...")

final_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    **best_params
}

final_model = xgb.XGBClassifier(**final_params)
final_model.fit(X, y, verbose=False)

# Evaluate on full dataset (for reference)
y_pred_proba = final_model.predict_proba(X)[:, 1]
y_pred = final_model.predict(X)

train_log_loss = log_loss(y, y_pred_proba)
train_accuracy = accuracy_score(y, y_pred)
train_brier = brier_score_loss(y, y_pred_proba)

print("\n📈 Final Model Performance (Full Dataset):")
print(f"   Log-Loss: {train_log_loss:.4f}")
print(f"   Accuracy: {train_accuracy:.3f}")
print(f"   Brier Score: {train_brier:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Feature Importance:")
for _, row in feature_importance.iterrows():
    print(f"   {row['feature']:<25} {row['importance']:.4f}")

# Save model
model_path = "models/xgboost_gold_elo_optimized.json"
final_model.save_model(model_path)
print(f"\n💾 Model saved to: {model_path}")

# Save hyperparameters
config_path = "models/gold_elo_hyperparameters.json"
with open(config_path, 'w') as f:
    json.dump({
        'hyperparameters': final_params,
        'cv_log_loss': best_log_loss,
        'train_log_loss': train_log_loss,
        'train_accuracy': train_accuracy,
        'feature_importance': feature_importance.to_dict('records'),
        'optimization_date': datetime.now().isoformat(),
        'n_trials': 1000,
        'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'optimization_time_hours': (datetime.now() - datetime.fromisoformat(study.trials[0].datetime_start.isoformat())).total_seconds() / 3600,
        'elo_parameters': {
            'k_factor': 15.0,
            'elo_scale': 40.0,
            'win_weight': 30.0,
            'mov_bias': 0.5
        }
    }, f, indent=2)
print(f"💾 Hyperparameters saved to: {config_path}")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE: MODEL OPTIMIZED FOR GOLD STANDARD ELO")
print("=" * 80)
print("\n📋 Next Steps:")
print("   1. ✅ Historical ELO re-synced with Gold Standard parameters")
print("   2. ✅ XGBoost hyperparameters optimized with Optuna")
print("   3. ⏭️  Test dashboard predictions for 12/18/2025 (DET vs DAL)")
print("   4. ⏭️  Set up weekly auto-calibration script")
print("\nThe model now properly understands the new ELO scale!")
