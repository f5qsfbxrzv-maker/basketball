# Calibration and Backtest Result Workflow

## Overview
This document describes how Brier scores, calibration metrics, and backtest results flow through the NBA betting system, from data collection to dashboard display.

## System Components

### 1. Data Collection
- **PBP Data**: Play-by-play events stored in `pbp_logs` table
- **Game Results**: Final outcomes in `game_results` table
- **ELO Ratings**: Tracked in `elo_history` table

### 2. Backtesting Pipeline
**File**: `core/live_model_backtester.py`

**Flow**:
1. Load historical PBP data and game results for a season
2. Replay each play chronologically through the live win probability model
3. Generate predictions at each game state (score, time, possession)
4. Compare predictions to actual final outcomes
5. Calculate Brier score: `mean((prediction - actual)^2)`
6. **Log results to database**:
   - Insert record into `backtest_history` table
   - Includes: timestamp, season, Brier score, quality rating, games count, PBP count

**Key Metrics**:
- **Brier Score < 0.15**: Excellent calibration
- **Brier Score < 0.20**: Good calibration
- **Brier Score < 0.25**: Fair calibration
- **Brier Score >= 0.25**: Poor calibration (needs tuning)

**Database Schema** (`backtest_history`):
```sql
CREATE TABLE backtest_history (
    timestamp TEXT,
    season TEXT,
    brier_score REAL,
    quality TEXT,
    games_count INTEGER,
    pbp_count INTEGER
)
```

### 3. Model Training Pipeline
**File**: `scripts/retrain_pipeline.py`

**Flow**:
1. Load training data from `data/training_data_with_features.csv`
2. Train ATS, Moneyline, and Total models with calibration
3. Calculate validation metrics:
   - **Classification models** (ATS, Moneyline): Brier score, log loss, ROC AUC
   - **Regression model** (Total): MAE, RMSE
4. **Log metrics to database**:
   - Insert records into `training_history` table
   - One row per model type (ATS, Moneyline, Total)
5. Save models to `models/production/` directory
6. Update `models/manifest.json` with metadata

**Database Schema** (`training_history`):
```sql
CREATE TABLE training_history (
    timestamp TEXT,
    model_type TEXT,
    brier_score REAL,
    log_loss REAL,
    roc_auc REAL,
    mae REAL,
    rmse REAL,
    roi REAL
)
```

### 4. Admin Dashboard Display
**File**: `admin_dashboard.py`

**Calibration Tab**:
- Queries latest record from `backtest_history`
- Displays: Brier score, games analyzed, last updated timestamp
- **Error Handling**: Shows warning popup if table is missing or empty

**Backtesting Tab**:
- Queries last 50 records from `backtest_history` (ordered by timestamp DESC)
- Displays table with: timestamp, season, Brier score, quality, games count, PBP count
- **Error Handling**: Shows warning popup if data cannot be loaded

**Model Health Tab**:
- Checks existence and age of model files in `models/`
- Displays status for each model (ATS, Moneyline, Total)
- **Error Handling**: Shows "Missing" status if model files don't exist

## Workflow Steps

### Running a Backtest
1. **Collect PBP data** (if not already present):
   ```powershell
   python -m core.nba_stats_collector_v2 --download-pbp --season 2023-24
   ```

2. **Run backtest**:
   ```python
   from core.live_model_backtester import LiveModelBacktester
   backtester = LiveModelBacktester()
   backtester.run_hyperparameter_tune(season="2023-24")
   ```

3. **Results automatically logged** to `backtest_history` table

4. **View in dashboard**:
   - Launch `admin_dashboard.py`
   - Navigate to Calibration or Backtesting tab
   - Results appear immediately

### Training Models
1. **Prepare training data**:
   ```powershell
   python scripts/prepare_training_data.py
   ```

2. **Train models**:
   ```powershell
   python scripts/retrain_pipeline.py
   ```

3. **Results automatically logged** to `training_history` table

4. **View in dashboard**:
   - Model Health tab shows last trained timestamp
   - Training metrics available for future dashboard expansion

### Hyperparameter Tuning
1. **Run grid search**:
   ```python
   backtester.run_hyperparameter_tune(season="2023-24")
   ```

2. **Results saved** to:
   - `config/best_live_wp_params_v2_{timestamp}.json`
   - `hypertune_history` table in database

3. **Review in dashboard**:
   - Backtesting tab → Hypertune History section
   - Select a row to preview parameters
   - Click "Apply Selected to Runtime" to deploy

## Database Tables Reference

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `backtest_history` | Backtest run results | `brier_score`, `quality`, `games_count` |
| `training_history` | Model training metrics | `model_type`, `brier_score`, `mae`, `rmse` |
| `elo_history` | Team ELO ratings over time | `team`, `offensive_elo`, `defensive_elo` |
| `predictions` | Pre-game predictions | `pred_spread`, `pred_total`, `edge` |
| `hypertune_history` | Hypertuning run audit trail | `best_brier`, `params`, `applied` |

## Error Handling & Fallbacks

### Dashboard Startup
- **Missing tables**: Warning popup with table names, dashboard continues with empty views
- **DB connection failure**: Error logged to console and log file
- **Schema mismatch**: Warning in console, creates missing tables automatically

### Data Loading
- **Empty tables**: Displays "No data" in UI, no crash
- **Query errors**: Warning popup + console log, graceful degradation
- **Invalid data**: Skipped with warning, continues processing valid records

## Calibration Quality Guidelines

### Brier Score Interpretation
- **< 0.15**: Excellent - Model is well-calibrated, predictions are reliable
- **0.15-0.20**: Good - Acceptable for production use, minor adjustments may help
- **0.20-0.25**: Fair - Consider hyperparameter tuning or feature engineering
- **> 0.25**: Poor - Major issues, check for data leakage or model bugs
- **> 0.30**: Critical - Worse than random guessing, requires immediate investigation

### When to Retrain
- Brier score degrades by >5% from baseline
- New season starts (different team dynamics)
- Major roster changes (trades, injuries)
- After collecting 250+ new settled predictions

### When to Retune Hyperparameters
- Brier score consistently above 0.25
- After major model architecture changes
- When testing new features or data sources
- Quarterly maintenance schedule

## Maintenance Tasks

### Daily
- Monitor Brier score trends in dashboard
- Check for settled predictions to update calibration

### Weekly
- Review backtest results for recent games
- Verify ELO ratings are updating correctly
- Check dashboard logs for errors

### Monthly
- Run full backtest on latest season data
- Review hyperparameter tuning results
- Update documentation if workflow changes

### Seasonal
- Retrain models with full season data
- Refresh feature engineering pipeline
- Archive old backtest results (>1 year)

## Troubleshooting

### "No data" in Calibration tab
**Cause**: `backtest_history` table is empty  
**Fix**: Run a backtest via admin dashboard or command line

### "Missing expected tables" warning
**Cause**: Fresh database or tables dropped  
**Fix**: Tables auto-created on next dashboard launch; run data prep to populate

### Brier score suspiciously low (< 0.10)
**Cause**: Possible data leakage (using future info in predictions)  
**Fix**: Audit feature calculation for look-ahead bias, verify chronological ordering

### Brier score very high (> 0.30)
**Cause**: Model issues, bad hyperparameters, or insufficient training data  
**Fix**: Check model health, retune hyperparameters, verify training data quality

## References
- **Brier Score**: [Wikipedia](https://en.wikipedia.org/wiki/Brier_score)
- **Calibration Theory**: Probabilities should match frequencies (70% predictions → 70% win rate)
- **NBA Stats API**: Used for PBP and box score data collection
- **Kelly Criterion**: Position sizing based on edge and calibrated probabilities

## Change Log
- 2025-11-21: Initial documentation created
- Added database logging for backtest and training metrics
- Implemented dashboard error handling and schema verification
