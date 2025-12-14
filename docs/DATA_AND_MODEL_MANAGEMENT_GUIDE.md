# 🔧 NBA Betting System - Data & Model Management Guide

## 📁 Where Your Data Is Stored

### Primary Database
**Location**: `nba_betting_data.db` (SQLite database in project root)

**Contents**:
- Game results and schedules
- Team statistics (Four Factors, game logs)
- ELO ratings history
- Betting history and outcomes
- Live game tracking data

**Access**:
- Click "📁 Open Database Location" button in System Admin tab
- Use SQLite browser: `sqlite3 nba_betting_data.db`
- View with DB Browser for SQLite (free tool)

### Training Data CSV
**Location**: `data/master_training_data_v5.csv`

**Contents**:
- Engineered features for all historical games
- Rolling averages, momentum indicators
- Advanced statistics (eFG%, TOV%, etc.)
- ELO ratings at game time
- Target variables (outcomes)

**Size**: Typically 50-200 MB depending on seasons downloaded

**Access**:
- Click "📊 View Training Data" in System Admin tab
- Open directly in Excel or pandas
- Contains all features used for ML training

### Trained Models
**Location**: `models/`

**Files**:
- `model_v5_ats.xgb` - Against The Spread model
- `model_v5_ml.xgb` - Moneyline model  
- `model_v5_total.xgb` - Over/Under totals model

**Format**: XGBoost binary format (.xgb)

### Hyperparameter Configs
**Location**: `config/`

**Files**:
- `best_model_params_v5_classifier.json` - Best classification params
- `best_model_params_v5_regressor.json` - Best regression params

**Contents**: Optimal hyperparameters found through grid search

### Backtest Results
**Location**: `backtest_logs/`

**Files**:
- `backtest_summary_[timestamp].csv` - Performance metrics
- `backtest_details_[timestamp].json` - Detailed predictions
- `backtest_[timestamp].csv` - Individual bet results

## 🎯 How to Hypertune Your Models

### Method 1: Via Dashboard (Easiest)
1. Open Gold Standard Dashboard v4.1
2. Go to "⚙️ System Admin" tab
3. Click "3. Hyperparameter Tuning (Advanced)"
4. Wait 30-60 minutes for grid search to complete
5. Check terminal for best parameters
6. Results saved to `config/best_model_params_*.json`

### Method 2: Via Command Line
```bash
# Navigate to project directory
cd "C:\Users\d76do\OneDrive\Documents\New Basketball Model"

# Run hyperparameter tuning script
.\.venv\Scripts\python.exe live_model_backtester.py

# Results will be saved and displayed
```

### Method 3: Custom Grid Search
Edit `ml_model_trainer.py` line ~150-250 to customize parameter grid:

```python
def get_hyperparameter_grids(self):
    return {
        'xgboost': {
            'n_estimators': [100, 200, 300],      # More trees = better but slower
            'max_depth': [3, 5, 7],                # Deeper = more complex
            'learning_rate': [0.01, 0.05, 0.1],   # Lower = slower but better
            'subsample': [0.7, 0.8, 0.9],         # Row sampling
            'colsample_bytree': [0.7, 0.8, 0.9],  # Column sampling
        }
    }
```

### What Gets Tuned
- **n_estimators**: Number of boosting rounds (trees)
- **max_depth**: How deep each tree can go
- **learning_rate**: Step size for each tree
- **subsample**: Fraction of samples for each tree
- **colsample_bytree**: Fraction of features per tree
- **min_child_weight**: Minimum sum of weights in a leaf
- **gamma**: Minimum loss reduction for split

### Tuning Results
After tuning completes, check:
1. **Terminal output** - Shows best score and parameters
2. **config/best_model_params_*.json** - Saved best parameters (both timestamped and a canonical `best_live_wp_params_v2.json`)
3. **Auto-apply** - If `config/config.yaml` has `hypertuning.auto_apply: true`, the system will write `config/live_wp_runtime_params_v2.json` with the chosen parameters and the model will use them on next process start. Auto-apply also updates the running model instance when hypertuning is executed from the dashboard (if enabled).
3. **Models automatically retrained** with optimal settings

## 📊 Comparing Your Original Model vs New Model

### View Original Model Performance
Look for these files from your original system:
- `BACKTEST_FIX_REPORT.md`
- `backtest_history_report.csv`
- `COMPOSITE_FEATURES_RESULTS.md`

### View New Model Performance
1. Click "🔍 Model Performance Report" in System Admin
2. Check accuracy in Analytics tab (Historical section)
3. Review `backtest_logs/` for detailed results

### Key Metrics to Compare
- **Accuracy**: % of correct predictions (aim for >55%)
- **ROI**: Return on investment (aim for >3%)
- **Brier Score**: Probability calibration (lower is better)
- **Win Rate**: % of profitable bets
- **Max Drawdown**: Largest losing streak

### If Original Was Better
1. **Check hyperparameters**: Your old model may have had better tuning
2. **Compare features**: Look at original feature engineering
3. **Retune new model**: Use hyperparameter optimization
4. **Adjust Kelly fraction**: May need more conservative betting (0.02 → 0.01)

## 🔄 Full Model Refresh Workflow

### Complete Rebuild (Recommended Monthly)
```bash
# 1. Download fresh historical data
Click "1. Download Historical Data" in dashboard
# Saves to nba_betting_data.db

# 2. Run hyperparameter tuning
Click "3. Hyperparameter Tuning (Advanced)"
# Finds best parameters → config/

# 3. Train final models with best params
Click "2. Train ML Models (Backtest)"
# Saves models → models/

# 4. View performance
Click "🔍 Model Performance Report"
```

### Quick Update (Weekly)
```bash
# Just retrain with existing parameters
Click "2. Train ML Models (Backtest)"
```

## 📈 Accessing Your Data Programmatically

### Load Database
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('nba_betting_data.db')
games = pd.read_sql("SELECT * FROM games", conn)
conn.close()
```

### Load Training Data
```python
df = pd.read_csv('data/master_training_data_v5.csv')
print(f"Total samples: {len(df)}")
print(df.columns.tolist())  # All features
```

### Load Trained Model
```python
import xgboost as xgb

model = xgb.Booster()
model.load_model('models/model_v5_ml.xgb')
# Now use for predictions
```

### Load Best Parameters
```python
import json

with open('config/best_model_params_v5_classifier.json', 'r') as f:
    params = json.load(f)

print(f"Best score: {params['best_score']}")
print(f"Best params: {params['best_params']}")
```

## 🚀 Next Steps

1. **Download Data**: Click button in dashboard to get 3 seasons
2. **Hypertune**: Let it run overnight for best parameters  
3. **Compare**: Check if new model beats your original
4. **Adjust**: Tweak Kelly fraction in Risk Management
5. **Monitor**: Use Analytics tab to track live performance

---

**All data locations are accessible via the System Admin tab buttons for easy management!**