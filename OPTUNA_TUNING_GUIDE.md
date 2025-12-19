# Optuna Tuning - Quick Start Guide

## Overview
Training 25-feature XGBoost model with optimized injury_matchup_advantage metric using conservative hyperparameters and 3000 Optuna trials.

## Features (25 Total)
1. **ELO/Ratings (4)**: home_composite_elo, away_composite_elo, off_elo_diff, def_elo_diff
2. **EWMA Stats (6)**: ewma_efg_diff, ewma_pace_diff, ewma_tov_diff, ewma_orb_diff, ewma_vol_3p_diff, ewma_chaos_home
3. **Injuries (4)**: injury_impact_diff, injury_shock_diff, star_mismatch, **injury_matchup_advantage** ← NEW
4. **Advanced (11)**: net_fatigue_score, ewma_foul_synergy_home, total_foul_environment, league_offensive_context, season_progress, pace_efficiency_interaction, projected_possession_margin, three_point_matchup, net_free_throw_advantage, star_power_leverage, offense_vs_defense_matchup

## Dataset
- **File**: `data/training_data_matchup_with_injury_advantage.csv`
- **Games**: 12,205
- **Target**: target_moneyline_win (balance: 56.6%)
- **Date Range**: Historical NBA games with temporal ordering

## Hyperparameter Strategy
**Conservative Approach** (shallow trees, strong regularization):
```python
max_depth:          3-5     # Shallow trees
min_child_weight:   25-75   # Aggressive pruning
gamma:              2.0-10.0  # Strong split requirement
learning_rate:      0.001-0.02  # Very slow learning
n_estimators:       5000-12000  # Many weak learners
subsample:          0.5-0.7   # Bagging
colsample_bytree:   0.5-0.7   # Feature sampling
reg_alpha:          5.0-20.0  # L1 regularization (strong)
```

**Objective**: Minimize log loss with 5-fold time-series cross-validation

## Scripts Created

### 1. Pre-Flight Check
```bash
python scripts/preflight_check.py
```
**Purpose**: Validate data, features, dependencies before tuning  
**Runtime**: ~10 seconds  
**Output**: ✅/❌ checks for data, training, dependencies

### 2. Main Tuning Script
```bash
python scripts/optuna_tune_25features.py
```
**Purpose**: Run 3000-trial Optuna optimization  
**Runtime**: 8-25 hours (estimated)  
**Output**: 
- `models/xgboost_25features_optuna_YYYYMMDD_HHMMSS.json` (trained model)
- `models/optuna_study_YYYYMMDD_HHMMSS.pkl` (study results)
- `models/metadata_YYYYMMDD_HHMMSS.json` (hyperparams, metrics)
- `models/feature_importance_YYYYMMDD_HHMMSS.csv` (feature rankings)

### 3. Progress Monitor
```bash
python scripts/monitor_tuning.py
```
**Purpose**: Real-time monitoring of tuning progress (run in separate terminal)  
**Refresh**: Every 30 seconds  
**Shows**: Trial count, best log loss, speed, ETA, parameter updates

## Workflow

### Step 1: Pre-Flight Check ✅ DONE
```bash
python scripts/preflight_check.py
```
**Status**: ALL CHECKS PASSED

### Step 2: Start Tuning (Long-Running)
```bash
# Start tuning (8-25 hour runtime)
python scripts/optuna_tune_25features.py
```

### Step 3: Monitor Progress (Optional)
```bash
# In separate terminal
python scripts/monitor_tuning.py
```

### Step 4: Evaluate Results
After completion, check:
- Log loss improvement vs baseline
- Feature importance ranking (injury_matchup_advantage position)
- Convergence (last 100 trials vs best)
- Model file for deployment

## Expected Outcomes

### injury_matchup_advantage Performance
- **Pre-flight test**: Rank #9/25 (importance 0.0 on quick 100-tree test)
- **Expected after tuning**: Rank #10-15 (based on 0.0522 correlation)
- **Interpretation**: Complementary signal for ensemble, not primary driver

### Model Performance Targets
- **Log Loss**: < 0.650 (aim for improvement over baseline)
- **AUC**: > 0.575 (modest but meaningful)
- **Brier Score**: < 0.240

### Feature Importance (Expected Top 10)
1. ELO differentials (off_elo_diff, def_elo_diff, composite_elo)
2. EWMA stats (efg_diff, pace_diff, tov_diff)
3. Advanced features (star_power_leverage, offense_vs_defense_matchup)
4. Injury features (injury_matchup_advantage, injury_shock_diff)

## Monitoring During Run

**First 10 trials**: Verify speed (should be ~10-30s per trial)  
**Every 100 trials**: Check convergence (best log loss improving?)  
**Every 500 trials**: Review best hyperparameters (stable or oscillating?)

**Stop early if**:
- Convergence after 1500-2000 trials (last 100 trials within 0.001 of best)
- Speed too slow (>60s per trial = 50+ hours total)
- Technical errors

## Technical Details

### Time-Series Split
- **Folds**: 5
- **Method**: TimeSeriesSplit (respects temporal ordering)
- **Prevents**: Look-ahead bias in validation

### Early Stopping
- **Rounds**: 100
- **Metric**: Validation log loss
- **Purpose**: Prevent overfitting within each trial

### Optuna Settings
- **Sampler**: TPESampler (Tree-structured Parzen Estimator)
- **Pruner**: MedianPruner (n_warmup_steps=10)
- **Seed**: 42 (reproducibility)

## File Locations

```
New Basketball Model/
├── data/
│   └── training_data_matchup_with_injury_advantage.csv  # Training data
├── scripts/
│   ├── preflight_check.py          # Pre-flight validation
│   ├── optuna_tune_25features.py   # Main tuning script
│   └── monitor_tuning.py           # Progress monitor
└── models/
    ├── xgboost_25features_optuna_*.json      # Trained model (output)
    ├── optuna_study_*.pkl                    # Study results (output)
    ├── metadata_*.json                       # Hyperparams (output)
    └── feature_importance_*.csv              # Rankings (output)
```

## Troubleshooting

### Issue: Trials too slow (>60s each)
- **Solution**: Reduce n_estimators upper bound (e.g., 5000-8000 instead of 5000-12000)
- **Edit**: Line 43 in optuna_tune_25features.py

### Issue: Memory errors
- **Solution**: Reduce subsample and colsample_bytree ranges
- **Edit**: Lines 44-45 in optuna_tune_25features.py

### Issue: Not converging after 2000 trials
- **Action**: Let it continue, or stop and use best so far
- **Check**: Last 100 trials improvement vs overall best

### Issue: injury_matchup_advantage importance = 0.0
- **Note**: This happened in pre-flight (100 trees, shallow)
- **Expected**: Non-zero after 3000 trials with proper tuning
- **Acceptable**: Rank 10-20 (ensemble contribution, not primary)

## Post-Tuning Steps

1. **Evaluate model**:
   ```python
   import xgboost as xgb
   model = xgb.Booster()
   model.load_model('models/xgboost_25features_optuna_TIMESTAMP.json')
   ```

2. **Check feature importance**:
   ```python
   import pandas as pd
   df = pd.read_csv('models/feature_importance_TIMESTAMP.csv')
   print(df.head(10))
   print(df[df['feature'] == 'injury_matchup_advantage'])
   ```

3. **Compare to baseline**: Load previous model and compare log loss

4. **Deploy if improved**: Copy to production model path

## Notes

- **Conservative by design**: Prevents overfitting, better generalization
- **Injury feature goal**: Provide complementary signal, not replace ELO/EWMA
- **Expected improvement**: Modest (0.5-1% log loss reduction) but meaningful
- **Ensemble benefit**: injury_matchup_advantage captures edges individual features miss
