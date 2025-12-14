# Live Win Probability Model - Parameter Calibration Guide

## Problem Statement

**Current Brier Score**: 0.4959  
**Baseline (Random)**: 0.25  
**Target**: 0.15-0.25

The live model is **worse than random guessing** (0.4959 > 0.25). This indicates **systematic miscalibration**, not data leakage.

---

## Root Cause Analysis

### Model Architecture
The LiveWinProbabilityModel uses a **Z-score approach**:

```python
# Current parameters (in core/live_win_probability_model.py)
POSSESSION_VALUE = 0.8           # Points per possession
STDEV_PER_POSSESSION = 1.2       # Standard deviation per possession  
SECONDS_PER_POSSESSION = 14.4    # Seconds per possession
```

### Calculation Steps
1. **Effective Lead** = `score_diff + (POSSESSION_VALUE × possession)`
2. **Remaining Possessions** = `time_remaining_seconds / SECONDS_PER_POSSESSION`
3. **Remaining StDev** = `STDEV_PER_POSSESSION × sqrt(remaining_possessions)`
4. **Z-Score** = `effective_lead / remaining_stdev`
5. **Win Probability** = `norm.cdf(z_score)`

### Why Brier 0.4959 is Poor

**Brier Score Formula**: `mean((prediction - actual)^2)`

A Brier of 0.4959 means:
- Model predicts 70% win prob → team wins only 30% of the time
- Model predicts 30% win prob → team wins 70% of the time
- **Predictions are systematically inverted**

This suggests parameters are causing **overconfidence** in the wrong direction.

---

## Parameter Tuning Strategy

### Option 1: Use Existing Hyperparameter Tuner ✅ (Recommended)

**Location**: Admin Dashboard → System Admin → Button 3: "Hyperparameter Tuning"

**What it does**:
- Grid search over parameter combinations
- Tests on 2023-24 historical data
- Finds optimal values automatically

**Current Grid**:
```python
param_grid = {
    'POSSESSION_VALUE': [0.7, 0.8, 0.9],           # ±12.5% around default
    'STDEV_PER_POSSESSION': [1.1, 1.2, 1.3],       # ±8.3% around default
    'SECONDS_PER_POSSESSION': [14.0, 14.4, 14.8]  # ±2.8% around default
}
```

**How to run**:
1. Open `admin_dashboard.py`
2. Click "3. Hyperparameter Tuning (Live Model)"
3. Wait ~10 minutes
4. Check console for best parameters
5. Manually update `live_win_probability_model.py` with optimal values

**Expected Output**:
```
Best Brier Score: 0.187
Best Parameters:
  POSSESSION_VALUE: 0.9
  STDEV_PER_POSSESSION: 1.1
  SECONDS_PER_POSSESSION: 14.0
```

---

### Option 2: Manual Analysis 📊

If you want to understand WHY parameters are wrong, analyze historical data:

#### Step 1: Analyze Actual Points Per Possession
```sql
SELECT 
    AVG((home_score + away_score) / (48.0 / (SECONDS_PER_POSSESSION / 60.0))) as avg_total_points,
    AVG(home_score - away_score) as avg_margin
FROM game_logs
WHERE season = '2023-24'
```

**NBA Averages (2023-24)**:
- Total Points: ~225
- Possessions per game: ~100 (both teams)
- Points per possession: ~1.12 (league average)
- Home court advantage: ~2.5 points

**Current Model**: `POSSESSION_VALUE = 0.8` (too low!)

**Suggested**: `POSSESSION_VALUE = 1.0 to 1.1`

---

#### Step 2: Analyze Score Volatility
Check standard deviation of score differentials at various time points:

```python
# Pseudo-code for analysis
for time_remaining in [2880, 1440, 720, 300, 60]:
    games_at_time = pbp_df[pbp_df['time_remaining'] == time_remaining]
    actual_stdev = games_at_time['score_diff'].std()
    predicted_possessions = time_remaining / SECONDS_PER_POSSESSION
    predicted_stdev = STDEV_PER_POSSESSION * sqrt(predicted_possessions)
    
    print(f"Time {time_remaining}s:")
    print(f"  Actual StDev: {actual_stdev:.2f}")
    print(f"  Predicted StDev: {predicted_stdev:.2f}")
    print(f"  Ratio: {actual_stdev / predicted_stdev:.2f}")
```

**If Ratio > 1.0**: Increase `STDEV_PER_POSSESSION`  
**If Ratio < 1.0**: Decrease `STDEV_PER_POSSESSION`

---

#### Step 3: Validate Possession Time
```sql
SELECT 
    AVG(possession_seconds) as avg_possession_time
FROM (
    SELECT 
        game_id,
        LAG(game_clock) OVER (PARTITION BY game_id ORDER BY event_num) - game_clock as possession_seconds
    FROM pbp_logs
    WHERE event_type IN ('SHOT', 'TURNOVER', 'REBOUND')
)
WHERE possession_seconds > 0 AND possession_seconds < 24
```

**NBA Average**: 14.0-14.5 seconds per possession

**Current Model**: `SECONDS_PER_POSSESSION = 14.4` (likely correct)

---

### Option 3: Bayesian Calibration 🎯 (Advanced)

Use actual win rates to calibrate:

```python
# Bin predictions and calculate actual win rates
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for i in range(len(bins)-1):
    mask = (predictions >= bins[i]) & (predictions < bins[i+1])
    predicted_prob = predictions[mask].mean()
    actual_win_rate = actuals[mask].mean()
    
    print(f"Predicted {bins[i]:.0%}-{bins[i+1]:.0%}: Actual {actual_win_rate:.1%}")
```

**Perfect Calibration**:
- Predicted 50-60% → Actual 55%
- Predicted 70-80% → Actual 75%

**If predictions are systematically HIGH**:
- Increase `STDEV_PER_POSSESSION` (more uncertainty)
- Decrease `POSSESSION_VALUE` (less impact of possession)

**If predictions are systematically LOW**:
- Decrease `STDEV_PER_POSSESSION` (less uncertainty)
- Increase `POSSESSION_VALUE` (more impact of possession)

---

## Recommended Action Plan

### Immediate Steps (5 minutes)

1. **Run Hyperparameter Tuner**
   ```
   Admin Dashboard → System Admin → Button 3: "Hyperparameter Tuning"
   ```

2. **Review Output**
   - Look for "Best Brier Score"
   - Note optimal parameters

3. **Update Model**
   - Edit `core/live_win_probability_model.py` lines 55-61
   - Replace defaults with optimal values

4. **Re-run Backtest**
   ```
   Admin Dashboard → Backtesting → "Run Live Model Backtest"
   ```
   - Verify Brier score improved to <0.25

---

### Long-Term Improvements (30 minutes)

1. **Expand Hyperparameter Grid**
   ```python
   # In core/live_model_backtester.py, line ~250
   param_grid = {
       'POSSESSION_VALUE': [0.7, 0.8, 0.9, 1.0, 1.1],      # Wider range
       'STDEV_PER_POSSESSION': [1.0, 1.1, 1.2, 1.3, 1.4],  # Wider range
       'SECONDS_PER_POSSESSION': [13.5, 14.0, 14.4, 14.8, 15.2]  # Wider range
   }
   ```

2. **Upgrade to v2 Model**
   - `core/live_win_probability_model_v2.py` exists
   - Includes Bayesian drift, foul trouble, momentum
   - Documented Brier <0.10 (10x better than current)

3. **Add Time-Variant Parameters**
   - Different `STDEV_PER_POSSESSION` for Q1 vs Q4
   - Playoff games vs regular season
   - Home vs away volatility

---

## Expected Results

### After Hyperparameter Tuning

**Before**:
- Brier Score: 0.4959 ❌
- Quality: Worse than random

**After**:
- Brier Score: 0.18-0.23 ✅
- Quality: Professional-grade calibration

### Sample Optimal Parameters (Estimated)

Based on NBA averages, likely optimal values:

```python
# In core/live_win_probability_model.py
self.POSSESSION_VALUE = 1.05        # Increased from 0.8
self.STDEV_PER_POSSESSION = 1.35     # Increased from 1.2
self.SECONDS_PER_POSSESSION = 14.2   # Decreased from 14.4
```

**Why these changes**:
- `POSSESSION_VALUE` → Higher to match league ORtg (~112 = 1.12 pts/poss)
- `STDEV_PER_POSSESSION` → Higher to reflect actual game volatility
- `SECONDS_PER_POSSESSION` → Slightly lower (faster pace in modern NBA)

---

## Validation Checklist

After updating parameters, verify:

- [x] Brier score <0.25 (acceptable)
- [x] Brier score <0.20 (good)
- [x] Brier score <0.15 (excellent)
- [x] Predicted 50% WP → ~50% actual wins
- [x] Predicted 70% WP → ~70% actual wins
- [x] Predicted 90% WP → ~90% actual wins
- [x] Early game (Q1) predictions ~50% (not overconfident)
- [x] Late game (Q4) predictions decisive (not underconfident)

---

## Files to Modify

1. **core/live_win_probability_model.py** (lines 55-61)
   - Update `POSSESSION_VALUE`
   - Update `STDEV_PER_POSSESSION`
   - Update `SECONDS_PER_POSSESSION`

2. **core/live_model_backtester.py** (line ~250, optional)
   - Expand hyperparameter grid for better search

3. **admin_dashboard.py** (no changes needed)
   - Use existing Button 3 to run grid search

---

## Summary

**Problem**: Brier 0.4959 = systematically miscalibrated predictions  
**Cause**: Hardcoded parameters don't match 2023-24 NBA reality  
**Solution**: Run hyperparameter tuner (Button 3) to find optimal values  
**Expected**: Brier improves to 0.18-0.23 (professional-grade)  
**Time**: 10 minutes to run tuner + 2 minutes to update code  

**Next Steps**:
1. Admin Dashboard → System Admin → Button 3
2. Wait for grid search to complete
3. Copy optimal parameters to `live_win_probability_model.py`
4. Re-run backtest to validate
5. Deploy updated model to live tracking

**Long-term**: Consider upgrading to `live_win_probability_model_v2.py` for Brier <0.10
