# NBA Dashboard Backtesting Guide

## How to Run Backtests from the Dashboard

### Prerequisites
1. **Download Historical Data First**
   - Open Dashboard → Admin tab
   - Click "1️⃣ Download Historical Data (NBA Stats)"
   - Wait for download to complete (includes PBP data)
   - Console should show: "Downloaded X seasons"

### Running a Backtest

#### Option 1: Quick Backtest (Validate Current Model)
1. Go to **Admin** tab
2. Click **"3️⃣ Train ML Models (with Backtest)"**
3. Wait 2-3 minutes for completion
4. Results shown in console:
   ```
   [BACKTEST] Loaded 50,000+ PBP events, 1,230 games
   [BACKTEST] Running backtest...
   [BACKTEST] ✅ Complete! Brier Score: 0.0847
   [BACKTEST] Model Quality: EXCELLENT (Professional Grade)
   ```

**What It Tests:**
- Live Win Probability Model accuracy
- Uses 2023-24 season PBP data
- Calculates Brier Score (lower is better)
- Target: <0.10 for professional grade

**Interpreting Brier Scores:**
- `< 0.10` = EXCELLENT (Professional Grade)
- `0.10-0.15` = GOOD
- `0.15-0.20` = FAIR
- `> 0.20` = NEEDS IMPROVEMENT

#### Option 2: Hyperparameter Optimization (Find Best Settings)
1. Go to **Admin** tab
2. Click **"4️⃣ Hyperparameter Tuning (Live Model)"**
3. ⚠️ **This takes 10-20 minutes** (grid search of 27 combinations)
4. Progress shown in console:
   ```
   [HYPERTUNE] Starting hyperparameter optimization...
   [HYPERTUNE] Grid search testing 27 parameter combinations...
   [HYPERTUNE] ⚠️ This will take 10-20 minutes - be patient!
   
   Testing Params: Pos=0.7, Stdev=1.1, Sec=14.0
      -> Brier Score: 0.092341 (Duration: 45.3s)
   Testing Params: Pos=0.7, Stdev=1.1, Sec=14.4
      -> Brier Score: 0.091287 (Duration: 44.8s)
   ...
   
   --- HYPERTUNING COMPLETE ---
   Best Brier Score: 0.084521
   Best Parameters:
   {'POSSESSION_VALUE': 0.8, 'STDEV_PER_POSSESSION': 1.2, 'SECONDS_PER_POSSESSION': 14.4}
   ```

5. **Update Model with Best Params:**
   - Open `live_win_probability_model.py`
   - Update the constants at the top:
     ```python
     POSSESSION_VALUE = 0.8  # From hypertuning results
     STDEV_PER_POSSESSION = 1.2
     SECONDS_PER_POSSESSION = 14.4
     ```
      Optional: If `config/config.yaml` sets `hypertuning.auto_apply=true`, the dashboard will auto-apply the best parameters to the running Live WP model and write `config/live_wp_runtime_params_v2.json`. For production safety, we recommend running ROI simulations before enabling auto-apply — see `hypertuning.apply_by` in config for options ('brier' or 'roi'). ROI requires historical odds and is experimental.
   - Save and restart dashboard

### Backtest Data Requirements

**Required Database Tables:**
- `pbp_logs` - Play-by-play event data
  - Columns: game_id, event_num, period, clock, home_score, away_score, score_margin, event_type
- `game_results` - Final game outcomes
  - Columns: game_id, season, home_won
- `games` - Game metadata
  - Columns: game_id, season, game_date, home_team_name, away_team_name

**Sample Size Recommendations:**
- Minimum: 500 games (1 season)
- Recommended: 1,000+ games (2 seasons)
- Optimal: 2,000+ games (3+ seasons)

### Behind the Scenes

**What the Backtest Does:**
1. Loads all PBP events from selected season
2. Loads final game results (who actually won)
3. Replays each game event-by-event
4. At each event, model predicts home win probability
5. Compares predictions to actual outcomes
6. Calculates Brier Score: `mean((prediction - actual)²)`

**Example:**
```
Game: Lakers vs Celtics
Event 1: 12:00 Q1, 0-0 → Model: 52% Lakers win, Actual: Lakers won (1) → Error²: (0.52-1)² = 0.2304
Event 2: 11:30 Q1, 2-0 → Model: 54% Lakers win, Actual: Lakers won (1) → Error²: (0.54-1)² = 0.2116
...
Event N: 0:00 Q4, 108-105 → Model: 98% Lakers win, Actual: Lakers won (1) → Error²: (0.98-1)² = 0.0004

Brier Score = mean of all Error² values
```

### Troubleshooting

**Error: "No backtest data available"**
- Solution: Run "Download Historical Data" first
- Check console for download errors
- Verify `pbp_logs` table has data: `SELECT COUNT(*) FROM pbp_logs;`

**Error: "Backtester not available"**
- Solution: Check `live_model_backtester.py` exists
- Verify imports at top of dashboard file
- Restart dashboard

**Backtest takes too long (>5 minutes)**
- Normal for 50,000+ PBP events
- Consider reducing data: `WHERE game_date >= '2024-01-01'`
- Check CPU usage (should be 80-100%)

**Brier Score > 0.20 (Poor Performance)**
- Run hyperparameter tuning to find better params
- Check if PBP data is clean (no missing scores)
- Verify game results are accurate

### Advanced: Custom Backtest Seasons

To test on specific seasons, edit `_task_train_models()`:

```python
# Change this line:
pbp_df, results_df = self.backtester._load_backtest_data("2023-24")

# To test multiple seasons:
pbp_df, results_df = self.backtester._load_backtest_data("2022-23")
# Or modify the method to accept date ranges
```

### Performance Benchmarks

**Typical Backtest Times (on modern CPU):**
- 500 games: ~1 minute
- 1,000 games: ~2-3 minutes
- 1,230 games (full season): ~3-4 minutes

**Typical Hypertuning Times:**
- 27 combinations × 3 minutes each = ~81 minutes (worst case)
- Actual: ~15-25 minutes (varies by CPU)

### Next Steps After Backtesting

1. **If Brier < 0.10:** Model is production-ready
   - Enable live betting features
   - Monitor performance on upcoming games
   - Track ROI in Performance tab

2. **If Brier 0.10-0.15:** Model is acceptable
   - Run hyperparameter tuning
   - Consider adding more features
   - Test on different seasons

3. **If Brier > 0.15:** Model needs improvement
   - Run hyperparameter tuning FIRST
   - Check data quality
   - Review model assumptions in `live_win_probability_model.py`

---

**Last Updated:** 2024-11-19  
**Compatible With:** NBA Dashboard Enhanced v5.0
