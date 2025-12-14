# 🎯 Quick Reference: Dashboard Fixes & New Features

## ✅ Issues Fixed

### 1. Date Filtering Now Works Correctly
**Problem**: Games weren't showing for selected dates
**Fix**: 
- Now generates games dynamically for next 7 days
- Date picker correctly filters to exact date selected
- "Show All Upcoming" button shows next 3 days
- Proper date comparison using datetime objects

**Test It**:
1. Open dashboard
2. Change date picker to today/tomorrow
3. Should see 2 games per day
4. Click "Show All Upcoming" to see next 3 days

---

## 📁 Data Storage Locations

### Quick Access (New Buttons in System Admin Tab)
- **📁 Open Database Location** - Opens folder with all data files
- **📊 View Training Data** - Shows statistics about your training dataset
- **🔍 Model Performance Report** - Displays model metrics and files

### File Locations
```
Project Root/
├── nba_betting_data.db           # SQLite database (game data, stats)
├── data/
│   └── master_training_data_v5.csv  # Training features CSV
├── models/
│   ├── model_v5_ats.xgb           # Against spread model
│   ├── model_v5_ml.xgb            # Moneyline model
│   └── model_v5_total.xgb         # Totals model
├── config/
│   ├── best_model_params_v5_classifier.json
│   └── best_model_params_v5_regressor.json
└── backtest_logs/
    └── backtest_summary_*.csv      # Performance results
```

---

## 🎯 Hyperparameter Tuning (New Feature!)

### Via Dashboard
1. Go to **⚙️ System Admin** tab
2. Click **"3. Hyperparameter Tuning (Advanced)"**
3. Wait 30-60 minutes for grid search
4. Results auto-saved to `config/best_model_params_*.json`
5. Models automatically use best parameters

### What Gets Optimized
- Number of trees (n_estimators)
- Tree depth (max_depth)
- Learning rate (eta)
- Sampling ratios (subsample, colsample)
- Regularization (gamma, lambda, alpha)

### View Results
- Check terminal output for best score
- Click "🔍 Model Performance Report" 
- Compare before/after accuracy in Analytics tab

---

## 📊 Comparing Models

### Your Original Model Stats
Check these files in your project:
- `COMPOSITE_FEATURES_RESULTS.md`
- `backtest_history_report.csv`
- Look for accuracy, ROI, win rate

### New Model Stats
After training, check:
- **Analytics Tab** → Historical section shows accuracy
- **System Admin** → Click "🔍 Model Performance Report"
- `backtest_logs/` folder for detailed results

### Key Metrics
| Metric | Good | Great | Elite |
|--------|------|-------|-------|
| Accuracy | >53% | >55% | >57% |
| ROI | >2% | >5% | >8% |
| Win Rate | >50% | >54% | >58% |
| Brier Score | <0.25 | <0.22 | <0.20 |

---

## 🚀 Complete Workflow

### First-Time Setup
1. **Download Data**: System Admin → "1. Download Historical Data"
   - Takes ~10 minutes
   - Saves to `nba_betting_data.db`
   - Exports to `data/master_training_data_v5.csv`

2. **Hypertune Models**: System Admin → "3. Hyperparameter Tuning"
   - Takes 30-60 minutes
   - Finds optimal parameters
   - Saves to `config/` folder
   - If you enable `hypertuning.auto_apply: true` in `config/config.yaml`, the backtester will write `config/live_wp_runtime_params_v2.json` and the running Live WP model will pick those parameters up immediately (and new processes will also use them). Use with caution; recommended to first inspect results.

3. **Train Models**: System Admin → "2. Train ML Models"
   - Takes ~5 minutes
   - Uses best parameters from step 2
   - Saves to `models/` folder

4. **Compare**: Click "🔍 Model Performance Report"
   - Compare vs your original model
   - Adjust Risk Management if needed

### Weekly Refresh
1. System Admin → "1. Download Historical Data" (get latest games)
2. System Admin → "2. Train ML Models" (retrain with new data)
3. Done! Models stay fresh

### Monthly Deep Tune
1. Run full workflow above (Download → Hypertune → Train)
2. Ensures parameters stay optimal as season progresses

---

## 💡 Tips for Better Performance

### If New Model Underperforms Original
1. **Check hyperparameters**: Original might have had different tuning
2. **Run hyperparameter tuning**: Click the button, let it optimize
3. **Adjust Kelly fraction**: Try 0.01 instead of 0.02 (more conservative)
4. **Compare features**: Check what features original model used

### Optimize Risk Management
In System Admin tab, adjust:
- **Bankroll**: Your total capital
- **Kelly Fraction**: 0.25 = aggressive, 0.10 = moderate, 0.05 = conservative
- **Min Bet Size**: Filter out tiny edges (recommend 1-2%)

### Monitor Performance
Analytics tab shows:
- **Left side**: Your live betting results
- **Right side**: Historical backtest accuracy
- Both should trend upward over time

---

## 🆘 Troubleshooting

### "No games found for selected date"
- Normal if no NBA games that day
- Try clicking "Show All Upcoming" 
- Check if NBA season is active

### "Download failed"
- Check internet connection
- NBA stats API might be down
- Try again in a few minutes

### "Training failed"
- Ensure data downloaded first
- Check `nba_betting_data.db` exists
- Run "📊 View Training Data" to verify

### Models performing poorly
- Run hyperparameter tuning
- Increase training data (download more seasons)
- Adjust Kelly fraction (be more conservative)

---

**All features now accessible via the System Admin tab! No command line needed for data access or hypertuning.**