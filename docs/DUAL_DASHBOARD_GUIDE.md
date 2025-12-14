# NBA Betting System - Dual Dashboard Architecture

## 🎯 Overview

The NBA Betting System now uses a **dual-dashboard architecture** for optimal workflow separation:

1. **Main Betting Dashboard** (`NBA_Dashboard_v6_Streamlined.py`) - Clean, betting-focused interface
2. **Admin Dashboard** (`admin_dashboard.py`) - Comprehensive system administration, diagnostics, and model management

---

## 🚀 Quick Start

### Launch Both Dashboards
```bash
python launch_dashboards.py
```

This automatically opens both dashboards with proper positioning.

### Launch Individual Dashboards
```bash
# Main betting dashboard only
python NBA_Dashboard_v6_Streamlined.py

# Admin dashboard only
python admin_dashboard.py
```

---

## 📊 Main Betting Dashboard

**Purpose**: Day-to-day betting operations

### Tabs:
1. **Synopsis** - Today's top picks with highest edge
2. **Today's Games** - All predictions for today's slate
3. **Live Tracker** - Real-time game monitoring (30s refresh)
4. **Bet History** - Track placed bets, results, ROI
5. **Bankroll** - Manage bankroll and Kelly settings

### Features:
- Clean, distraction-free betting interface
- Professional sportsbook color scheme (deep blacks, bright whites)
- Full team names (not abbreviations)
- Edge display as units (+5.2u format)
- One-click access to admin dashboard
- Real-time data refresh

---

## ⚙️ Admin & Diagnostics Dashboard

**Purpose**: System administration, model training, diagnostics, and performance monitoring

### Tabs:

#### 1. **System Admin**
- **Data Pipeline (6 Steps)**:
  1. Download Historical Data (NBA Stats)
  2. Scrape Current Injuries
  3. Train ML Models + Backtest
  4. Hyperparameter Tuning (Live Model)
  5. Generate Today's Predictions
  6. Export Reports & Data
- **Data Management**: Database location, training data viewer, performance reports
- **Console Output**: Real-time logging with color-coded messages

#### 2. **Backtesting**
- Run full backtests on historical data (2023-24 season)
- View Brier score, quality assessment, games analyzed
- Backtest history table with timestamps
- Safe background execution (no UI freeze)

#### 3. **Calibration**
- Monitor calibration quality (Brier score tracking)
- Reliability curve analysis
- Calibration metrics: games analyzed, last updated timestamp
- Ensures predicted probabilities match actual outcomes

#### 4. **Model Health**
- Check if model files exist (ATS, Moneyline, Total)
- View model age (days since last training)
- Last training timestamp
- Health status indicators (✅ OK / ❌ Missing)

#### 5. **Model Metrics**
- Track model accuracy, ROI, edge distribution over time
- Historical performance analysis
- Market-specific metrics

#### 6. **Risk Simulator**
- Monte Carlo simulation for bankroll volatility
- Customize: trials, starting bankroll, expected edge
- View: average final balance, median, profitability percentage
- Best/worst case scenarios

#### 7. **ELO Diagnostics**
- Current ELO ratings (Offensive, Defensive, Composite)
- Team rankings by ELO
- Historical ELO trends

#### 8. **Logs**
- System log viewer (last 500 lines)
- Console-style display with monospace font
- Real-time refresh capability
- Clear display function

---

## 🛠️ Data Pipeline Workflow

### Complete Training Flow (from Admin Dashboard):

```
1. Download Historical Data
   ↓
2. Scrape Current Injuries
   ↓
3. Train ML Models + Backtest
   ├─ Feature Engineering (120+ features)
   ├─ Train XGBoost Models (ATS, ML, Total)
   ├─ Run Backtest (1,230+ games)
   └─ Compute Brier Score
   ↓
4. Hyperparameter Tuning (optional)
   ↓
5. Generate Today's Predictions
   ↓
6. Export Reports (CSV, JSON)
```

### Automated Execution:
All pipeline steps run in **WorkerThread** (background thread) so UI stays 100% responsive.

Progress updates stream to console in real-time.

---

## 🎨 Design Philosophy

### Professional Sportsbook Theme

**Colors**:
- Background: `#0a0e14` (deep black)
- Panels: `#121820`, `#1a1f2e` (dark cards)
- Text: `#ffffff` (pure white - maximum contrast)
- Accent: `#2563eb` (professional blue like FanDuel)
- Positive/Edge: `#34d399` (bright green)
- Negative: `#f87171` (bright red)
- Borders: `#2a3441` (subtle)

**Typography**:
- All text pure white for maximum readability
- Full team names ("Los Angeles Lakers" not "[LAL]")
- Bold headers, clean sans-serif fonts
- Monospace for console/logs

**No Emojis**: Professional interface (except in internal comments)

---

## 🔧 Technical Architecture

### Separation of Concerns:

| Component | Main Dashboard | Admin Dashboard |
|-----------|---------------|-----------------|
| **Focus** | Betting Operations | System Administration |
| **User** | Bettor | Power User / Dev |
| **Complexity** | Low | High |
| **Update Frequency** | High (live games) | Low (training cycles) |
| **Database Access** | Read-only predictions | Read/Write (training, calibration) |

### Shared Infrastructure:
- **Database**: `nba_betting_data.db` (SQLite)
  - Tables: predictions, bets, bankroll_history, backtest_history, elo_history, calibration_data
- **Models**: `models/` directory
  - `model_v5_ats.xgb`, `model_v5_ml.xgb`, `model_v5_total.xgb`
- **Data**: `data/` directory
  - `master_training_data_v5.csv`
- **Logs**: `nba_betting_dashboard.log`, `admin_dashboard.log`, `nba_system.log`

### Threading Model:
- **Main Dashboard**: UI thread + optional live tracking timer
- **Admin Dashboard**: UI thread + WorkerThread for all pipeline tasks
- **Backtest**: Runs in WorkerThread (10,000+ play iterations safe in background)

---

## 📋 Workflow Examples

### Daily Betting Routine

**Morning Preparation**:
1. Launch main betting dashboard
2. Click "Refresh Data" to load today's predictions
3. Review "Synopsis" tab for top picks
4. Check "Today's Games" tab for full slate

**During Games**:
1. Switch to "Live Tracker" tab
2. Click "Start Live Tracking" (30s auto-refresh)
3. Monitor win probabilities and in-play opportunities

**After Games**:
1. Go to "Bet History" tab
2. Click "Add Bet" to log placed bets
3. Review results and ROI

### Weekly Model Retraining

**From Admin Dashboard**:
1. Open "System Admin" tab
2. Click "1. Download Historical Data" (wait for completion)
3. Click "2. Scrape Current Injuries"
4. Click "3. Train ML Models + Backtest"
   - Watch console for progress updates
   - Backtest runs automatically after training
   - Brier score computed and saved
5. Check "Backtest" tab to verify quality
6. Review "Model Health" tab to confirm models updated
7. Optionally run "Calibration" analysis

**Result**: Models retrained, calibrated, backtested, ready for production

### Performance Monitoring

**From Admin Dashboard**:
1. **Model Health Tab**: Check if models are recent (age < 7 days recommended)
2. **Calibration Tab**: Verify Brier score < 0.15 (excellent) or < 0.20 (good)
3. **Model Metrics Tab**: Track ROI and accuracy trends
4. **ELO Diagnostics Tab**: Ensure ratings are up-to-date
5. **Logs Tab**: Review system errors or warnings

---

## ⚠️ Critical Reminders

### ALWAYS Calibrate Probabilities
- Never use raw XGBoost probabilities for Kelly sizing
- All predictions must pass through `CalibrationFitter.apply()`
- Track calibration health via Brier score

### Kelly Sizing Safety
- Default: Quarter Kelly (0.25x)
- Max bet: 5% of bankroll
- Drawdown scaling: >20% DD → 25% Kelly, >10% DD → 50% Kelly
- Commission-adjusted edge: `edge - KALSHI_BUY_COMMISSION`

### Model Retraining Schedule
- Retrain weekly (Sundays) or after 50+ new games
- Always run backtest after retraining
- Check Brier score before deploying models
- Use time-series split for validation (no shuffling)

### Data Pipeline Integrity
- Run steps 1-6 in sequence (don't skip steps)
- Wait for each step to complete before next
- Monitor console output for errors
- Verify database updates after each step

---

## 🐛 Troubleshooting

### Admin Dashboard Issues

**"Backtest button does nothing"**:
- Backtest runs in background (WorkerThread)
- Check console output for progress
- May take 60-120 seconds
- Look for "Backtest Complete" message

**"Data Pipeline buttons unreadable"**:
- Fixed in latest version with QPalette + inline stylesheets
- White text on blue buttons (#2563eb background)
- If still unreadable, check Windows display scaling

**"Training fails with ImportError"**:
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check virtual environment activated
- Verify `V5_train_all.py` exists in project root

### Main Dashboard Issues

**"No predictions showing"**:
- Models must be trained first (use admin dashboard)
- Check database has `predictions` table
- Run "Refresh Data" button
- Verify today's date has games scheduled

**"Live Tracker not updating"**:
- Click "Start Live Tracking" to begin
- Requires active internet connection
- NBA API may have rate limits
- Check console for errors

---

## 📁 File Structure

```
New Basketball Model/
├── NBA_Dashboard_v6_Streamlined.py   # Main betting dashboard
├── admin_dashboard.py                 # Admin/diagnostics dashboard
├── launch_dashboards.py               # Dual-dashboard launcher
├── NBA_Dashboard_Enhanced_v5.py       # Legacy (full dashboard)
├── V5_train_all.py                    # Model training script
├── nba_betting_data.db                # SQLite database
├── core/
│   ├── prediction_engine.py
│   ├── kelly_optimizer.py
│   ├── live_model_backtester.py
│   ├── live_win_probability_model.py
│   ├── calibration_fitter.py
│   ├── nba_stats_collector_v2.py
│   └── ...
├── models/
│   ├── model_v5_ats.xgb
│   ├── model_v5_ml.xgb
│   └── model_v5_total.xgb
├── data/
│   └── master_training_data_v5.csv
└── logs/
    ├── nba_betting_dashboard.log
    ├── admin_dashboard.log
    └── nba_system.log
```

---

## 🔄 Migration from v5

### For Existing Users:

**Old Workflow** (single dashboard):
```bash
python NBA_Dashboard_Enhanced_v5.py
```
- All features in one window (cluttered)
- Admin panel as dialog overlay
- Backtest could freeze UI

**New Workflow** (dual dashboards):
```bash
python launch_dashboards.py
```
- Betting ops in main window (clean)
- Admin in separate window (powerful)
- Backtest runs safely in background

**No Data Loss**: Both use same `nba_betting_data.db` database

**Backward Compatible**: Old dashboard still works if preferred

---

## 🎓 Best Practices

1. **Separate Workflows**: Use main dashboard for betting, admin dashboard for system management
2. **Regular Retraining**: Retrain models weekly to capture recent trends
3. **Monitor Calibration**: Keep Brier score < 0.20 for reliable probabilities
4. **Conservative Sizing**: Start with quarter Kelly (0.25x) until edge proven
5. **Track Everything**: Log all bets in database for future analysis
6. **Check Health**: Review model health tab weekly
7. **Backup Database**: Copy `nba_betting_data.db` before major updates

---

## 📞 Support

**Issues with Dashboards**:
- Check logs: `nba_betting_dashboard.log`, `admin_dashboard.log`
- Verify Python 3.12+ installed
- Ensure virtual environment activated
- Run: `pip install -r requirements.txt`

**Model Performance Issues**:
- Check calibration Brier score
- Review backtest results
- Verify ELO ratings updated
- Consider retraining with more data

**Database Issues**:
- Backup: `cp nba_betting_data.db nba_betting_data_backup.db`
- Rebuild tables from admin dashboard (System Admin tab)
- Check file permissions

---

## 🚀 Future Enhancements

- [ ] Live odds integration (Kalshi API)
- [ ] Automated bet placement
- [ ] SMS/email alerts for high-edge opportunities
- [ ] Advanced charting (matplotlib/plotly)
- [ ] Multi-season backtest comparison
- [ ] Ensemble model voting
- [ ] Real-time feature importance
- [ ] Bankroll optimization simulations

---

**Built with**: Python 3.12, PyQt6, XGBoost, scikit-learn, pandas, numpy

**License**: Proprietary

**Version**: 6.0 (Dual Dashboard Architecture)

**Last Updated**: November 2025
