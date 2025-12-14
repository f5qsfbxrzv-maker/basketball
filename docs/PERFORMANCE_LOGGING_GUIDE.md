# Performance Logging System - How It Works

## Overview
The paper trading tracker automatically logs all predictions and tracks their outcomes for performance analysis. This system is integrated into the dashboard for seamless bet tracking.

---

## Architecture

### Components
1. **PaperTradingTracker** (`paper_trading_tracker.py`)
   - Core tracking engine
   - SQLite database storage
   - Automatic outcome updates

2. **Dashboard Integration** (`nba_gui_dashboard_v2.py`)
   - Automatic prediction logging on refresh
   - Performance tab for viewing results
   - Manual outcome updates

3. **Database Table** (`paper_predictions`)
   - Stores all predictions with metadata
   - Tracks outcomes and profit/loss
   - Unique constraint prevents duplicates

---

## Prediction Logging Workflow

### 1. **Automatic Logging** (Lines 422-449 in dashboard)
When you generate predictions in the dashboard:

```python
# For every bet with edge ≥ MIN_EDGE (3%):
if PAPER_TRADING_AVAILABLE and best_bet and best_bet['edge'] >= MIN_EDGE:
    tracker = PaperTradingTracker()
    tracker.log_prediction(
        game_date=game_date,
        home_team=home_team,
        away_team=away_team,
        prediction_type=pred_type,
        predicted_winner=predicted_winner,
        model_probability=model_prob,
        fair_probability=fair_prob,
        odds=odds,
        edge=edge,
        stake=stake  # Calculated via Kelly criterion
    )
```

**What gets logged:**
- Game details (date, teams)
- Prediction type (Moneyline/ATS/Total)
- Predicted outcome
- Probabilities (model vs market)
- American odds
- Calculated edge
- Kelly stake size
- Timestamp

### 2. **Database Storage**
All predictions stored in: `data/live/nba_betting_data.db`

**Table Schema:**
```sql
CREATE TABLE paper_predictions (
    id INTEGER PRIMARY KEY,
    game_date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    prediction_type TEXT NOT NULL,
    predicted_winner TEXT,
    model_probability REAL,
    fair_probability REAL,
    odds REAL,
    edge REAL,
    stake REAL,
    outcome TEXT,           -- WIN/LOSS (NULL until game settles)
    actual_winner TEXT,     -- Actual game winner
    profit_loss REAL,       -- $ profit or loss
    timestamp TEXT NOT NULL,
    UNIQUE(game_date, home_team, away_team, prediction_type)
)
```

**Duplicate Prevention:**
The UNIQUE constraint ensures each game/prediction type combination is only logged once. If you refresh predictions for the same game, it updates the existing entry rather than creating duplicates.

---

## Outcome Updates

### 3. **Manual Update** (Performance Tab)
Click **"🔄 Update Outcomes"** button to fetch game results:

```python
def update_outcomes(self):
    tracker = PaperTradingTracker()
    tracker.update_outcomes_from_api(game_date)
```

**What it does:**
1. Queries `paper_predictions` for pending bets (outcome IS NULL)
2. Looks up actual results from `game_results` table
3. Determines actual winner (home_score vs away_score)
4. Compares to predicted winner
5. Calculates profit/loss based on American odds:
   - **WIN**: 
     * Positive odds: `profit = stake × (odds / 100)`
     * Negative odds: `profit = stake × (100 / |odds|)`
   - **LOSS**: `profit = -stake`
6. Updates database with outcome, actual_winner, profit_loss

### 4. **Automatic Updates** (Optional)
You can automate outcome updates using the nightly tasks script:

```python
# In scripts/nightly_tasks.py
from paper_trading_tracker import PaperTradingTracker

tracker = PaperTradingTracker()
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
tracker.update_outcomes_from_api(yesterday)
```

---

## Performance Metrics

### 5. **Load Performance** (Performance Tab)
Click **"📈 Load Performance"** to view aggregate metrics:

```python
report = tracker.generate_performance_report()
```

**Metrics Calculated:**
- **Total Bets**: Count of all predictions
- **Win Rate**: % of winning bets
- **ROI**: (Total Profit / Total Staked) × 100%
- **Total Profit/Loss**: Sum of all profit_loss values
- **Average Stake**: Mean stake size across all bets
- **Brier Score**: Calibration quality (lower = better)
- **Edge Buckets**: Performance by edge range
  - 15%+ edge
  - 10-15% edge
  - 7-10% edge
  - 5-7% edge
  - 3-5% edge

**Example Report:**
```
Total Bets: 47
Win Rate: 63.8%
ROI: 18.2%
Total Profit/Loss: $412.50
Average Stake: $48.32

Calibration:
  Brier Score: 0.1842

Edge Buckets:
  15%+ edge: 8 bets, ROI: 42.1%
  10-15% edge: 12 bets, ROI: 28.3%
  7-10% edge: 15 bets, ROI: 15.7%
  5-7% edge: 10 bets, ROI: 8.4%
  3-5% edge: 2 bets, ROI: -12.5%
```

---

## Color Gradient Legend

The dashboard now uses a **5-level color gradient** for better visual distinction:

| Edge Range | Color | RGB | Description |
|------------|-------|-----|-------------|
| **≥ 15%** | 🟢 Dark Green | (34, 139, 34) | Excellent edge - Forest Green |
| **10-15%** | 🟢 Lime Green | (50, 205, 50) | Strong edge - Bright Green |
| **7-10%** | 🟡 Gold | (255, 215, 0) | Good edge - Yellow |
| **5-7%** | 🟠 Dark Orange | (255, 140, 0) | Moderate edge - Orange |
| **3-5%** | 🟠 Peach | (255, 200, 150) | Minimum edge - Light Orange |
| **< 3%** | ⚪ No Color | Default | Below betting threshold |

**Text Colors:**
- White text on dark green (15%+) for contrast
- Black text on all other colors

---

## Best Practices

### When to Use Manual Updates
- **After games finish** - Update same day for immediate feedback
- **Morning after** - Update previous day's games
- **Weekly reviews** - Batch update for weekly performance analysis

### When to Use Automatic Logging
- **Always enabled** - Every prediction refresh auto-logs to database
- **No action needed** - Happens transparently in background
- **Safe to refresh** - UNIQUE constraint prevents duplicates

### Monitoring Performance
1. **Daily**: Check Performance tab after games settle
2. **Weekly**: Review edge bucket performance
3. **Monthly**: Analyze Brier score trends for calibration quality
4. **Season**: Compare actual ROI vs expected edge

---

## Troubleshooting

### "Paper trading tracker not available"
**Cause**: `paper_trading_tracker.py` not found or import failed

**Solution**: Verify file exists at:
```
c:\Users\d76do\OneDrive\Documents\New Basketball Model\paper_trading_tracker.py
```

### No outcomes showing
**Cause**: Game results not in `game_results` table

**Solution**: 
1. Check if game has finished
2. Run data ingestion script:
   ```powershell
   python src/services/nba_stats_collector_v2.py
   ```
3. Click "🔄 Update Outcomes" again

### Duplicate predictions
**Cause**: UNIQUE constraint failing (shouldn't happen)

**Solution**:
```sql
-- Check for duplicates
SELECT game_date, home_team, away_team, prediction_type, COUNT(*)
FROM paper_predictions
GROUP BY game_date, home_team, away_team, prediction_type
HAVING COUNT(*) > 1;
```

### Incorrect profit calculations
**Cause**: Odds format mismatch (American vs Decimal)

**Solution**: Verify odds are in American format (-110, +150, etc.)
- Dashboard uses American odds from Kalshi
- Profit calculations assume American odds

---

## Integration with Calibration System

Performance logging feeds into the broader calibration infrastructure:

1. **Predictions logged** → `paper_predictions` table
2. **Outcomes updated** → WIN/LOSS recorded
3. **Calibration fitter** → Uses outcomes to refit isotonic/Platt models
4. **Updated models** → Improve future probability predictions
5. **Better edges** → More accurate betting opportunities

**Data Flow:**
```
Dashboard Predictions
        ↓
PaperTradingTracker.log_prediction()
        ↓
paper_predictions table
        ↓
PaperTradingTracker.update_outcomes_from_api()
        ↓
Outcome data (WIN/LOSS)
        ↓
CalibrationFitter.auto_refit_nightly()
        ↓
Improved probability calibration
        ↓
Better future predictions
```

---

## Quick Reference

### Common Operations

**View all predictions:**
```sql
SELECT * FROM paper_predictions ORDER BY timestamp DESC LIMIT 50;
```

**Check pending predictions:**
```sql
SELECT game_date, home_team, away_team, prediction_type, edge, stake
FROM paper_predictions
WHERE outcome IS NULL
ORDER BY game_date;
```

**Calculate current ROI:**
```sql
SELECT 
    COUNT(*) as total_bets,
    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    SUM(profit_loss) as total_profit,
    SUM(stake) as total_staked,
    SUM(profit_loss) * 100.0 / SUM(stake) as roi
FROM paper_predictions
WHERE outcome IS NOT NULL;
```

**Performance by edge bucket:**
```sql
SELECT 
    CASE 
        WHEN edge >= 0.15 THEN '15%+'
        WHEN edge >= 0.10 THEN '10-15%'
        WHEN edge >= 0.07 THEN '7-10%'
        WHEN edge >= 0.05 THEN '5-7%'
        ELSE '3-5%'
    END as edge_bucket,
    COUNT(*) as bets,
    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    SUM(profit_loss) * 100.0 / SUM(stake) as roi
FROM paper_predictions
WHERE outcome IS NOT NULL
GROUP BY edge_bucket
ORDER BY 
    CASE 
        WHEN edge >= 0.15 THEN 1
        WHEN edge >= 0.10 THEN 2
        WHEN edge >= 0.07 THEN 3
        WHEN edge >= 0.05 THEN 4
        ELSE 5
    END;
```

---

## Summary

**Automatic Features:**
- ✅ Predictions logged on every dashboard refresh (edge ≥ 3%)
- ✅ Duplicate prevention via UNIQUE constraint
- ✅ Kelly stake calculation
- ✅ Database persistence

**Manual Actions:**
- 🔘 Click "🔄 Update Outcomes" to fetch game results
- 🔘 Click "📈 Load Performance" to view metrics

**Performance Analysis:**
- 📊 Win rate, ROI, profit/loss tracking
- 📊 Edge bucket performance breakdown
- 📊 Calibration quality (Brier score)
- 📊 Prediction log with full details

**System Health:**
- 🎯 Continuous calibration improvement via outcomes
- 🎯 Kelly optimization based on real performance
- 🎯 Drawdown scaling integrated with tracker
- 🎯 Complete audit trail for all predictions
