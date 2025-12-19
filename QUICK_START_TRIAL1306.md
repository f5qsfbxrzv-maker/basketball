# Trial 1306 Quick Start Guide 🚀

## Launch Dashboard

```bash
cd "c:\Users\d76do\OneDrive\Documents\New Basketball Model"
python nba_gui_dashboard_v2.py
```

## Daily Workflow

### 1. Morning - Get Predictions ☀️

1. Launch dashboard
2. Go to **📊 Predictions** tab
3. Click **🔄 Refresh Predictions**
4. Review qualifying bets:
   - ✅ **Favorites** (odds < 2.00): Edge ≥ 2%
   - ✅ **Underdogs** (odds ≥ 2.00): Edge ≥ 10%
5. **Bets auto-logged** to database!

### 2. Evening - Grade Bets 🌙

1. Go to **📈 Trial 1306 Metrics** tab
2. Click **✅ Grade Yesterday's Bets**
3. View updated:
   - ROI (target: 49.7%)
   - Win Rate (target: >55%)
   - Bankroll progress
   - Drawdown status

## Key Tabs

### 📊 Predictions
- Today's games with Trial 1306 predictions
- Edge calculations
- Stake recommendations
- Live injuries display
- **NEW**: Live odds from Kalshi (add API keys)

### 📈 Trial 1306 Metrics
- **Overall**: Total bets, win rate, avg edge
- **Financial**: ROI, P/L, bankroll
- **Calibration**: Brier score, log loss
- **Bankroll Tracker**: Visual progress, drawdown
- **Bet Type Performance**: Breakdown by type
- **Recent Bets**: Last 20 with outcomes
- **Actions**: Refresh, Grade, Export

### 📉 Legacy Performance
- Historical tracking (old system)

### ⚙️ Settings
- Bankroll management
- Threshold adjustments
- Model configuration

## Metrics Explained

### ROI (Return on Investment)
- **Formula**: Total P/L / Total Staked
- **Target**: 49.7% (Trial 1306 validated)
- **Status**: Green = profitable, Red = losing

### Win Rate
- **Formula**: Wins / (Wins + Losses)
- **Target**: >55%
- **Breakeven**: 52.4% at -110 odds

### Brier Score (Calibration)
- **Formula**: Avg((Predicted - Actual)²)
- **Target**: <0.20 (excellent), <0.25 (good)
- **Lower = Better calibration**

### Max Drawdown
- **Formula**: (Peak - Current) / Peak
- **Target**: <10%
- **Triggers Kelly reduction at >10%**

## Enable Live Odds (Optional)

1. Get Kalshi API credentials from https://kalshi.com
2. Edit `config/kalshi_config.json`:
   ```json
   {
     "api_key": "your_key_here",
     "api_secret": "your_secret_here",
     "environment": "demo"
   }
   ```
3. Restart dashboard
4. Look for: `[ODDS] ... source=kalshi`

## Troubleshooting

### "No predictions available"
- Check today's date
- Ensure games scheduled
- Click "Show All Games"

### "BetTracker not available"
```bash
python src/core/bet_tracker.py
```

### "No bets graded"
- Ensure game_logs table has recent data
- Check date format: YYYY-MM-DD
- Verify games finished

## Quick Commands

### Test Systems
```bash
# Test BetTracker
python src/core/bet_tracker.py

# Test LiveOddsFetcher
python src/services/live_odds_fetcher.py

# Check database tables
python check_db_tables.py
```

### Manual Grading
```bash
python -c "from src.core.bet_tracker import BetTracker; bt = BetTracker(); bt.grade_bets('2024-12-14')"
```

### Export Bets
```bash
python -c "from src.core.bet_tracker import BetTracker; bt = BetTracker(); bt.get_recent_bets(1000).to_csv('my_bets.csv')"
```

## Expected Performance

Based on Trial 1306 backtesting:

| Metric | Target | Notes |
|--------|--------|-------|
| ROI | 49.7% | Grid search optimized |
| Win Rate | 55-60% | With 2%/10% thresholds |
| Avg Edge | 3-5% | Per qualifying bet |
| Bet Frequency | 30-40% | Of all games |
| Brier Score | <0.25 | Good calibration |
| Kelly | 25% | Quarter Kelly |

## Risk Management

- **Max Bet**: 5% of bankroll
- **Commission**: 3% Kalshi fee deducted from edge
- **Drawdown Protection**:
  - 0-5% DD: 75% Kelly
  - 5-10% DD: 50% Kelly
  - 10-20% DD: 25% Kelly
  - >20% DD: 12.5% Kelly

## Files Location

```
New Basketball Model/
├── nba_gui_dashboard_v2.py          # Main dashboard
├── src/
│   ├── core/
│   │   └── bet_tracker.py           # Bet tracking system
│   ├── services/
│   │   ├── live_odds_fetcher.py     # Kalshi odds
│   │   ├── kalshi_client.py         # API client
│   │   └── live_injury_updater.py   # ESPN injuries
│   └── dashboard/
│       └── metrics_tab.py           # Metrics UI
├── config/
│   └── kalshi_config.json           # API credentials
├── data/live/
│   └── nba_betting_data.db         # Main database
└── models/
    └── xgboost_22features_trial1306_20251215_212306.json
```

## Support

📚 Full docs: `TRIAL1306_DASHBOARD_INTEGRATION.md`
🏀 Model info: `README_TRIAL1306.md`
⚡ Quick ref: `QUICK_REFERENCE.md`

---

**You're ready to make Brandi money! 💰**
