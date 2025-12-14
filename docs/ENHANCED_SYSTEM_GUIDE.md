# NBA Betting System - Enhanced Live Betting Quick Start

## 🚀 NEW FEATURES INTEGRATED

### Live Betting System
- **Live Bet Tracker**: Real-time opportunity detection with Kelly criterion position sizing
- **Live Win Probability Model**: Z-score based statistical approach for in-game predictions
- **Enhanced Stats Collector**: Live scoreboard integration with play-by-play data
- **Enhanced Dashboard**: Real-time GUI with live betting interface

### Performance Enhancements  
- **Feature Calculator Enhanced**: In-memory pandas caching for massive speed improvements
- **Live Model Backtester**: Hyperparameter tuning for live betting models

## 🖥️ ENHANCED DASHBOARD FEATURES

### New Dashboard Tabs:
1. **📊 Overview**: System metrics, bankroll tracking, performance charts
2. **🏀 Live Games**: Real-time game monitoring with win probabilities  
3. **💰 Live Betting**: Live opportunities detection and bet execution
4. **🔮 Predictions**: Pre-game model predictions and recommendations
5. **📈 Performance**: Advanced analytics and historical tracking
6. **⚙️ Settings**: System configuration and risk management

### Live Betting Interface:
- Auto-refresh live games every 15 seconds
- Real-time win probability calculations
- Kelly criterion position sizing
- Automated bet execution (when enabled)
- Live P&L tracking

## 🏃‍♂️ QUICK START

### 1. Launch Enhanced System:
```python
python main.py
```

### 2. Enable Live Betting:
- Open the enhanced dashboard (auto-launches)
- Go to "💰 Live Betting" tab
- Check "Live Betting Enabled" 
- System will auto-detect opportunities

### 3. Monitor Live Games:
- Go to "🏀 Live Games" tab
- Games auto-refresh every 15 seconds
- Win probabilities update in real-time
- Click "🔄 Refresh" for manual updates

## 🔧 CONFIGURATION

### Key Settings in config.json:
```json
{
  "live_betting_enabled": true,
  "kelly_fraction": 0.02,
  "auto_trading": false,
  "refresh_interval": 15,
  "auto_launch_gui": true
}
```

### Risk Management:
- **Kelly Fraction**: Default 2% (adjustable in Settings tab)
- **Auto Trading**: Disabled by default (manual approval required)
- **Live Betting**: Can be toggled on/off in real-time

## 📊 LIVE BETTING WORKFLOW

### 1. Game Detection:
- System monitors live NBA games automatically
- Collects real-time scores and game state

### 2. Opportunity Analysis:
- Calculates live win probabilities using Z-score methodology
- Compares with market odds from multiple sources
- Identifies positive expected value opportunities

### 3. Position Sizing:
- Uses Kelly criterion for optimal bet sizing
- Considers current bankroll and risk parameters
- Suggests stake amounts based on edge and confidence

### 4. Execution:
- Manual approval required (unless auto-trading enabled)
- Integrates with Kalshi API for automated placement
- Tracks active positions and P&L in real-time

## 🛡️ SAFETY FEATURES

### Risk Controls:
- **Paper Trading Mode**: Test without real money
- **Manual Approval**: Review all bets before execution  
- **Position Limits**: Kelly fraction caps maximum bet size
- **Real-time Monitoring**: Track all positions continuously

### Data Validation:
- Multiple odds sources for comparison
- Real-time data feeds with error handling
- Automatic fallbacks if data sources fail

## 🔍 MONITORING AND ALERTS

### Dashboard Monitoring:
- Real-time bankroll tracking
- Live game scores and probabilities
- Active bet positions and P&L
- System health and connection status

### Performance Tracking:
- Win rate calculations
- ROI measurements  
- Historical performance charts
- Model accuracy metrics

## 🚨 TROUBLESHOOTING

### Common Issues:
1. **Dashboard not loading**: Check if enhanced dashboard files are present
2. **Live data not updating**: Verify internet connection and API credentials
3. **Betting disabled**: Check live_betting_enabled setting in config
4. **No opportunities found**: Normal during low-edge periods

### Fallback Systems:
- Enhanced components fall back to basic versions if issues occur
- System continues in headless mode if GUI fails
- Manual trading mode if auto-execution fails

## 📁 FILE STRUCTURE (AFTER CLEANUP)

### Core Components:
- `main.py` - System orchestrator with enhanced integration
- `live_bet_tracker.py` - Live betting opportunity tracking
- `live_win_probability_model.py` - Real-time win probability calculations
- `feature_calculator_enhanced.py` - High-performance feature engineering
- `NBA_Betting_Dashboard_GUI_Enhanced.py` - Live betting dashboard

### Archived Files:
- `archive/` - Contains old test files and deprecated components

## 🎯 NEXT STEPS

1. **Test System**: Run in paper trading mode first
2. **Verify Data Feeds**: Ensure live data is updating correctly  
3. **Configure Risk**: Set appropriate Kelly fraction and limits
4. **Monitor Performance**: Track results and adjust parameters
5. **Enable Auto-Trading**: Only after thorough testing

---

**⚠️ Important**: Always test thoroughly in paper trading mode before enabling live trading with real money.