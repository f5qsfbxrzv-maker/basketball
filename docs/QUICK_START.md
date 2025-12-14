# NBA Betting Dashboard - Organized Structure

## 🚀 **HOW TO LAUNCH**

### **Option 1: Double-click** (Easiest)
```
RUN_DASHBOARD.bat
```

### **Option 2: Command Line**
```bash
python NBA_Dashboard_Enhanced_v5.py
```

---

## 📁 **New Organized Structure**

```
NBA Betting System/
│
├── RUN_DASHBOARD.bat          ← DOUBLE-CLICK TO START
├── NBA_Dashboard_Enhanced_v5.py   Main GUI application
├── .kalshi_credentials        API credentials (private)
├── nba_betting_data.db       Database (predictions, ELO, bets)
├── config.json               System configuration
├── requirements.txt          Python dependencies
├── README.md                 This file
│
├── core/                     🔧 Core Prediction System
│   ├── prediction_engine.py      Main prediction orchestrator
│   ├── feature_calculator_v5.py  120+ feature engineering
│   ├── calibration_fitter.py     Isotonic/Platt calibration
│   ├── calibration_logger.py     Prediction tracking
│   ├── kelly_optimizer.py        Bet sizing (Kelly criterion)
│   ├── nba_stats_collector_v2.py NBA API data ingestion
│   ├── injury_data_collector_v2.py Injury scraping & impact
│   ├── off_def_elo_system.py     Offensive/Defensive ELO
│   ├── kalshi_client.py          Kalshi market integration
│   ├── ml_model_trainer.py       XGBoost/LightGBM training
│   ├── advanced_models.py        Poisson/Bayesian models
│   └── mlops_infrastructure.py   Model registry & monitoring
│
├── utils/                    🛠️ Shared Utilities
│   ├── constants.py              All configuration constants
│   ├── data_models.py            Dataclass definitions
│   ├── logger_setup.py           Structured logging
│   ├── interfaces.py             Abstract interfaces
│   └── config_manager.py         Config file handler
│
├── models/                   🤖 Trained Models
│   ├── model_v5_ats.xgb          Against-the-spread model
│   ├── model_v5_ml.xgb           Moneyline model
│   └── model_v5_total.xgb        Totals (over/under) model
│
├── data/                     💾 Training Data
│   └── master_training_data_v5.csv
│
├── scripts/                  ⚙️ Automation & Utilities
│   ├── retrain_pipeline.py       Automated model retraining
│   ├── nightly_tasks.py          Scheduled calibration updates
│   └── v5_rolling_backtest_enhanced.py  Performance testing
│
├── tests/                    ✅ Test Suite
│   └── test_*.py                 Unit & integration tests
│
├── logs/                     📊 Application Logs
│   ├── nba_system.log           Main application log
│   ├── backtest_logs/           Backtest results
│   └── prediction_logs/         Prediction history
│
├── archive/                  📦 Old/Unused Files
│   └── check_*.py, test_*.py, debug_*.py (moved out of root)
│
└── docs/                     📚 Documentation
    ├── README.md
    ├── QUICK_START.md
    └── guides/

```

---

## ✨ **What Changed**

### Before (Messy)
- 150+ files in root directory
- Test files mixed with core code
- Import paths unclear
- Hard to find anything

### After (Organized)
- ✅ Core modules in `core/`
- ✅ Utilities in `utils/`
- ✅ Tests in `tests/`
- ✅ Old files in `archive/`
- ✅ Clear import structure
- ✅ Easy launcher (`RUN_DASHBOARD.bat`)

---

## 🎯 **Key Features**

1. **Live Predictions** - ML models for ATS, Moneyline, Totals
2. **Kalshi Integration** - Real-time moneyline market prices
3. **Kelly Criterion** - Optimal bet sizing with edge calculation
4. **Calibration System** - Isotonic regression for probability reliability
5. **ELO Ratings** - Separate offensive/defensive team strength
6. **Injury Analysis** - Replacement-level impact modeling
7. **Risk Management** - Drawdown scaling, bankroll tracking

---

## 🔧 **Configuration**

### Kalshi API Setup
Edit `.kalshi_credentials`:
```
API_KEY=your-api-key-here
PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----
...your private key...
-----END RSA PRIVATE KEY-----
```

### System Settings
Edit `config.json` for:
- Kelly fraction (default: 0.25)
- Maximum bet size
- Calibration thresholds
- Update intervals

---

## 📊 **Dashboard Tabs**

1. **Predictions** - Today's games with odds & bet sizing
2. **Calibration** - Reliability curves & Brier scores
3. **Scenarios** - Monte Carlo simulation
4. **Model Health** - Performance tracking
5. **Metrics** - Historical statistics
6. **Risk** - Bankroll & drawdown monitoring
7. **Advanced** - ELO trends & feature analysis
8. **Logs** - System event viewer

---

## 🐛 **Troubleshooting**

**Dashboard won't start:**
```bash
# Check Python environment
python --version  # Should be 3.12+

# Activate virtual environment
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

**No Kalshi odds showing:**
- Verify `.kalshi_credentials` exists
- Check `nba_system.log` for API errors
- Ensure markets exist for today's games

**Import errors:**
- Run from project root directory
- Don't rename core folders

---

## 📈 **Performance**

- **Prediction Accuracy**: ~58-62% ATS (backtested)
- **Calibration**: Brier score < 0.15 (well-calibrated)
- **Kelly Sizing**: Quarter-Kelly with drawdown scaling
- **Edge Threshold**: Minimum 3% edge required

---

## 🔒 **Security**

- Keep `.kalshi_credentials` private (in `.gitignore`)
- Never commit API keys to version control
- Use environment variables for production

---

## 📝 **Next Steps**

1. **Launch**: Double-click `RUN_DASHBOARD.bat`
2. **Configure**: Set Kalshi credentials
3. **Predict**: View today's games
4. **Bet**: Follow Kelly recommendations
5. **Track**: Monitor calibration & results

---

## 🆘 **Support**

- **Logs**: Check `nba_system.log` for errors
- **Documentation**: See `docs/` folder
- **Tests**: Run `pytest tests/` to validate setup

---

**Version**: 5.0 (Organized)  
**Last Updated**: November 20, 2025  
**Python**: 3.12+  
**License**: Private Use
