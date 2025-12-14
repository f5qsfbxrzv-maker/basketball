# NBA Betting Dashboard - Quick Start

## 🚀 Launch Dashboard

**Easiest way:** Double-click `RUN_DASHBOARD.bat`

**Or from terminal:**
```bash
python NBA_Dashboard_Enhanced_v5.py
```

## 📁 Project Structure

```
NBA Betting System/
├── RUN_DASHBOARD.bat          # ← CLICK THIS TO START
├── NBA_Dashboard_Enhanced_v5.py   # Main dashboard application
├── .kalshi_credentials        # API credentials (keep private)
├── nba_betting_data.db       # SQLite database
│
├── core/                      # Core prediction & data modules
│   ├── prediction_engine.py
│   ├── feature_calculator_v5.py
│   ├── nba_stats_collector_v2.py
│   ├── calibration_fitter.py
│   ├── kelly_optimizer.py
│   └── off_def_elo_system.py
│
├── models/                    # Trained ML models
│   ├── model_v5_ats.xgb
│   ├── model_v5_ml.xgb
│   └── model_v5_total.xgb
│
├── data/                      # Training data & caches
│   └── master_training_data_v5.csv
│
├── scripts/                   # Utilities & automation
│   ├── retrain_pipeline.py
│   └── nightly_tasks.py
│
├── tests/                     # Test suite
│   └── test_*.py
│
└── logs/                      # Application logs
    └── nba_system.log

```

## ⚙️ Configuration

Edit `.kalshi_credentials` to set your Kalshi API key and private key.

## 📊 Features

- Live game predictions with ML models
- Kalshi market integration (moneyline odds)
- Kelly criterion bet sizing
- Calibration tracking & reliability curves
- Injury impact analysis
- ELO rating system

## 🔧 Troubleshooting

**Dashboard won't start:**
- Make sure `.venv` is activated
- Check `.kalshi_credentials` exists
- View logs in `nba_system.log`

**No Kalshi odds:**
- Verify API credentials in `.kalshi_credentials`
- Check if markets exist for today's games
- Look for Kalshi errors in `nba_system.log`
