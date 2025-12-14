# NBA Betting System - Clean File Structure

## 📁 Directory Organization

### **Root Directory** (Core Application Files)
Production-ready files only. No duplicates, no test files.

**Core System:**
- `main.py` - Main application entry point
- `health_check.py` - System health diagnostics
- `config.json` - System configuration
- `dashboard_settings.json` - Dashboard preferences

**Data Collection:**
- `nba_stats_collector_v2.py` - NBA API data collector (with PACE calculation)
- `injury_data_collector_v2.py` - Injury data scraper (CBS Sports + ESPN + Historical)

**Feature Engineering:**
- `feature_calculator_v5.py` - In-memory optimized feature calculator (50-100x faster)
- `feature_analyzer.py` - Feature importance analysis
- `dynamic_elo_calculator.py` - Dynamic ELO rating system

**ML & Prediction:**
- `ml_model_trainer.py` - XGBoost model trainer
- `model_comparator.py` - Model performance comparison
- `kelly_optimizer.py` - Kelly criterion position sizing

**User Interface:**
- `NBA_Dashboard_Enhanced_v5.py` - Main dashboard (6 tabs, injury tracking, DB explorer)

**Live Trading:**
- `live_bet_tracker.py` - Live bet monitoring
- `live_win_probability_model.py` - In-game probability updates
- `live_model_backtester.py` - Backtest live strategies

**API Clients:**
- `odds_api_client.py` - Multi-source odds aggregation
- `kalshi_client.py` - Kalshi trading integration
- `kalshi_starter_clients.py` - Kalshi API helpers

**Utilities:**
- `comprehensive_analysis.py` - System-wide analysis

---

### **docs/** (All Documentation)
All markdown guides, planning documents, and text files.

**Quick Guides:**
- `QUICK_START.md` - Get started in 5 minutes
- `QUICK_REFERENCE.md` - Common commands and operations
- `README.md` - Full system documentation

**Implementation Guides:**
- `GOLD_STANDARD_IMPLEMENTATION.md` - PACE calculation & Four Factors
- `FEATURE_CALCULATOR_V5_GUIDE.md` - In-memory calculator documentation
- `DASHBOARD_V5_FEATURES.md` - Dashboard feature list
- `MANUAL_BET_ENTRY_GUIDE.md` - How to use manual bet entry
- `ODDS_API_GUIDE.md` - Odds API integration

**Analysis & Reports:**
- `COMPREHENSIVE_FUNCTIONALITY_REVIEW.md` - Full system analysis
- `FEATURE_MODEL_ANALYSIS_GUIDE.md` - Model feature importance
- `SYSTEM_STATUS.md` - Current system state

**Planning:**
- `Gold Standard NBA System Roadmap.txt` - Development roadmap

---

### **tests/** (Test & Verification Files)
All test scripts, schema checkers, and verification utilities.

**Test Scripts:**
- `test_feature_calculator.py` - Feature calculator tests
- `test_injury_stats.py` - Injury collector tests
- `test_leaguegamelog.py` - NBA API tests
- `test_collector_columns.py` - Data collector tests

**Verification:**
- `verify_pace_calculation.py` - PACE accuracy verification
- `verify_backtest_data.py` - Backtest data integrity

**Schema & Data Checks:**
- `check_schema.py` - Database schema inspector
- `check_db_status.py` - Database status checker
- `check_data.py` - Data completeness checker
- `check_nba_api_columns.py` - NBA API column verification

---

### **scripts/** (Utility Scripts)
One-time use scripts and maintenance utilities.

- `add_recent_seasons.py` - Add new seasons to database
- `add_pace_to_existing_data.py` - Retroactive PACE calculation
- `populate_game_results.py` - Populate game results table
- `Dependency Installation Script.py` - Install all dependencies

---

### **data_downloads/** (Data Download Scripts)
Scripts for downloading historical NBA data.

- `download_gold_standard_data.py` - Complete download (2015-2024, all features)
- `download_complete_data.py` - Quick download (4 recent seasons)

---

### **archived_versions/** (Old Versions)
Deprecated versions kept for reference. **Do not use in production.**

- `injury_data_collector.py` - V1 (replaced by V2)
- `nba_stats_collector_enhanced.py` - Enhanced version (replaced by V2)
- `feature_calculator.py` - Basic version (replaced by V5)
- `feature_calculator_enhanced.py` - Enhanced version (replaced by V5)
- `NBA_Dashboard_Gold_Standard_v4_1.py` - V4.1 (replaced by Enhanced V5)
- `download_historical_data_v2.py` - Old download script

---

### **logs/** (Log Files)
Application logs and verification results.

- `nba_betting_system.log` - Main application log
- `nba_system.log` - System events log
- `pace_verification_results.csv` - PACE verification results
- `backtest_logs/` - Backtest execution logs
- `prediction_logs/` - Prediction history

---

### **backups/** (Database Backups)
Database backups and snapshots.

- `nba_betting_data_backup_old_schema.db` - Pre-PACE schema backup

---

### **data/** (Processed Data)
ML training data and processed datasets.

- `master_training_data_v5.csv` - Feature-engineered training data

---

### **models/** (Trained Models)
Saved ML models and hyperparameters.

- `model_v5_ml.xgb` - Moneyline model
- `model_v5_ats.xgb` - Against-the-spread model
- `model_v5_total.xgb` - Over/under model
- `best_model_params_*.json` - Optimized hyperparameters

---

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
python scripts/Dependency Installation Script.py
```

### 2. **Download Data**
```bash
# Quick (4 recent seasons)
python data_downloads/download_complete_data.py

# Or comprehensive (10 seasons)
python data_downloads/download_gold_standard_data.py
```

### 3. **Launch Dashboard**
```bash
python main.py
```

---

## 📊 Key Features

✅ **31,338 game logs** with PACE calculations (10 seasons, 2015-2025)  
✅ **103 active injuries** tracked (CBS Sports + ESPN)  
✅ **15+ gold standard features** (Four Factors, PACE, SOS, ELO, etc.)  
✅ **In-memory feature calculator** (50-100x faster than SQL)  
✅ **Dual-source injury collector** (live + historical backfilling)  
✅ **6-tab enhanced dashboard** (Predictions, Performance, Bankroll, Admin, DB Explorer, Logs)  
✅ **1.81% PACE accuracy** vs NBA official stats  

---

## 🔧 Maintenance

### Update Injury Data (Daily)
```bash
python -c "from injury_data_collector_v2 import InjuryDataCollectorV2; c = InjuryDataCollectorV2(); c.scrape_live_injuries()"
```

### Add New Season Data
```bash
python scripts/add_recent_seasons.py
```

### Verify System Health
```bash
python health_check.py
```

### Check Database Status
```bash
python tests/check_db_status.py
```

---

## 📝 Version Notes

**Current Versions:**
- Stats Collector: **V2** (nba_api with PACE)
- Injury Collector: **V2** (dual-source with historical)
- Feature Calculator: **V5** (in-memory optimized)
- Dashboard: **Enhanced V5** (6 tabs, full featured)

**All old versions archived** - Clean, organized, production-ready! 🎉
