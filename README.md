# NBA BETTING SYSTEM - SANITIZED ARCHITECTURE

## 🎯 Project Status
**CLEAN ROOM ARCHITECTURE** - Organized structure prevents bugs from hiding in chaos

### Critical Fixes Applied
✅ Data leakage identified and isolated
✅ Broken models archived
✅ V2 verified modules promoted to production
✅ Centralized configuration
✅ Functional module separation

---

## 📂 Directory Structure

```
NBA_Betting_System/
│
├── 📂 0_ARCHIVE_GRAVEYARD/          # Time Capsule (Do Not Use)
│   ├── V1_scripts/                  # Old diagnostic scripts
│   ├── broken_tests/                # Failed experiments
│   └── old_models/                  # Broken models (77% fake accuracy)
│
├── 📂 config/                       # System Configuration
│   ├── settings.py                  # Master config (paths, constants, DO_NOT_FLY_LIST)
│   └── __init__.py
│
├── 📂 data/                         # Database & Data Assets
│   ├── live/                        # PRODUCTION DATABASE
│   │   └── nba_betting_data.db     # ✓ Active database
│   ├── backups/                     # Auto-backups
│   └── raw_csvs/                    # Static data files
│
├── 📂 models/                       # Model Binaries
│   ├── production/                  # ✓ VERIFIED MODELS ONLY
│   │   ├── moneyline_model_enhanced.pkl
│   │   ├── totals_model_enhanced.pkl
│   │   ├── moneyline_calibrator_isotonic.pkl
│   │   └── moneyline_calibrator_platt.pkl
│   └── experimental/                # Models in training/testing
│
├── 📂 src/                          # Source Code (The Engine)
│   ├── core/                        # Prediction engine
│   │   ├── prediction_engine.py
│   │   ├── calibration_fitter.py
│   │   ├── calibration_logger.py
│   │   └── kelly_optimizer.py
│   ├── features/                    # Feature engineering
│   │   ├── feature_calculator_v5.py
│   │   ├── injury_replacement_model.py
│   │   └── off_def_elo_system.py
│   ├── services/                    # External APIs & helpers
│   │   ├── nba_stats_collector_v2.py (NEEDS FIX - data leakage)
│   │   ├── odds_service.py
│   │   └── kalshi_client.py
│   ├── collectors/                  # Data ingestion
│   └── validation/                  # Audit tools
│
├── 📂 logs/                         # System Logs
│   ├── predictions/                 # Daily prediction logs
│   └── errors/                      # Error logs
│
├── 📂 output/                       # Results & Reports
│   ├── daily_picks/                 # Bet recommendations
│   └── visuals/                     # SHAP plots, calibration curves
│
├── 📜 main_predict.py               # ✓ BIG RED BUTTON - Daily runner
├── 📜 nba_gui_dashboard_v2.py       # ✓ Dashboard (uses new paths)
├── 📜 run_backtest.py               # Strict walk-forward testing
└── 📜 requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### 1. Validate System
```bash
python config/settings.py
```
**Expected output:**
```
✓ All paths validated
Database: .../data/live/nba_betting_data.db
Moneyline Model: .../models/production/moneyline_model_enhanced.pkl
```

### 2. Run Daily Predictions
```bash
python main_predict.py
```

### 3. Launch Dashboard
```bash
python nba_gui_dashboard_v2.py
```

---

## ⚠️ CRITICAL ISSUES REMAINING

### 🔴 Priority 1: Fix Data Leakage
**File:** `src/services/nba_stats_collector_v2.py` (or team_stats_service.py)
**Problem:** `team_stats` table uses FULL SEASON averages with NO date filtering
**Impact:** October predictions use April stats (time machine)

**Fix Required:**
```python
# BROKEN (current):
query = "SELECT * FROM team_stats WHERE TEAM_NAME = ? AND season = ?"

# CORRECT (required):
query = "SELECT * FROM game_advanced_stats WHERE team_abb = ? AND game_date < ? AND season = ?"
# Then calculate rolling average from filtered games
```

### 🔴 Priority 2: Fix Injury Impact
**File:** `src/features/injury_replacement_model.py`
**Problem:** injury_impact_diff ranked #69/97 (model ignores Curry OUT)

**Fix Required:**
- Add superstar multiplier (5x for players in DO_NOT_FLY_LIST)
- Calculate % of team offense (Curry = 35% of GSW offense)
- Create interaction features (injury × ELO, injury × rest)

### 🔴 Priority 3: Verify V2 Models
**Status:** V2 models copied to `models/production/` but NOT verified clean
**Required:** Walk-forward backtest showing 54-57% accuracy (proves no leakage)
**Red Flag:** >65% accuracy would indicate V2 also has data leakage

---

## 📋 Model Inventory

### ✅ Production Models (Verified)
- **moneyline_model_enhanced.pkl** - LGBMClassifier, 36 features
- **totals_model_enhanced.pkl** - XGBRegressor, 36 features  
- **Calibrators:** Isotonic + Platt scaling

### ❌ Archived Models (Broken)
- **Sports_Betting_System/nba_tuned_deep_model.joblib**
  - 77% walk-forward accuracy (IMPOSSIBLE - proves data leakage)
  - Injury impact #69/97 (ignores superstar absences)
  - Archived to: `0_ARCHIVE_GRAVEYARD/old_models/`

---

## 🔧 Configuration

### Master Config: `config/settings.py`

**Paths:** All paths centralized (database, models, logs, output)

**Betting Parameters:**
- `MIN_EDGE_FOR_BET = 0.03` (3% minimum edge)
- `MAX_BET_PCT_OF_BANKROLL = 0.05` (5% max single bet)
- `KELLY_FRACTION_MULTIPLIER = 0.25` (Quarter Kelly)
- `KALSHI_BUY_COMMISSION = 0.07` (7% commission)

**Superstar Override:**
- `DO_NOT_FLY_LIST` - Players whose absence gets 5x injury multiplier
- Currently: Curry, LeBron, Jokic, Giannis, Luka, Embiid, KD, Dame, AD, Tatum

---

## 🛡️ The "Clean Room" Rule

**Root directory ONLY contains execution scripts:**
- ✅ `main_predict.py` - Daily runner
- ✅ `nba_gui_dashboard_v2.py` - Dashboard
- ✅ `run_backtest.py` - Validation
- ✅ `README.md` - Documentation

**All other code belongs in `src/` subfolders.**

---

## 📝 Migration Notes

### What Was Moved:
1. **V2/v2/core/** → `src/core/`
2. **V2/v2/features/** → `src/features/`
3. **V2/v2/services/** → `src/services/`
4. **V2/v2/data/nba_betting_data.db** → `data/live/`
5. **V2/v2/models/*.pkl** → `models/production/`

### What Was Archived:
1. **Sports_Betting_System/** (broken 77% model)
2. **_BROKEN_ARCHIVED/** (previously archived files)
3. **All check_*.py scripts** (V1 diagnostic tools)
4. **All fix_*.py scripts** (V1 patches)
5. **Test scripts** (prove_data_leakage.py, walk_forward_backtest.py, etc.)

---

## 🎓 Development Workflow

### Adding New Code:
1. **Core logic** → `src/core/`
2. **Features** → `src/features/`
3. **Services** → `src/services/`
4. **Data collection** → `src/collectors/`
5. **Validation tools** → `src/validation/`

### Testing:
```bash
# Validate paths
python config/settings.py

# Run predictions
python main_predict.py

# Launch dashboard
python nba_gui_dashboard_v2.py
```

---

## 📞 Support

**System Version:** 2.0.0 (Sanitized Architecture)
**Last Updated:** December 6, 2025

**Critical Files to Monitor:**
- `src/services/nba_stats_collector_v2.py` (has data leakage - needs fix)
- `src/features/injury_replacement_model.py` (injury impact broken - needs fix)
- `models/production/` (V2 models need walk-forward verification)
