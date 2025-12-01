# 🎯 PROFESSIONAL SYSTEM - COMPLETE SETUP GUIDE

## ✅ WHAT YOU JUST BUILT

You now have a **production-grade** Sports Betting System with:

```
Sports_Betting_System/
├── src/                    # 18 GOLDEN MASTER FILES (no duplicates!)
│   ├── ingestion/          # nba_stats_collector.py, kalshi_client.py
│   ├── processing/         # feature_calculator.py (85 features!)
│   ├── training/           # train_nba_model.py
│   ├── prediction/         # prediction_engine.py
│   └── core/               # calibration_fitter.py, kelly_optimizer.py
│
├── data/
│   ├── processed/          # training_data_final.csv (12,188 rows, 85 features)
│   ├── raw/                # moneyline, spread, totals CSVs
│   └── database/           # 16 .db files (calibration, ELO history)
│
├── models/
│   └── production/         # best_params_moneyline.json
│
└── config.py, requirements.txt, .gitignore
```

---

## 📋 IMMEDIATE NEXT STEPS

### 1. **Initialize Git (THE GOLDEN RULE)**
```bash
cd "C:/Users/d76do/OneDrive/Documents/New Basketball Model/Sports_Betting_System"
git init
git add .
git commit -m "Initial professional structure - V2 Golden Masters migrated"
```

**Why Git?**
- NO MORE `model_v2.py`, `model_v3.py`, `model_FINAL_REAL.py`
- Just ONE file: `model.py`
- Use branches for experiments: `git checkout -b experiment/new-feature`
- If it works: merge. If it fails: discard with `git checkout main`

### 2. **Fix Import Paths**

All migrated files still have old V2 imports. You need to update them:

**OLD (WRONG):**
```python
from V2.v2.core.prediction_engine import PredictionEngine
from V2.v2.features.feature_calculator_v5 import FeatureCalculatorV5
```

**NEW (CORRECT):**
```python
from src.core.prediction_engine import PredictionEngine
from src.processing.feature_calculator import FeatureCalculatorV5
```

**Find & Replace:**
```bash
# In Sports_Betting_System directory
# Use your IDE's "Find in Files" and replace:
V2.v2.core     → src.core
V2.v2.features → src.processing
V2.v2.models   → src.training
V2.v2.services → src.ingestion
```

### 3. **Fix Data Paths**

**OLD (WRONG):**
```python
df = pd.read_csv('V2/training_data/training_data_final.csv')
df = pd.read_csv('../../V2/data/raw_odds_ehallmar/moneyline_history_2023_24.csv')
```

**NEW (CORRECT):**
```python
df = pd.read_csv('data/processed/training_data_final.csv')
df = pd.read_csv('data/raw/moneyline_2023_24.csv')
```

### 4. **Test the System**

```bash
# Navigate to Sports_Betting_System
cd Sports_Betting_System

# Test feature calculation
python -c "from src.processing.feature_calculator import FeatureCalculatorV5; print('✅ Imports work')"

# Test prediction engine
python -c "from src.prediction.prediction_engine import PredictionEngine; print('✅ Imports work')"
```

### 5. **Quarantine the Old Chaos**

```bash
# Go back to parent directory
cd ..

# Create quarantine folder
mkdir _OLD_CHAOS

# Move EVERYTHING except Sports_Betting_System
# DO THIS CAREFULLY - keep Sports_Betting_System folder!
```

**What goes to _OLD_CHAOS:**
- V2/ folder (original chaotic structure)
- All test_*.py files (already in Sports_Betting_System/notebooks/archive/)
- All *_v2.py, *_v3.py versioned files
- All check_*.py, debug_*.py scripts
- archive/ folder
- All the cleanup scripts we just created

**What stays:**
- `Sports_Betting_System/` (your new professional system)

---

## 🚨 CRITICAL RULES GOING FORWARD

### 1. **NO CODE IN `data/`**
- `data/raw/` = read-only original CSVs
- `data/processed/` = cleaned data ready for models
- `data/database/` = SQLite databases
- NO Python scripts allowed here!

### 2. **NO DATA IN `src/`**
- `src/` contains ONLY Python code
- No CSVs, no databases, no model artifacts
- Use relative paths to read from `data/`

### 3. **ONE VERSION PER FILE**
- Use Git for versions, NOT filenames
- Create branches for experiments: `git checkout -b test/new-elo`
- Merge if successful: `git merge test/new-elo`
- Discard if failed: `git branch -D test/new-elo`

### 4. **Production Model Deployment**
- Models go to `models/staging/` first
- Run backtests on staging models
- Only promote to `models/production/` if backtest passes
- Archive old production models to `models/archive/`

### 5. **Daily Workflow**
```python
# Morning: Get today's games
python src/ingestion/nba_stats_collector.py

# Process features
python src/processing/feature_calculator.py

# Generate predictions
python src/prediction/prediction_engine.py

# Output: logs/predictions_2024-11-30.csv
```

---

## 📂 DIRECTORY PURPOSE GUIDE

```
Sports_Betting_System/
│
├── data/
│   ├── raw/              # Original CSVs - NEVER EDIT, READ-ONLY
│   ├── processed/        # Cleaned, feature-engineered data
│   └── database/         # SQLite .db files (calibration, ELO)
│
├── models/
│   ├── production/       # CURRENT LIVE MODELS (making money today)
│   ├── staging/          # Models being tested (not live yet)
│   └── archive/          # Old models (don't delete, move here)
│
├── notebooks/            # Jupyter for messy experiments
│   ├── nba_experiments/  # Scratchpad work
│   └── archive/          # Old notebooks
│
├── src/                  # SOURCE CODE - The Engine Room
│   ├── ingestion/        # Fetch data from NBA API, Kalshi, etc.
│   ├── processing/       # Clean data, calculate features
│   ├── training/         # Build/train models
│   ├── backtesting/      # Test models on historical data
│   └── prediction/       # Generate today's picks
│
├── logs/                 # Execution logs, predictions, errors
│
├── config.py             # Global settings (paths, API keys, params)
├── requirements.txt      # Python dependencies
├── .gitignore            # Don't commit data/models to git
└── README.md             # How to run the system
```

---

## 🎯 YOUR CANONICAL FILES (DO NOT DELETE)

### Data (The Golden Datasets)
- `data/processed/training_data_final.csv` - 85 features, 12,188 rows ✅
- `data/raw/moneyline_2023_24.csv` - Historical odds
- `data/raw/moneyline_2024_25.csv` - Current season odds

### Code (The Golden Masters)
- `src/processing/feature_calculator.py` - Full 85-feature engineering
- `src/processing/elo_system.py` - Injury-aware off/def ELO
- `src/prediction/prediction_engine.py` - Complete prediction pipeline
- `src/core/calibration_fitter.py` - Isotonic/Platt calibration
- `src/core/kelly_optimizer.py` - Kelly criterion bet sizing

### Models
- `models/production/best_params_moneyline.json` - Tuned hyperparameters

---

## ⚠️ WHAT YOU LEFT BEHIND (In _OLD_CHAOS)

### Corrupted/Leaky Files (NEVER USE AGAIN)
- ❌ `training_data_final_modern.csv` (PLUS_MINUS, PTS leaks)
- ❌ `training_data_modern_raw.csv` (also leaky)
- ❌ `run_pipeline_locally.py` (creates the leaky files)

### Duplicate Scripts (18+ scripts per purpose)
- 32 duplicate ingestion scripts
- 38 duplicate processing scripts
- 34 duplicate training scripts
- 70 versioned files (*_v2.py, *_v3.py, etc.)

### Debug/Check Scripts (27 files)
- check_*.py, debug_*.py, diagnose_*.py
- These are now in notebooks/archive/ if you need them

---

## 🔧 FIXING COMMON ISSUES

### Issue: "ModuleNotFoundError: No module named 'V2'"
**Fix:** Update imports from `V2.v2.core` → `src.core`

### Issue: "FileNotFoundError: V2/training_data/training_data_final.csv"
**Fix:** Update path to `data/processed/training_data_final.csv`

### Issue: "ImportError: cannot import name 'FeatureCalculatorV5'"
**Fix:** The class name stays the same, just the import path changes:
```python
from src.processing.feature_calculator import FeatureCalculatorV5  # Correct
```

### Issue: Script can't find constants.py
**Fix:** Add parent directory to Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import MIN_EDGE, KELLY_FRACTION
```

---

## 📊 VERIFICATION CHECKLIST

Before declaring success, verify:

- [ ] Git initialized in Sports_Betting_System/
- [ ] All imports changed from `V2.v2.*` to `src.*`
- [ ] All data paths point to `data/processed/` or `data/raw/`
- [ ] Can run: `python -c "from src.prediction.prediction_engine import PredictionEngine"`
- [ ] `data/processed/training_data_final.csv` exists (85 features, 12,188 rows)
- [ ] NO `training_data_final_modern.csv` in new system (leaky!)
- [ ] Old project moved to `_OLD_CHAOS/`

---

## 🚀 DEPLOYMENT WORKFLOW

### Day 1: Fix Imports & Test
1. Update all imports in migrated files
2. Update all data paths
3. Run basic import tests
4. Verify feature_calculator can load training data

### Day 2: Backtest on Clean System
1. Create `src/backtesting/backtest_pipeline.py`
2. Run backtest on `data/processed/training_data_final.csv`
3. Verify realistic performance (not 99% win rate!)
4. Document baseline metrics

### Day 3: Live Prediction Test
1. Fetch today's games via `src/ingestion/nba_stats_collector.py`
2. Calculate features via `src/processing/feature_calculator.py`
3. Generate predictions via `src/prediction/prediction_engine.py`
4. Review output in `logs/predictions_YYYYMMDD.csv`

### Day 4+: Production
1. Automate daily pipeline
2. Monitor calibration drift
3. Retrain models monthly
4. Use Git branches for experiments

---

## 💪 YOU NOW HAVE

✅ **Separation of Concerns**: Data ≠ Code ≠ Models
✅ **Single Source of Truth**: ONE file per purpose
✅ **Version Control Ready**: Git instead of file renaming
✅ **Production Deployment Path**: staging → backtest → production
✅ **No More Leaky Data**: Canonical dataset with 85 features
✅ **No More Chaos**: 18 Golden Masters vs 619 duplicates

---

**The ecosystem is no longer perverted. It's professional.**

**NO MORE `model_v2_final_REAL.py`**
