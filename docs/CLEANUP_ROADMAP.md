# PROJECT CLEANUP ROADMAP
**Generated: 2024-11-30**
**Current State: 1,106 files, 1,181.7 MB**

---

## 🎯 PRIMARY ISSUES

### 1. **Duplicate Training Datasets**
You have **38 training data CSV files** (113.1 MB) when you only need **3 canonical datasets**:
- `V2/training_data/training_data_final.csv` (85 features, 12,188 rows) ✅ **USE THIS**
- `data/training_data_with_features.csv` (12,205 rows) ✅ **KEEP**
- `data/master_training_data_v6.csv` (outcomes only) ✅ **KEEP**

**Problem**: `run_pipeline_locally.py` keeps creating **leaky datasets** in `V2/data/training_data_final_modern.csv` with PLUS_MINUS, PTS, per-game Four Factors.

### 2. **File Version Chaos**
- **70 versioned files** (_v2, _v3, _v4, _v5, _v6) scattered everywhere
- No clear indication which version is current
- Old versions not archived

### 3. **226 Files in Deprecated Locations**
- `archive/` folder: 150+ files
- `V2/_archive/` folder
- `tests/` now has files from archive (just moved 57 test files there)
- Many backup/temp files in root

### 4. **66 Duplicate Filenames**
Same filename in multiple locations - causes confusion about which is "live"

---

## 📋 CLEANUP PHASES

### **PHASE 1: PROTECT CANONICAL FILES** (Do First!)

Create a "DO NOT DELETE" list in `V2/CANONICAL_FILES.txt`:

```
# CANONICAL DATASETS - DO NOT DELETE
V2/training_data/training_data_final.csv
data/training_data_with_features.csv  
data/master_training_data_v6.csv

# CORE V2 SYSTEM
V2/v2/core/prediction_engine.py
V2/v2/core/calibration_fitter.py
V2/v2/core/calibration_logger.py
V2/v2/core/kelly_optimizer.py
V2/v2/features/feature_calculator_v5.py
V2/v2/features/off_def_elo_system.py
V2/v2/features/injury_replacement_model.py
V2/v2/models/ml_model_trainer.py

# PRODUCTION MODELS
V2/models/tuned/best_params_moneyline_heavy.json
V2/models/trained/*.joblib (production models)

# DATABASES
*.db (all SQLite databases)

# CONFIGURATION
config.json
constants.py
requirements.txt
```

### **PHASE 2: DELETE CORRUPTED/LEAKY FILES** (High Priority)

**Action**: Delete these immediately - they contain data leaks:

```bash
# Leaky datasets from run_pipeline_locally.py
V2/data/training_data_final_modern.csv  # DELETE - has PLUS_MINUS, PTS leaks
V2/data/training_data_modern_raw.csv    # DELETE - also leaky
```

**Keep only**:
- `V2/data/training_data_final_modern_CLEAN.csv` (sanitized, 14 columns)
- `V2/data/training_data_final_modern_RAW_BACKUP.csv` (backup reference only)

### **PHASE 3: ARCHIVE OLD VERSIONS** (Medium Priority)

Create `_ARCHIVE_2024/` folder and move:

**Training data versions**:
```
V2/data/training_data_with_moneylines_2023_24*.csv (all versions)
V2/data/training_data_honest_elo*.csv
V2/data/training_data_sanitized_scorched.csv
V2/data/training_data_scorched_output.csv
V2/training_data/training_data_enhanced*.csv (check if duplicate of final.csv first)
```

**Versioned Python scripts**:
```
*_v2.py
*_v3.py
*_v4.py
*_v5.py
*_v6.py
(unless they're the CURRENT version - check git/dates)
```

**Old dashboard versions**:
```
admin_dashboard.py (keep admin_dashboard_v6.py)
NBA_Dashboard_v6_Streamlined.py (if superseded by main_dashboard.py)
```

### **PHASE 4: CONSOLIDATE SCRIPTS** (Medium Priority)

**Root directory has 27 debug/check scripts** - move to `scripts/debug/`:
```
check_*.py → scripts/debug/
debug_*.py → scripts/debug/
test_*.py → tests/ (already did this for 57 files)
tmp_*.py → DELETE or scripts/debug/
diagnose_*.py → scripts/debug/
```

**V2 folder has scattered test files** - already moved most to `tests/`

### **PHASE 5: REMOVE DUPLICATE CONFTEST/INIT FILES**

**Issue**: Multiple `conftest.py` and `__init__.py` in different locations

**Action**:
- Keep: `tests/conftest.py`, `V2/v2/tests/conftest.py`
- Review all `__init__.py` - many might be empty or redundant

### **PHASE 6: CONSOLIDATE DOCUMENTATION** (Low Priority)

Multiple markdown files with similar info:
```
BULLETPROOF_DASHBOARD.md
DASHBOARD_AUDIT_FIXES.md
DUAL_DASHBOARD_GUIDE.md
CALIBRATION_WORKFLOW_GUIDE.md
LIVE_MODEL_CALIBRATION_GUIDE.md
NAMING_STANDARDS_V6.md
FEATURE_IMPLEMENTATION_SUMMARY.md
FEATURE_STATUS_REPORT.md
IMPLEMENTATION_SUMMARY.md
PHASE_2_IMPLEMENTATION_SUMMARY.md
FOUR_FACTORS_VERIFICATION_REPORT.md
```

**Action**: Create single `docs/` folder, consolidate into:
- `docs/ARCHITECTURE.md` (system overview)
- `docs/CALIBRATION_GUIDE.md` (calibration workflow)
- `docs/FEATURE_ENGINEERING.md` (feature documentation)
- `docs/CHANGELOG.md` (implementation history)
- Move rest to `docs/archive/`

---

## 🚨 CRITICAL: PREVENT REVERSION TO CORRUPTED FILES

### **Root Cause**: 
You keep reverting to `V2/data/training_data_final_modern.csv` because:
1. Script `V2/scripts/run_pipeline_locally.py` creates it
2. Other scripts reference it by default
3. No clear indication it's corrupted

### **Solution**:

**1. Rename the leaky file permanently:**
```bash
mv V2/data/training_data_final_modern.csv V2/data/LEAKY_DO_NOT_USE_training_data_final_modern.csv
```

**2. Update all script references to point to canonical dataset:**
```python
# OLD (WRONG):
df = pd.read_csv('V2/data/training_data_final_modern.csv')

# NEW (CORRECT):
df = pd.read_csv('V2/training_data/training_data_final.csv')
```

**3. Add validation check to scripts:**
```python
def load_training_data():
    """Load training data with leak detection"""
    path = 'V2/training_data/training_data_final.csv'
    df = pd.read_csv(path)
    
    # Verify no leaks
    leak_columns = ['PLUS_MINUS', 'PTS_home', 'PTS_away', 'WL']
    found_leaks = [col for col in leak_columns if col in df.columns]
    
    if found_leaks:
        raise ValueError(f"LEAK DETECTED in {path}: {found_leaks}")
    
    print(f"✅ Loaded clean dataset: {len(df)} rows, {len(df.columns)} features")
    return df
```

---

## 📊 EXECUTION CHECKLIST

### **IMMEDIATE (Today)**
- [ ] Create `V2/CANONICAL_FILES.txt` with protected file list
- [ ] Rename leaky dataset: `training_data_final_modern.csv` → `LEAKY_DO_NOT_USE_...`
- [ ] Delete `training_data_modern_raw.csv` (also leaky)
- [ ] Run git status to see uncommitted changes (if using git)

### **THIS WEEK**
- [ ] Create `_ARCHIVE_2024/` folder
- [ ] Move all versioned files (_v2, _v3, etc.) to archive
- [ ] Move deprecated training CSVs to archive
- [ ] Consolidate debug scripts to `scripts/debug/`
- [ ] Create `docs/` folder and consolidate markdown files

### **NEXT WEEK**
- [ ] Audit all Python scripts for references to corrupted datasets
- [ ] Update scripts to use canonical `training_data_final.csv`
- [ ] Add leak detection validation to data loading functions
- [ ] Remove duplicate files (use CLEANUP_ROADMAP.csv)
- [ ] Update `.gitignore` to prevent committing temp files

### **ONGOING**
- [ ] Use version control (git) to track changes
- [ ] Adopt naming convention: `feature_v2.py` → `feature.py` + git tags
- [ ] Document which files are "production" vs "experimental"
- [ ] Regularly run cleanup analysis to catch new duplicates

---

## 🎯 SUCCESS METRICS

**Before**: 1,106 files, 1,181.7 MB, constant file confusion
**Target**: ~400 active files, ~1,000 MB, clear structure

**Key Improvements**:
1. ✅ Single source of truth for training data
2. ✅ No more leaky datasets
3. ✅ Clear separation: production code vs experiments vs archive
4. ✅ All tests in `tests/` folder
5. ✅ All docs in `docs/` folder
6. ✅ Versioning via git, not filenames

---

## 📁 RECOMMENDED FINAL STRUCTURE

```
New Basketball Model/
├── V2/
│   ├── v2/                          # Core V2 system (CANONICAL)
│   │   ├── core/                    # prediction_engine, calibration, kelly
│   │   ├── features/                # feature_calculator_v5, elo, injury
│   │   ├── models/                  # ml_model_trainer, advanced_models
│   │   ├── services/                # kalshi_client, nba_stats_collector
│   │   └── dashboard/               # main_dashboard, tabs
│   ├── training_data/               # CANONICAL DATASETS
│   │   └── training_data_final.csv  # 85 features, 12,188 rows ✅
│   ├── models/
│   │   ├── tuned/                   # best_params*.json
│   │   └── trained/                 # *.joblib production models
│   ├── data/
│   │   └── raw_odds_ehallmar/       # Source odds data
│   └── scripts/                     # Automation, utilities
├── data/
│   ├── training_data_with_features.csv  # Alternative full dataset
│   └── master_training_data_v6.csv      # Outcomes
├── tests/                           # All test files (done ✅)
├── docs/                            # All documentation
├── scripts/
│   └── debug/                       # Debug/check utilities
├── _ARCHIVE_2024/                   # Old versions, deprecated files
├── config.json
├── constants.py
└── requirements.txt
```

---

## ⚠️ WARNINGS

**DO NOT DELETE** without checking:
- Any `.db` file (SQLite databases with calibration history, ELO data)
- Any `*.joblib` or `*.pkl` file (trained models)
- Files in `V2/v2/` (core system)
- Files in `V2/models/tuned/` (hyperparameter results)

**SAFE TO DELETE** immediately:
- `*_v2.py`, `*_v3.py` (if not current version)
- Files in `archive/` (already archived)
- `tmp_*.py`, `temp_*.py`
- Duplicate CSVs identified in CLEANUP_ROADMAP.csv as "DELETE"

---

## 🔧 AUTOMATION SCRIPTS NEEDED

### 1. **Leak Detection Script**
```python
# scripts/detect_leaks.py
# Scans all CSV files for leak columns, flags suspicious files
```

### 2. **Canonical File Validator**
```python
# scripts/validate_canonical.py  
# Ensures canonical files exist and haven't been corrupted
```

### 3. **Reference Updater**
```python
# scripts/update_references.py
# Finds all references to old datasets, suggests replacements
```

---

**Next Step**: Review `V2/CLEANUP_ROADMAP.csv` for detailed file-by-file actions.
