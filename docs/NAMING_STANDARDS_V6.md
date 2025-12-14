# NBA Betting System - Naming Standards & Version Control (v6)

## Overview
This document defines **mandatory naming conventions** for all files, modules, features, and data artifacts to prevent version conflicts and maintain consistency across the codebase.

**Current System Version**: v6.0  
**Feature Version**: v6  
**Model Version**: v6  
**Last Updated**: 2025-11-21

---

## Version Numbering System

### Format: `vX.Y.Z`
- **X (Major)**: Breaking changes, architecture overhauls (e.g., v5 → v6)
- **Y (Minor)**: New features, non-breaking enhancements (v6.0 → v6.1)
- **Z (Patch)**: Bug fixes, calibration updates (v6.1.0 → v6.1.1)

### Current Versions by Component

| Component | Version | File/Module | Status |
|-----------|---------|-------------|--------|
| **Main Dashboard** | v6.0 | `NBA_Dashboard_v6_Streamlined.py` | ✅ Production |
| **Admin Dashboard** | v6.0 | `admin_dashboard_v6.py` | ✅ Production |
| **Feature Calculator** | v6 | `core/feature_calculator_v6.py` | ✅ Production |
| **Stats Collector** | v6 | `core/nba_stats_collector_v6.py` | ✅ Production |
| **Injury Collector** | v6 | `core/injury_data_collector_v6.py` | ✅ Production |
| **ELO System** | v6 | `core/off_def_elo_system_v6.py` | ✅ Production |
| **Live Win Probability** | v6 | `core/live_win_probability_model_v6.py` | ✅ Production |
| **Prediction Engine** | v6 | `core/prediction_engine_v6.py` | ✅ Production |
| **Calibration** | v6 | `core/calibration_fitter_v6.py` | ✅ Production |
| **ATS Model** | v6 | `models/model_v6_ats.xgb` | ✅ Production |
| **Moneyline Model** | v6 | `models/model_v6_ml.xgb` | ✅ Production |
| **Total Model** | v6 | `models/model_v6_total.xgb` | ✅ Production |
| **Training Data** | v6 | `data/master_training_data_v6.csv` | ✅ Production |

---

## File Naming Rules

### 1. Python Modules (Core)
**Format**: `<module_name>_v<major>.py`

✅ **Correct**:
- `feature_calculator_v6.py`
- `prediction_engine_v6.py`
- `live_win_probability_model_v6.py`

❌ **Incorrect**:
- `feature_calculator.py` (no version)
- `feature_calculator_v5.py` (old version)
- `feature_calculatorV6.py` (wrong case)

### 2. Dashboards
**Format**: `<dashboard_type>_v<major>_<variant>.py`

✅ **Correct**:
- `NBA_Dashboard_v6_Streamlined.py` (main betting interface)
- `admin_dashboard_v6.py` (admin/diagnostics)

❌ **Incorrect**:
- `NBA_Dashboard.py`
- `dashboard_v6.py`
- `NBA_Dashboard_Enhanced_v5.py` (old version - archive)

### 3. Model Files
**Format**: `model_v<major>_<type>.xgb`

✅ **Correct**:
- `model_v6_ats.xgb`
- `model_v6_ml.xgb`
- `model_v6_total.xgb`

❌ **Incorrect**:
- `model_ats.xgb`
- `ats_model_v6.xgb`
- `model_v5_ats.xgb` (old version - archive)

### 4. Training Data
**Format**: `<dataset_name>_v<major>.csv`

✅ **Correct**:
- `master_training_data_v6.csv`
- `training_data_with_features_v6.csv`

❌ **Incorrect**:
- `training_data.csv`
- `master_training_data.csv`
- `master_training_data_v5.csv` (old version - archive)

### 5. Configuration Files
**Format**: `<config_type>_v<major>_<timestamp?>.json`

✅ **Correct**:
- `best_model_params_v6_classifier.json`
- `live_wp_runtime_params_v6.json`
- `best_live_wp_params_v6_20251121.json` (timestamped)

❌ **Incorrect**:
- `params.json`
- `config.json` (no version for version-specific configs)

### 6. Scripts
**Format**: `<script_name>.py` (no version for utility scripts)

✅ **Correct**:
- `retrain_pipeline.py`
- `prepare_training_data.py`
- `nightly_tasks.py`

**Exception**: Scripts that are version-specific should include version:
- `migrate_v5_to_v6.py`

### 7. Archive/Legacy Files
**Format**: Move to `archive/v<X>/` directory

✅ **Correct**:
```
archive/
  v5/
    NBA_Dashboard_Enhanced_v5.py
    feature_calculator_v5.py
    model_v5_ats.xgb
  v4/
    ...
```

---

## Import Statement Standards

### Always Use Explicit Version Imports

✅ **Correct**:
```python
from core.feature_calculator_v6 import FeatureCalculatorV6
from core.prediction_engine_v6 import PredictionEngineV6
from core.live_win_probability_model_v6 import LiveWinProbabilityModelV6
```

❌ **Incorrect**:
```python
from core.feature_calculator import FeatureCalculator  # No version
from core.feature_calculator_v5 import FeatureCalculatorV5  # Old version
```

### Class Naming Convention
**Format**: `<ClassName>V<major>`

✅ **Correct**:
```python
class FeatureCalculatorV6:
    VERSION = "6.0"
    
class PredictionEngineV6:
    VERSION = "6.0"
```

❌ **Incorrect**:
```python
class FeatureCalculator:  # No version
class FeatureCalculatorv6:  # Wrong case
```

---

## Database Schema Versioning

### Table Naming
**Format**: `<table_name>` (no version in table name, use schema_version column)

✅ **Correct**:
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    schema_version TEXT DEFAULT 'v6',
    ...
);
```

### Version Tracking Column
**Every table MUST include**:
```sql
schema_version TEXT DEFAULT 'v6',
created_at TEXT DEFAULT CURRENT_TIMESTAMP,
updated_at TEXT DEFAULT CURRENT_TIMESTAMP
```

---

## Constants and Configuration

### Version Constants (utils/constants.py)

```python
# System Version
SYSTEM_VERSION = "6.0"
FEATURE_VERSION = "v6"
MODEL_VERSION = "v6"
CALIBRATION_VERSION = "v6"

# Component Versions
DASHBOARD_VERSION = "v6.0"
ADMIN_DASHBOARD_VERSION = "v6.0"
LIVE_MODEL_VERSION = "v6.0"

# Compatibility
MIN_SUPPORTED_VERSION = "v6.0"
MAX_SUPPORTED_VERSION = "v6.9"
```

---

## Version Check Enforcement

### Startup Version Validation

Every main module MUST validate versions on startup:

```python
def validate_versions():
    """Ensure all components use compatible versions."""
    from utils.constants import SYSTEM_VERSION, MODEL_VERSION
    import json
    
    # Check model manifest
    with open('models/manifest_v6.json') as f:
        manifest = json.load(f)
        if manifest.get('system_version') != SYSTEM_VERSION:
            raise ValueError(f"Model version mismatch: {manifest.get('system_version')} != {SYSTEM_VERSION}")
    
    # Check feature calculator
    from core.feature_calculator_v6 import FeatureCalculatorV6
    if FeatureCalculatorV6.VERSION != MODEL_VERSION:
        raise ValueError(f"Feature calculator version mismatch")
    
    print(f"✅ Version validation passed: System v{SYSTEM_VERSION}")

# Run at module load
validate_versions()
```

### Runtime Version Logging

```python
# In every prediction/operation, log version metadata
prediction_metadata = {
    'timestamp': datetime.now().isoformat(),
    'system_version': SYSTEM_VERSION,
    'model_version': MODEL_VERSION,
    'feature_version': FEATURE_VERSION,
    'calibration_version': calibrator.VERSION
}
```

---

## Migration Protocol

### Upgrading from v5 to v6

**Step 1: Archive Old Versions**
```powershell
# Create archive directory
New-Item -ItemType Directory -Force archive/v5

# Move old files
Move-Item NBA_Dashboard_Enhanced_v5.py archive/v5/
Move-Item core/feature_calculator_v5.py archive/v5/
Move-Item models/model_v5_*.xgb archive/v5/
Move-Item data/master_training_data_v5.csv archive/v5/
```

**Step 2: Rename All Files**
```powershell
# Rename core modules (use batch script or manual)
Rename-Item core/feature_calculator.py feature_calculator_v6.py
Rename-Item core/prediction_engine.py prediction_engine_v6.py
# ... etc
```

**Step 3: Update All Imports**
```python
# Run migration script
python scripts/migrate_imports_v5_to_v6.py
```

**Step 4: Update Version Constants**
```python
# Edit utils/constants.py
SYSTEM_VERSION = "6.0"
MODEL_VERSION = "v6"
```

**Step 5: Retrain Models**
```powershell
python scripts/retrain_pipeline.py --version v6
```

**Step 6: Validation**
```powershell
python scripts/validate_versions.py
```

---

## Prohibited Patterns

### ❌ NEVER Use These

1. **Generic Names Without Versions**:
   - `model.xgb`
   - `feature_calculator.py`
   - `dashboard.py`

2. **Inconsistent Version Formats**:
   - `calculator_v6.0.py` (too specific for filename)
   - `calculatorV6.py` (wrong case)
   - `calculator_version_6.py` (too verbose)

3. **Mixed Versions in Same Directory**:
   ```
   ❌ core/
       feature_calculator_v5.py
       feature_calculator_v6.py  # Archive v5 first!
   ```

4. **Hardcoded Paths Without Version**:
   ```python
   ❌ model = joblib.load('models/model_ats.xgb')
   ✅ model = joblib.load(f'models/model_{MODEL_VERSION}_ats.xgb')
   ```

---

## Documentation Standards

### Version References in Docs

All documentation MUST include version in title and metadata:

```markdown
# Feature Calculator v6.0 Documentation

**System Version**: v6.0  
**Last Updated**: 2025-11-21  
**Compatible With**: Dashboard v6.0+

## Overview
This is the v6 feature calculator...
```

### Code Comments
```python
# Feature Calculator v6 - 120+ features with PBP integration
# Compatible with: Model v6, Dashboard v6
# Last updated: 2025-11-21
class FeatureCalculatorV6:
    """
    v6 Feature Calculator
    
    Versions:
    - v6.0 (2025-11-21): Initial v6 release with PBP features
    - v5.0 (2024-xx-xx): Legacy version (archived)
    """
```

---

## Automated Enforcement

### Pre-commit Hook (`.git/hooks/pre-commit`)

```bash
#!/bin/bash
# Version naming enforcement

echo "Checking version naming compliance..."

# Check for unversioned core files
if git diff --cached --name-only | grep -E 'core/.*\.py$' | grep -v '_v[0-9]'; then
    echo "❌ ERROR: Core module without version detected"
    echo "All core modules MUST include version: <name>_v6.py"
    exit 1
fi

# Check for mixed versions
v5_files=$(find . -name '*_v5.py' -not -path './archive/*')
if [ ! -z "$v5_files" ]; then
    echo "❌ ERROR: v5 files found outside archive/"
    echo "$v5_files"
    echo "Move old versions to archive/v5/"
    exit 1
fi

echo "✅ Version naming checks passed"
```

---

## Version Manifest (models/manifest_v6.json)

```json
{
  "system_version": "6.0",
  "generated_at": "2025-11-21T12:00:00Z",
  "schema_version": 2,
  "models": [
    {
      "model_id": "basket_ats_model_v6",
      "version": "6.0",
      "feature_version": "v6",
      "filename": "model_v6_ats.xgb",
      "status": "production",
      "trained_at": "2025-11-21T10:00:00Z",
      "metrics": {
        "brier": 0.185,
        "accuracy": 0.557
      }
    },
    {
      "model_id": "basket_ml_model_v6",
      "version": "6.0",
      "feature_version": "v6",
      "filename": "model_v6_ml.xgb",
      "status": "production"
    },
    {
      "model_id": "basket_total_model_v6",
      "version": "6.0",
      "feature_version": "v6",
      "filename": "model_v6_total.xgb",
      "status": "production"
    }
  ],
  "dependencies": {
    "feature_calculator": "v6",
    "calibration": "v6",
    "elo_system": "v6"
  }
}
```

---

## FAQ

### Q: When should I increment the version?
**A**: 
- **Major (v5 → v6)**: Architecture changes, breaking API changes, new feature sets
- **Minor (v6.0 → v6.1)**: New non-breaking features, enhancements
- **Patch (v6.1.0 → v6.1.1)**: Bug fixes, calibration updates

### Q: Can I have multiple versions in production?
**A**: NO. Only ONE version should be in production at a time. Old versions go to `archive/`.

### Q: What about utility scripts?
**A**: Utility scripts (retrain, migrate, etc.) don't need versions unless they're version-specific migration tools.

### Q: How do I test a new version before deployment?
**A**: Use a separate testing environment or branch:
```
feature/v7-testing/
  core/feature_calculator_v7.py
  models/model_v7_ats.xgb
```

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-21 | v6.0 | Initial v6 naming standards document |
| 2025-11-21 | v6.0 | Implemented PBP features, unified calibration |
| 2025-11-20 | v5.0 | Legacy version (archived) |

---

## Enforcement Checklist

Before committing ANY code, verify:

- [ ] All core modules include `_v6` suffix
- [ ] All model files include `_v6` in name
- [ ] All imports use explicit version (e.g., `from core.x_v6 import`)
- [ ] Class names include version suffix (e.g., `ClassV6`)
- [ ] Version constants updated in `utils/constants.py`
- [ ] Manifest file (`models/manifest_v6.json`) reflects correct versions
- [ ] Old versions moved to `archive/v5/`
- [ ] Documentation includes version references
- [ ] No hardcoded paths without version variables

---

**REMEMBER**: Version consistency prevents bugs, confusion, and wasted development time. When in doubt, include the version!
