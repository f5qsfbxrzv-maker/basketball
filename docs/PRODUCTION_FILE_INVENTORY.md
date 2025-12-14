# Production File Inventory - NBA Betting Model v5.1
**Last Updated**: December 8, 2024  
**Model Status**: ✅ PRODUCTION (Dynamic Gravity Model)

---

## Core Production Files (DO NOT MODIFY WITHOUT TESTING)

### 1. Feature Calculation
**File**: `src/features/feature_calculator_v5.py`  
**Purpose**: Generates all 31 whitelisted features including Dynamic Gravity injury impact  
**Key Components**:
- `_normalize_player_name()` - Handles "Last, First" → "First Last" conversion
- `_calculate_dynamic_gravity_multiplier()` - Z-score based injury weighting (1.0x-4.5x)
- `_calculate_historical_injury_impact()` - Aggregates injury impacts per team
- `calculate_game_features()` - Main entry point for feature generation

**Last Modified**: Dec 8, 2025 7:49 PM  
**Status**: ✅ ACTIVE - Dynamic Gravity fully implemented

---

### 2. Model Training
**File**: `src/training/retrain_pruned_model.py`  
**Purpose**: Trains XGBoost model with 31-feature whitelist  
**Performance**: 68.37% accuracy, Log Loss 0.6239, Brier 0.2157  
**Outputs**:
- `models/xgboost_pruned_31features.pkl` - Trained model
- `output/feature_importance_pruned.csv` - Feature rankings

**Last Modified**: Dec 7, 2025 8:02 PM  
**Status**: ✅ ACTIVE - Latest training run successful

---

### 3. SHAP Analysis
**File**: `src/analysis/shap_analysis_dynamic_gravity.py`  
**Purpose**: Validates injury feature contributions via SHAP values  
**Key Metrics**:
- injury_impact_abs: Mean |SHAP| = 0.0708
- injury_impact_diff: Mean |SHAP| = 0.0559
- injury_elo_interaction: Mean |SHAP| = 0.0326

**Last Modified**: Dec 8, 2025 8:40 PM  
**Status**: ✅ ACTIVE - Validates Dynamic Gravity Model

---

### 4. Feature Configuration
**File**: `config/feature_whitelist.py`  
**Purpose**: Defines the 31 features used by model  
**Categories**:
- Mandatory Context (12): Injury, Rest, Fatigue, Altitude
- ELO Engine (3): Composite, Offensive, Defensive
- Foul Synergy (3): Home, Away, Total Environment
- EWMA Diffs (5): EFG, TOV, ORB, Pace, 3P Volume
- Key Absolutes (6): 3P%, TOV%, ORB, FTA Rate
- Chaos Metrics (2): Home Chaos, Net Chaos

**Last Modified**: Static configuration file  
**Status**: ✅ ACTIVE - Do not modify without retraining

---

## Model Artifacts

### Trained Model
**File**: `models/xgboost_pruned_31features.pkl`  
**Size**: ~2 MB  
**Algorithm**: XGBoost Classifier
**Hyperparameters**:
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8

**Performance**:
- Accuracy: 68.37%
- Log Loss: 0.6239
- Brier Score: 0.2157

---

### Feature Importance
**File**: `output/feature_importance_pruned.csv`  
**Top 10 Features**:
1. ewma_efg_diff (5.70%)
2. def_elo_diff (4.39%)
3. fatigue_mismatch (4.13%)
4. ewma_tov_diff (3.92%)
5. off_elo_diff (3.66%)
6. home_rest_days (3.55%)
7. total_foul_environment (3.54%)
8. altitude_game (3.52%)
9. away_rest_days (3.46%)
10. ewma_net_chaos (3.45%)

**Injury Features**: #19, #26, #27 (9.02% combined)

---

## Database Files

### Main Database
**File**: `data/live/nba_betting_data.db`  
**Tables**:
- `game_results` (3,224 games, 2023-2025)
- `historical_inactives` (57,085 injury records)
- `player_stats` (7,756 player-season records)
- `team_stats` (660 team-season records - DEPRECATED)
- `game_advanced_stats` (Newer, preferred over team_stats)

**Coverage**: 97.5% of training games have injury data

---

## Archived Files (Non-Production)

### Temporary Scripts
**Location**: `archive/temp_scripts_2024-12-08/`  
**Files**:
- `list_features.py` - Debugging script for feature display (replaced by proper analysis)
- `shap_dynamic_gravity.py` - Initial SHAP attempt (replaced by src/analysis version)

**Status**: ⚠️ ARCHIVED - Do not use in production

---

### Deprecated Analysis Scripts
**File**: `src/analysis/shap_injury_analysis.py`  
**Purpose**: Original SHAP analysis using MANUAL multipliers  
**Status**: ⚠️ DEPRECATED - Use `shap_analysis_dynamic_gravity.py` instead  
**Reason**: Uses hardcoded SUPERSTAR_MULTIPLIERS dict instead of Dynamic Gravity

---

## Legacy Feature Calculators

### Advanced Feature Calculator
**File**: `src/features/advanced_feature_calculator.py`  
**Last Modified**: Nov 25, 2025  
**Status**: ⚠️ LEGACY - Superseded by feature_calculator_v5.py  
**Note**: May contain experimental features not in production

### EWMA Feature Calculator
**File**: `src/features/ewma_feature_calculator.py`  
**Last Modified**: Nov 27, 2025  
**Status**: ⚠️ LEGACY - EWMA logic integrated into feature_calculator_v5.py  
**Note**: Standalone EWMA calculator, functionality now in v5

---

## Configuration Reference

### Dynamic Gravity Constants
```python
LEAGUE_AVG_PIE = 0.0855  # Calibrated from 7,714 players
LEAGUE_STD_PIE = 0.0230  # Standard deviation
```

### Injury Impact Formula
```python
base_impact = player_pie * 20.0
gravity_multiplier = calculate_dynamic_gravity_multiplier(player_pie)
final_impact = base_impact * gravity_multiplier
# Team total capped at 15.0
```

### Multiplier Ranges
- **Z ≤ 1.0**: 1.0x (average/role players)
- **1.0 < Z ≤ 2.5**: 1.0 to 3.0x (stars, aggressive ramp)
- **Z > 2.5**: 3.0 to 4.5x (MVPs, capped at 4.5x)

---

## Workflow Commands

### Retrain Model
```powershell
cd 'C:\Users\d76do\OneDrive\Documents\New Basketball Model'
python src/training/retrain_pruned_model.py
```

### Run SHAP Analysis
```powershell
cd 'C:\Users\d76do\OneDrive\Documents\New Basketball Model'
python src/analysis/shap_analysis_dynamic_gravity.py
```

### Display Feature List
```powershell
cd 'C:\Users\d76do\OneDrive\Documents\New Basketball Model'
python -c "import pandas as pd; df = pd.read_csv('output/feature_importance_pruned.csv'); print(df.to_string(index=False))"
```

---

## Version History

### v5.1 (Current - Dynamic Gravity)
- **Date**: December 8, 2024
- **Key Change**: Replaced manual SUPERSTAR_MULTIPLIERS with Z-score based Dynamic Gravity
- **Performance**: 68.37% accuracy (+0.77% vs v5.0)
- **Maintenance**: Zero hardcoded player names

### v5.0 (Manual Multipliers)
- **Date**: December 7, 2024
- **Key Change**: PIE-weighted injury impact with SHAP-calibrated multipliers
- **Performance**: 67.60% accuracy
- **Issue**: Required manual updates for 13+ players per season

### v4.x (Simple Count)
- **Date**: November 2024
- **Key Change**: Basic injury counting (0.5 per player)
- **Performance**: ~66% accuracy
- **Issue**: Didn't differentiate between stars and role players

---

## Maintenance Schedule

### Daily
- ✅ None (automated injury data ingestion)

### Weekly
- ✅ None

### Monthly
- ✅ None

### Seasonal
- ✅ None (Dynamic Gravity auto-adapts to player performance)

### Annual
- 🔍 Optional: Recalibrate PIE distribution if league-wide changes occur
- 🔍 Optional: Retrain model with latest season data

---

## Critical Warnings

### DO NOT:
❌ Modify `feature_calculator_v5.py` without running full test suite  
❌ Change feature whitelist without retraining model  
❌ Use `shap_injury_analysis.py` (uses deprecated manual multipliers)  
❌ Rely on `team_stats` table (deprecated, use `game_advanced_stats`)  
❌ Hardcode player names in injury calculations  

### DO:
✅ Use `shap_analysis_dynamic_gravity.py` for validation  
✅ Run `retrain_pruned_model.py` after any feature changes  
✅ Archive old scripts before creating new versions  
✅ Keep `DYNAMIC_GRAVITY_FINAL_REPORT.md` updated  
✅ Trust the Dynamic Gravity Model - it's empirically validated  

---

## Support & Documentation

### Main Documentation
- `DYNAMIC_GRAVITY_FINAL_REPORT.md` - Comprehensive validation report
- `config/feature_whitelist.py` - Feature definitions with comments
- `output/feature_importance_pruned.csv` - Current feature rankings

### Code Documentation
- `src/features/feature_calculator_v5.py` - Inline docstrings for all methods
- `src/training/retrain_pruned_model.py` - Training pipeline documentation
- `src/analysis/shap_analysis_dynamic_gravity.py` - SHAP validation logic

---

**Production Status**: ✅ STABLE  
**Last Validation**: December 8, 2024  
**Next Review**: As needed (no scheduled maintenance required)
