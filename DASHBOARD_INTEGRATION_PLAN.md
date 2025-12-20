# 🔧 MDP MODEL DASHBOARD INTEGRATION PLAN

## Current State Analysis

### Existing Dashboard (nba_gui_dashboard_v2.py)
- **Model**: Trial 1306 (classifier)
- **Features**: 22 GOLD_ELO features
- **Thresholds**: 2% fav / 10% dog
- **Architecture**: Binary classification

### Target MDP Configuration
- **Model**: XGBoost Regressor (nba_mdp_production_tuned.json)
- **Features**: 19 MDP features (VIF < 2.34)
- **Thresholds**: 1.5% fav / 8.0% dog
- **Architecture**: Regression → norm.cdf(margin / 13.42)

---

## Integration Steps

### Step 1: Configuration Migration
**File**: `nba_gui_dashboard_v2.py` (lines 40-80)

**Current**:
```python
from trial1306_config import (
    TRIAL1306_MODEL_PATH,
    TRIAL1306_FEATURES,
    TRIAL1306_FAVORITE_EDGE,
    TRIAL1306_UNDERDOG_EDGE
)
```

**Target**:
```python
from production_config_mdp import (
    MODEL_PATH,
    ACTIVE_FEATURES,
    MIN_EDGE_FAVORITE,
    MIN_EDGE_UNDERDOG,
    FILTER_MIN_OFF_ELO,
    NBA_STD_DEV
)
```

**Changes**:
- Import from `production_config_mdp` instead of `trial1306_config`
- Update edge thresholds: 1.5% / 8.0%
- Import NBA_STD_DEV (13.42) for probability conversion

---

### Step 2: Model Loading
**File**: `nba_gui_dashboard_v2.py` (model loading section)

**Current (Classifier)**:
```python
model = joblib.load(TRIAL1306_MODEL_PATH)
predictions = model.predict_proba(features)[:, 1]  # Binary classification
```

**Target (Regressor)**:
```python
import xgboost as xgb
from scipy.stats import norm

model = xgb.Booster()
model.load_model('models/nba_mdp_production_tuned.json')
dtest = xgb.DMatrix(features, feature_names=ACTIVE_FEATURES)
margins = model.predict(dtest)
probabilities = norm.cdf(margins / NBA_STD_DEV)  # Margin → Probability
```

**Changes**:
- Use xgb.Booster().load_model() for JSON models
- Create DMatrix with correct feature names
- Convert margins to probabilities using norm.cdf

---

### Step 3: Feature Calculator
**File**: `nba_gui_dashboard_v2.py` (feature calculation)

**Current**:
```python
from feature_calculator_v5 import FeatureCalculatorV5
features = calculator.calculate(TRIAL1306_FEATURES)  # 22 features
```

**Target**:
```python
from feature_calculator_v5 import FeatureCalculatorV5
features = calculator.calculate(ACTIVE_FEATURES)  # 19 features
```

**Verification Needed**:
- Confirm `feature_calculator_v5.py` can compute all 19 MDP features
- Check feature names match exactly: `off_elo_diff`, `def_elo_diff`, etc.
- Verify feature order matches training data

---

### Step 4: Edge Calculation
**File**: `nba_gui_dashboard_v2.py` (edge calculation logic)

**Current**:
```python
if is_favorite:
    edge_threshold = 0.02  # 2%
else:
    edge_threshold = 0.10  # 10%
```

**Target**:
```python
if is_favorite:
    edge_threshold = MIN_EDGE_FAVORITE  # 1.5%
else:
    edge_threshold = MIN_EDGE_UNDERDOG  # 8.0%
    
# Apply physics filter
if off_elo_diff < FILTER_MIN_OFF_ELO:  # -90
    # Skip bet (broken offense)
    continue
```

**Changes**:
- Update thresholds to 1.5% / 8.0%
- Add physics filter check
- No pricing filters (MAX_FAV_ODDS removed)

---

### Step 5: Remove Old References
**Files**: All Python files in project

**Search Patterns**:
```bash
grep -r "trial1306" .
grep -r "GOLD_ELO" .
grep -r "22_features" .
grep -r "classifier" .
```

**Replace With**:
- `trial1306` → `production_config_mdp`
- `GOLD_ELO` → `MDP`
- `22_features` → `19_features`
- References to classifier logic → regressor logic

---

### Step 6: Feature Names Mapping

| Old Feature (Trial 1306) | MDP Feature | Status |
|--------------------------|-------------|--------|
| off_elo_diff | off_elo_diff | ✅ Same |
| def_elo_diff | def_elo_diff | ✅ Same |
| home_composite_elo | home_composite_elo | ✅ Same |
| net_fatigue_score | net_fatigue_score | ✅ Same |
| injury_matchup_advantage | injury_matchup_advantage | ✅ Same |
| injury_shock_diff | injury_shock_diff | ✅ Same |
| star_power_leverage | star_power_leverage | ✅ Same |
| star_mismatch | star_mismatch | ✅ Same |
| season_progress | season_progress | ✅ Same |
| league_offensive_context | league_offensive_context | ✅ Same |
| projected_possession_margin | projected_possession_margin | ✅ Same |
| ewma_pace_diff | ewma_pace_diff | ✅ Same |
| ewma_efg_diff | ewma_efg_diff | ✅ Same |
| ewma_vol_3p_diff | ewma_vol_3p_diff | ✅ Same |
| three_point_matchup | three_point_matchup | ✅ Same |
| total_foul_environment | total_foul_environment | ✅ Same |
| net_free_throw_advantage | net_free_throw_advantage | ✅ Same |
| offense_vs_defense_matchup | offense_vs_defense_matchup | ✅ Same |
| pace_efficiency_interaction | pace_efficiency_interaction | ✅ Same |
| (other 3 features) | ❌ REMOVED | Dropped |

**Good news**: Most features are identical! Only need to remove 3 old features.

---

### Step 7: Dashboard Display Updates

**Prediction Tab**:
- Update model version label: "MDP v2.2 (Regression)"
- Show margin prediction alongside probability
- Display "Physics Filter" status (off_elo_diff check)

**Performance Tab**:
- Update expected ROI: 29.1% (from validation)
- Update threshold display: "1.5% Fav / 8.0% Dog"

**Settings Tab**:
- Add NBA_STD_DEV display (13.42)
- Show RMSE metric (13.42 points)
- Display MAE metric (11.06 points)

---

### Step 8: Service Integrations

**Injury Tracker** (`injury_replacement_model.py`):
- ✅ Already integrated in dashboard
- Verify it computes: `injury_matchup_advantage`, `injury_shock_diff`, `star_power_leverage`

**Kalshi Odds** (NEW - TO BE BUILT):
```python
# kalshi_odds_fetcher.py
def fetch_current_odds(games_list):
    """Fetch live odds from Kalshi API"""
    # Implementation needed
    pass
```

**ELO System** (`off_def_elo_system.py`):
- ✅ Already exists
- Verify it computes: `off_elo_diff`, `def_elo_diff`, `home_composite_elo`

---

## Testing Checklist

### Unit Tests
- [ ] Model loads correctly (JSON format)
- [ ] All 19 features compute without errors
- [ ] Margin → probability conversion matches training (13.42)
- [ ] Edge calculation uses correct thresholds (1.5% / 8.0%)
- [ ] Physics filter applies correctly (off_elo_diff >= -90)

### Integration Tests
- [ ] Dashboard launches without errors
- [ ] Predictions display for all games
- [ ] Feature values match expected ranges
- [ ] Edge filtering works (only shows 1.5%+ fav, 8%+ dog)
- [ ] No references to old models appear in UI

### End-to-End Tests
- [ ] Generate predictions for today's games
- [ ] Verify against manual calculation (1 game)
- [ ] Check bet recommendations match backtest logic
- [ ] Confirm bankroll management uses Quarter Kelly
- [ ] Validate commission adjustment (2%)

---

## Rollback Plan

If integration fails, revert to Trial 1306:
1. Restore `trial1306_config.py` imports
2. Reload old model: `models/trial1306_model.joblib`
3. Use 22 GOLD_ELO features
4. Revert thresholds: 2% / 10%

---

## Success Criteria

✅ Dashboard displays predictions using MDP model
✅ All 19 features compute correctly
✅ Edge thresholds match config (1.5% / 8.0%)
✅ No errors in logs
✅ Predictions match manual backtest results
✅ UI shows correct model version (v2.2)
✅ No references to old models remain

---

## File Modification List

### Must Edit:
1. `nba_gui_dashboard_v2.py` - Config imports, model loading, prediction logic
2. `feature_calculator_v5.py` - Verify 19-feature support (may be OK as-is)
3. `main_predict.py` - Update to use MDP model (if exists)

### Must Verify:
1. `injury_replacement_model.py` - Check feature compatibility
2. `off_def_elo_system.py` - Verify ELO calculations
3. `nba_stats_collector_v2.py` - Ensure data format matches

### Must Create:
1. `kalshi_odds_fetcher.py` - Live odds integration (TBD)

### Must Remove:
1. References to `trial1306_config`
2. References to `GOLD_ELO_22_features`
3. Classifier-specific logic
4. Old model paths

---

**Priority**: HIGH  
**Risk**: MEDIUM (classifier → regressor is significant change)  
**Timeline**: 2-3 hours with testing  
**Rollback Time**: 15 minutes if needed
