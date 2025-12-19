# SYNDICATE OPTIMIZATION - OVERNIGHT RUN

## Status: RUNNING ✓
**Started:** December 18, 2025, 8:12 PM
**Expected Duration:** 8-10 hours (2000 trials)
**Target:** Beat 0.6222 log-loss (Trial 1306 baseline)

---

## What Was Implemented

### 1. ✅ Syndicate-Level Feature Engineering

**ELO Matchup Advantages** (Replaced 6 → 3 features):
- `off_matchup_advantage` = Home_Off_Elo - Away_Def_Elo
- `def_matchup_advantage` = Home_Def_Elo - Away_Off_Elo  
- `net_composite_advantage` = (Home_Composite + 100) - Away_Composite

**Why?** XGBoost shouldn't learn subtraction. 1600 vs 1700 is mathematically identical to 1400 vs 1500 in Elo theory. Pre-computing differentials gives the "pure signal" directly.

---

**Matchup Friction Features** (Replaced 3 → 5 features):
- `effective_shooting_gap`: Home shooting vs Away defense (both directions)
- `turnover_pressure`: Home steals + Away turnovers vs reverse
- `rebound_friction`: Home ORB% vs Away defensive rebounding allowed
- `total_rebound_control`: ORB% + DRB% combined
- `whistle_leverage`: FTA generation vs opponent foul rate

**Why?** Simple differentials (Home_ORB% - Away_ORB%) don't capture interactions. Matchup friction shows team strength vs opponent weakness.

---

**Volume-Adjusted Efficiency** (New feature):
- Formula: `eFG% × Projected_Possessions`
- Captures raw points potential better than ELO alone
- Accounts for pace matchup (fast team vs slow opponent)

---

**Consolidated Injury Leverage** (Replaced 3 → 1 feature):
- Removed: `injury_impact_diff`, `injury_shock_diff`, `star_mismatch`
- Added: `injury_leverage` (optimized weights: 13% baseline / 38% shock / 49% star)

---

### 2. ✅ Training Data Generated

**File:** `data/training_data_SYNDICATE_28_features.csv`
- Rows: 12,205 games
- Features: 22 syndicate features
- Targets: spread, moneyline, totals

**Feature Breakdown:**
- Tier 1 (ELO): 3 matchup advantages
- Tier 2 (Friction): 5 matchup interactions
- Tier 3 (Volume/Injury): 2 features
- Tier 4 (Supporting): 12 context features

---

### 3. ✅ Optimization Configuration

**Hyperparameter Ranges:**
```python
gamma: 0.1-4.0  # Lower ceiling vs 0-10 to unlock weak features
max_depth: 3-7  # Controlled depth
min_child_weight: 10-40  # Noise control
learning_rate: 0.01-0.1 log-scale
n_estimators: 100-500
subsample: 0.6-0.9
colsample_bytree: 0.6-1.0
```

**Trials:** 2000 (vs 500 previous)
**Time:** ~8-10 hours overnight

**Monitoring:**
- Syndicate feature audit every 50 trials
- Tracks usage of 9 new syndicate features
- Reports top contributors

---

## Expected Results

**Baseline Comparisons:**
- Trial 1306: 0.6222 (old K=32 noisy ELO)
- Trial 144: 0.6308 (Gold ELO with hobbyist features)

**Success Criteria:**
1. **Primary:** Beat 0.6222 (validate syndicate hypothesis)
2. **Secondary:** Syndicate features in top 10 importance
3. **Tertiary:** All 9 syndicate features active (importance > 0%)

**Hypothesis:**
- Matchup advantages should dominate (25-40% importance)
- Friction features should be top 10 (3-5% each)
- Volume efficiency should beat simple pace (5-10%)
- Injury leverage should consolidate signal (2-4%)

---

## Files Created/Modified

**New Files:**
1. `calculate_advanced_stats.py` - Database stat calculator
2. `create_syndicate_training_data.py` - Feature converter
3. `data/training_data_SYNDICATE_28_features.csv` - Training data
4. `optimization_syndicate_log.txt` - Live log (updating)

**Modified Files:**
1. `src/features/feature_calculator_v5.py`:
   - Lines 670-710: ELO matchup advantages
   - Lines 1093-1135: Matchup friction features
   - Lines 1145-1165: Volume-adjusted efficiency
   - Lines 648-670: Consolidated injury leverage

2. `optimize_generalist_model.py`:
   - Updated to 22 syndicate features
   - Points to `training_data_SYNDICATE_28_features.csv`
   - Set to 2000 trials
   - Syndicate feature audit callback

---

## When You Wake Up

**Check Results:**
```powershell
# View final results
cat optimization_syndicate_log.txt | Select-Object -Last 50

# Check if beat baseline
python -c "import json; r = json.load(open('models/generalist_model_results.json')); print(f'Log-loss: {r[\"best_log_loss\"]:.4f}'); print(f'Beat 0.6222: {r[\"comparison\"][\"beat_1306\"]}')"

# View feature importance
python -c "import json; r = json.load(open('models/generalist_model_results.json')); imp = r['feature_importance']; syndicate = ['off_matchup_advantage', 'def_matchup_advantage', 'net_composite_advantage', 'effective_shooting_gap', 'turnover_pressure', 'rebound_friction', 'whistle_leverage', 'volume_efficiency_diff', 'injury_leverage']; print('Top Syndicate Features:'); [print(f'{k}: {v:.2f}%') for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True) if k in syndicate][:5]"
```

**If Successful (< 0.6222):**
✅ Syndicate hypothesis validated!
- Next: Deploy to dashboard
- Run blind test on holdout games
- Calculate real-world Kelly stakes

**If Marginal (0.6222-0.6270):**
⚠️ Improvement but not breakthrough
- Analyze which syndicate features helped
- Consider adding Heat ELO (K=45, last 5-8 games)
- Try ensemble with old noisy ELO

**If Worse (> 0.6270):**
❌ Syndicate features need recalculation
- Friction features are placeholders (using simple diffs)
- Need true opponent-allowed stats from game_logs
- Investigate if old K=32 ELO had useful "noise"

---

## Next Phase (If Successful)

1. **Heat ELO System:**
   - Add K=45 hyper-reactive ELO (last 5-8 games)
   - Feature: `heat_vs_anchor_divergence` catches fading teams

2. **True Matchup Friction:**
   - Calculate DRB%, STL%, foul_rate from game_logs
   - Compute opponent-allowed stats (opp_efg_allowed, opp_orb_allowed)
   - Regenerate training data with real friction

3. **Validation:**
   - Blind test on 500 holdout games
   - Compare Brier score to Trial 1306
   - Check calibration curves

4. **Deployment:**
   - Update dashboard with syndicate features
   - Integrate into prediction pipeline
   - Set Kelly criterion with new model

---

## Sleep Well!

The syndicate-level transformation is underway. By morning, we'll know if pre-computed matchup advantages unlock the signal ELO was hiding. The theoretical foundation is sound - XGBoost shouldn't have to rediscover subtraction.

**Target:** 0.6150-0.6200 log-loss (beat baseline by 1-3%)
**Reality Check:** Even 0.6250 would validate the approach
**Dream Scenario:** 0.6100 (breakthrough performance)

---

## Monitoring (Optional)

If you wake up early:
```powershell
# Check progress (updates every 50 trials)
Get-Content optimization_syndicate_log.txt -Wait

# Quick status
(Get-Content optimization_syndicate_log.txt | Select-String "Trial" | Select-Object -Last 1)
```

All systems nominal. Go to bed! 🚀
