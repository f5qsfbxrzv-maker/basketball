# Experimental Variants - Performance Comparison
**Generated:** December 19, 2025

---

## **Results Summary**

| Variant | Description | Features | CV Log Loss | Δ Log Loss | CV Accuracy | Δ Accuracy | Status |
|---------|-------------|----------|-------------|------------|-------------|------------|--------|
| **Trial 1306 (Baseline)** | Production model | 22 | **0.6330** | — | **63.89%** | — | ✅ Baseline |
| **Variant A** | Remove orb/tov diffs | 20 | **0.6317** | **-1.3e-3** ✨ | **64.13%** | **+0.24%** ✨ | ✅ **IMPROVEMENT** |
| **Variant B1** | Remove home_composite | 21 | 0.6334 | +0.4e-3 | 63.96% | +0.07% | ✅ Neutral |
| **Variant B2** | Remove away_composite | 21 | 0.6333 | +0.3e-3 | 63.96% | +0.07% | ✅ Neutral |
| **Variant B3** | Remove both composites | 20 | 0.6339 | +0.9e-3 | 64.06% | +0.17% | ✅ Neutral |

---

## **Key Findings**

### **🏆 Variant A: Best Performer**
**Configuration:** Remove `ewma_orb_diff` + `ewma_tov_diff`, keep `projected_possession_margin`

**Performance:**
- ✨ **Improved log loss** by 1.3e-3 (0.6317 vs 0.6330)
- ✨ **Improved accuracy** by 0.24% (64.13% vs 63.89%)
- ✨ **Reduced features** by 2 (22 → 20)
- ✨ **Expected VIF improvement** from 999 → manageable

**Feature Importance Shift:**
- `off_elo_diff` increased: 17.6% → **20.9%** (more weight on primary predictor)
- `projected_possession_margin` now captures full possession battle signal
- ELO features remain dominant: top 4 spots

**Recommendation:** ✅ **PROMOTE TO PHASE 2** - Combine with best ELO config

---

### **📊 Variant B: ELO Configuration Testing**

#### **B1: Remove `home_composite_elo`**
- CV Log Loss: 0.6334 (+0.4e-3)
- CV Accuracy: 63.96% (+0.07%)
- Feature importance: `off_elo_diff` (19.3%), `def_elo_diff` (8.6%), `away_composite_elo` (6.9%)

#### **B2: Remove `away_composite_elo`** 
- CV Log Loss: 0.6333 (+0.3e-3) ← **Slightly better**
- CV Accuracy: 63.96% (+0.07%)
- Feature importance: `off_elo_diff` (19.4%), `home_composite_elo` (9.9%), `def_elo_diff` (8.0%)

#### **B3: Remove BOTH composites (diffs only)**
- CV Log Loss: 0.6339 (+0.9e-3) ← Worst
- CV Accuracy: 64.06% (+0.17%)
- Feature importance: `off_elo_diff` (19.3%), `def_elo_diff` (10.3%)

**Analysis:**
- **B2 (remove away) is marginally best** but difference is tiny (0.0001 log loss)
- **B3 (diffs only)** loses signal - composites do add value beyond diffs
- Keeping ONE composite provides context the diffs don't capture alone
- **Home team anchor** (B2) slightly better than away team anchor (B1)

**Recommendation:** ✅ **Use B2 configuration** (remove `away_composite_elo`, keep `home_composite_elo`)

---

## **Optimal Combined Configuration (Variant D)**

**Remove these 5 features:**
1. `ewma_orb_diff` (component of margin)
2. `ewma_tov_diff` (component of margin)
3. `away_composite_elo` (redundant with home + diffs, per B2 analysis)
4. `ewma_foul_synergy_home` (to be tested in Variant C)
5. `net_free_throw_advantage` (broken formula, low importance)

**Expected result: 17 features, all VIF < 10**

**Remaining Features (17):**
- ELO System (3): `home_composite_elo`, `off_elo_diff`, `def_elo_diff`
- EWMA Stats (5): `ewma_efg_diff`, `ewma_pace_diff`, `ewma_vol_3p_diff`, `ewma_chaos_home`, `total_foul_environment`
- Context (4): `net_fatigue_score`, `injury_matchup_advantage`, `season_progress`, `league_offensive_context`
- Interactions (5): `pace_efficiency_interaction`, `projected_possession_margin`, `three_point_matchup`, `star_power_leverage`, `offense_vs_defense_matchup`

---

## **Next Steps**

1. ✅ **Train Variant C** (test foul feature consolidation)
2. ✅ **Train Variant D** (A + B2 + C combined = 17 features)
3. ✅ **Re-run VIF analysis** on Variant D (expect all < 10)
4. ✅ **Backtest on 2024-25** season (compare ROI vs 49.7% baseline)
5. ⚠️ **Decision gate**: If Variant D maintains performance with 5 fewer features → production candidate

---

## **Statistical Significance Notes**

- All variants show **<1% accuracy difference** from baseline
- All variants show **<0.001 log loss difference** from baseline
- Differences are SMALL - focus on:
  - VIF reduction (multicollinearity health)
  - Feature count reduction (model simplicity)
  - Interpretability improvement
  - Generalization on holdout data

**Success is maintaining performance while reducing complexity**, not necessarily improving metrics on training data.
