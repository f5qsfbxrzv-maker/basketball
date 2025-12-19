# NBA Betting System - Complete Feature Documentation

## All 25 Features Used in Training (Gold Standard ELO Model)

### **ELO FEATURES (8 features - 41.2% importance)**

1. **`off_elo_diff`** (22.1% importance - #1 MOST IMPORTANT)
   - **Logic**: Home offensive ELO - Away defensive ELO
   - **What it measures**: Offensive firepower advantage when home team HAS the ball
   - **Range**: Typically -400 to +400
   - **Interpretation**: +100 means home offense is 100 ELO points better than away defense

2. **`def_elo_diff`** (12.1% importance - #2)
   - **Logic**: Away offensive ELO - Home defensive ELO  
   - **What it measures**: Defensive vulnerability when away team HAS the ball
   - **Interpretation**: +100 means away offense is 100 points better than home defense (bad for home)

3. **`home_composite_elo`** (3.6% importance - #4)
   - **Logic**: (Home off_elo + (1500 - Home def_elo)) / 2
   - **What it measures**: Overall home team strength (combines offense and inverted defense)
   - **Baseline**: 1500 is league average

4. **`away_composite_elo`** (3.3% importance - #6)
   - **Logic**: Same formula as home_composite_elo
   - **What it measures**: Overall away team strength

5-8. **Deprecated ELO features** (home_off_elo, home_def_elo, away_off_elo, away_def_elo)
   - Still in data but replaced by differentials above

---

### **INJURY FEATURES (4 features)**

9. **`injury_matchup_advantage`** (3.2% importance - #7)
   - **Logic**: Weighted combination of:
     - 13% baseline PIE differential  
     - 38% shock differential (NEW injuries vs rolling average)
     - 49% star_mismatch binary flags
   - **Formula**: `0.008127 * injury_impact_diff - 0.023904 * injury_shock_diff + 0.031316 * star_mismatch`
   - **Interpretation**: Positive = home has injury advantage

10. **`injury_shock_diff`** (3.2% importance - #8)
    - **Logic**: (Home today injury - Home EWMA injury) - (Away today - Away EWMA)
    - **What it measures**: NEW/UNEXPECTED injuries (shock value)
    - **Example**: If team usually has 2.0 PIE impact but today has 6.0, shock = +4.0
    - **Why it matters**: EWMA stats "absorb" long-term injuries, this catches surprises

11. **`injury_impact_diff`** (2.9% importance - #13)
    - **Logic**: Home total PIE lost - Away total PIE lost
    - **Calculation**: Sum of (PIE × 20 × gravity_multiplier) for all injured players
    - **Threshold**: Only players with PIE ≥ 0.08 counted
    - **Cap**: Max 15.0 total per team (prevents extreme outliers)

12. **`star_mismatch`** (**0.0% importance - BLOCKED** ❌)
    - **Logic**: `(1 if home_injury ≥ 4.0 else 0) - (1 if away_injury ≥ 4.0 else 0)`
    - **What it measures**: Binary flag for ELITE star missing (4.0 PIE = top-tier player)
    - **Values**: 
      - +1 = Home missing star, away not (home disadvantage)
      - 0 = Both have stars or both don't
      - -1 = Away missing star, home not (home advantage)
    - **PURPOSE**: "Loud" binary signal for decision trees to split on
    - **WHY BLOCKED**: gamma=3.99 too high - XGBoost won't split on it when `injury_shock_diff` and `injury_matchup_advantage` provide better continuous signal
    - **NOTE**: This feature IS used in the `injury_matchup_advantage` calculation (49% weight), so its signal is indirectly captured

---

### **EWMA RECENT FORM FEATURES (12 features)**

These use Exponentially Weighted Moving Average (EWMA) over last 10 games with span=10 (α ≈ 0.18, giving ~50% weight to last 3.8 games)

13. **`ewma_efg_diff`** (3.0% importance - #11)
    - **Logic**: Home recent eFG% - Away recent eFG%
    - **What it measures**: Shooting efficiency trend
    - **Formula**: eFG% = (FGM + 0.5 * FG3M) / FGA

14. **`ewma_tov_diff`** (2.8% importance - #15) ✓ RECOVERED
    - **Logic**: Away recent TOV% - Home recent TOV%
    - **Interpretation**: Positive = away turns it over more (good for home)

15. **`ewma_orb_diff`** (3.0% importance - #10)
    - **Logic**: Home recent ORB% - Away recent ORB%
    - **What it measures**: Offensive rebounding advantage

16. **`ewma_pace_diff`** (3.6% importance - #5)
    - **Logic**: Home recent pace - Away recent pace
    - **What it measures**: Tempo differential (possessions per 48 min)

17. **`ewma_vol_3p_diff`** (2.9% importance - #14)
    - **Logic**: Home recent 3PA/100 poss - Away recent 3PA/100
    - **What it measures**: Three-point volume differential

18. **`ewma_foul_synergy_home`** (2.6% importance - #22) ✓ RECOVERED
    - **Logic**: Home recent FTA rate × 100
    - **What it measures**: Home team's ability to get to the free throw line

19. **`ewma_chaos_home`** (2.6% importance - #23)
    - **Logic**: Standard deviation of home's recent net rating
    - **What it measures**: Team consistency/volatility
    - **Interpretation**: High = inconsistent (dangerous or beatable)

20-24. **Other EWMA features** (lower importance):
    - `home_ewma_3p_pct`, `away_ewma_3p_pct`: Recent 3P shooting
    - `away_ewma_tov_pct`: Away turnover rate (absolute)
    - `home_orb`, `away_orb`: Rebounding rates (absolute)
    - `total_foul_environment`: Sum of both teams' foul rates

---

### **ADVANCED CALCULATED FEATURES (5+ features)**

25. **`net_fatigue_score`** (3.9% importance - #3 OVERALL)
    - **Logic**: Composite rest advantage calculation
    - **Includes**: 
      - Rest days differential (home - away)
      - Back-to-back penalties
      - 3-in-4 night penalties
      - Travel distance fatigue
    - **Why it matters**: Physical edge from rest differential

26. **`projected_possession_margin`** (3.1% importance - #9)
    - **Logic**: Expected possessions × efficiency differential
    - **What it measures**: How many extra possessions home will get (via pace/turnovers)

27. **`offense_vs_defense_matchup`** (3.0% importance - #12)
    - **Logic**: (Home off_rating - Away def_rating) - (Away off_rating - Home def_rating)
    - **What it measures**: Net offensive/defensive matchup advantage

28. **`net_free_throw_advantage`** (2.8% importance - #16)
    - **Logic**: (Home FTA rate - Away FTA rate) × (Away foul propensity)
    - **What it measures**: Expected FT advantage from matchup styles

29. **`season_progress`** (2.8% importance - #17) ✓ RECOVERED
    - **Logic**: Games played / 82
    - **What it measures**: Point in season (early = more variance, late = locked rotations)
    - **Range**: 0.0 to 1.0

30. **`pace_efficiency_interaction`** (2.8% importance - #18)
    - **Logic**: (Home pace + Away pace) / 2 × efficiency_diff
    - **What it measures**: How pace amplifies or dampens efficiency edge

31. **`star_power_leverage`** (2.7% importance - #19) ✓ RECOVERED
    - **Logic**: (Home top 3 PIE sum - Away top 3 PIE sum) × playoff_weight
    - **What it measures**: Star power differential (matters more in close games)

32. **`total_foul_environment`** (2.7% importance - #20)
    - **Logic**: Home FTA rate + Away FTA rate
    - **What it measures**: Expected game flow (free throws slow game, more variance)

33. **`league_offensive_context`** (2.7% importance - #21)
    - **Logic**: League average offensive rating for this season
    - **What it measures**: Era adjustment (offense higher in recent years)

34. **`three_point_matchup`** (2.6% importance - #24)
    - **Logic**: (Home 3PA/100 - Away 3PA/100) × (Home 3P% - League avg)
    - **What it measures**: Three-point style matchup advantage

---

## Feature Recovery Status (After Gamma Constraint Optimization)

**✓ RECOVERED (4/5):**
1. `ewma_tov_diff` - 2.8% importance (was 0.0%)
2. `ewma_foul_synergy_home` - 2.6% importance (was 0.0%)
3. `season_progress` - 2.8% importance (was 0.0%)
4. `star_power_leverage` - 2.7% importance (was 0.0%)

**✗ STILL BLOCKED (1/5):**
1. `star_mismatch` - 0.0% importance
   - **Why**: gamma=3.99 still too high for binary feature
   - **Impact**: Signal IS captured via `injury_matchup_advantage` (which uses star_mismatch with 49% weight)
   - **Solution options**:
     - Lower gamma ceiling to 0.1-2.5 (more aggressive)
     - Accept that continuous `injury_shock_diff` is superior
     - Feature is redundant but theoretically valid

---

## Feature Importance Hierarchy (Trial #144)

```
Tier 1 (ELO Foundation): 41.2% total
  - off_elo_diff: 22.1%
  - def_elo_diff: 12.1%
  - home_composite_elo: 3.6%
  - away_composite_elo: 3.3%

Tier 2 (Context): 20.5% total
  - net_fatigue_score: 3.9%
  - ewma_pace_diff: 3.6%
  - injury_matchup_advantage: 3.2%
  - injury_shock_diff: 3.2%
  - projected_possession_margin: 3.1%
  - offense_vs_defense_matchup: 3.0%
  - ewma_orb_diff: 3.0%
  - ewma_efg_diff: 3.0%

Tier 3 (Refinement): 35.6% total
  - All other 15 features contribute 1.8-2.9% each
  - Includes EWMA differentials, matchup advantages, situational adjustments

Tier 4 (Blocked): 0.0%
  - star_mismatch: 0.0% (signal captured in injury_matchup_advantage)
```

---

## Why This Matters

**Old Model (Trial 1306):**
- ELO importance: 26.3% (noisy K=32 ELO where Brooklyn was #3)
- Used 25 features, ALL non-zero
- Log-loss: **0.6222** (baseline to beat)

**First Fix (Trial 340):**
- ELO importance: 47.7% (+81% increase with K=15 Gold Standard)
- gamma=9.537 blocked 5 features → 0.0% importance
- Log-loss: **0.6297** (1.2% worse despite better ELO!)

**Current Model (Trial 144):**
- ELO importance: 41.2% (strong foundation)
- gamma=3.99 (constrained to 0.1-4.0 range)
- 4/5 features recovered, 1 blocked but signal captured elsewhere
- Log-loss: **0.6308** (0.5% worse than Trial 340, 1.4% worse than baseline)

**Still Not Beating Baseline** - Options:
1. Lower gamma further (0.1-2.5) to force star_mismatch usage
2. Accept that continuous features > binary flags
3. Re-optimize min_child_weight range (currently 10-40)
4. Run more trials (stopped at 228/500)
5. Investigate if old K=32 ELO had useful "noise" signal
