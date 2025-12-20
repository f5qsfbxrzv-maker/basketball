# Trial 1306 Feature Deep Dive
## Understanding Collinearity Issues & Calculation Methods

Generated: December 19, 2025

---

## **Problem Features Analysis**

### **Group 1: Possession Margin Components (r=0.89, VIF=999)**

#### **`ewma_orb_diff`** - Offensive Rebound Differential
**Formula:**
```python
ewma_orb_diff = home_ewma['orb_pct'] - away_ewma['orb_pct']
```
**What it measures:** Home team's offensive rebounding advantage (recency-weighted)
- Higher value = Home gets more second-chance points
- Typical range: -0.10 to +0.10 (~10% swing)
- NBA average ORB%: ~25%

#### **`ewma_tov_diff`** - Turnover Differential  
**Formula:**
```python
ewma_tov_diff = away_ewma['tov_pct'] - home_ewma['tov_pct']
```
**What it measures:** Turnover advantage (AWAY - HOME, so positive favors home)
- Higher value = Away turns it over more = Home gets extra possessions
- Typical range: -0.05 to +0.05 (~5% swing)
- NBA average TOV%: ~13.5%

#### **`projected_possession_margin`** - Combined Possession Battle
**Formula:**
```python
projected_possession_margin = ewma_orb_diff + ewma_tov_diff
```
**What it measures:** Total extra possessions from rebounds + turnovers
- **WHY COLLINEAR:** This is a DIRECT SUM of the other two features!
- VIF=999 because it's perfectly predictable from ewma_orb_diff + ewma_tov_diff

**Why This Matters:**
- More offensive rebounds = extra shots
- Fewer turnovers = keep possession  
- Combined effect: 10 extra possessions ≈ 10-15 extra points

**The Redundancy:**
- Model has `ewma_orb_diff` (Component A)
- Model has `ewma_tov_diff` (Component B)
- Model ALSO has `projected_possession_margin` (A + B)
- XGBoost can perfectly reconstruct margin from the components

**Recommendation:** Keep `projected_possession_margin`, remove `ewma_orb_diff` and `ewma_tov_diff`
- Rationale: The combined metric captures the strategic concept better
- Removes 2 features, keeps the interpretable aggregate

---

### **Group 2: ELO Rating System (r=0.64, VIF=56-95)**

#### **`home_composite_elo`** - Home Team Overall Rating
**Formula:**
```python
composite_elo = (off_elo + def_elo) / 2
```
**What it measures:** Team's overall strength (1500 baseline)
- Higher = Better team (offense + defense combined)
- Example: Lakers 1580, Pistons 1420
- Range: ~1350 (worst) to 1650 (best)

#### **`away_composite_elo`** - Away Team Overall Rating
Same formula, different team. Measures opponent strength.

#### **`off_elo_diff`** - Offensive Mismatch
**Formula:**
```python
off_elo_diff = home_off_elo - away_off_elo
```
**What it measures:** Offensive firepower advantage
- Positive = Home offense > Away offense
- This is NOT redundant with composite - it's a COMPONENT breakdown
- Example: Home 1600 off / 1400 def, Away 1400 off / 1600 def
  - Composite diff = 0 (both 1500)
  - Off diff = +200 (home offense much better)
  - Tells different story!

#### **`def_elo_diff`** - Defensive Mismatch
**Formula:**
```python
def_elo_diff = home_def_elo - away_def_elo  
```
**What it measures:** Defensive strength advantage
- Positive = Home defense > Away defense
- Higher def_elo = BETTER defense (fewer points allowed)

**The Collinearity Issue:**
- `home_composite_elo` = average of home's off/def
- `away_composite_elo` = average of away's off/def
- `off_elo_diff` and `def_elo_diff` derive from same base ratings
- **Correlation r=0.64** because good teams tend to be good at both
- But NOT perfect collinearity - specialization matters!

**Example Showing Non-Redundancy:**
```
Team A: off_elo=1650, def_elo=1350 → composite=1500 (balanced bad defense)
Team B: off_elo=1350, def_elo=1650 → composite=1500 (balanced bad offense)

If A plays B at home:
- Composite diff = 0 (both 1500)
- Off diff = +300 (A much better offense)  
- Def diff = -300 (A much worse defense)
- This is HIGH VARIANCE game - diffs tell the story composite misses!
```

**Recommendation:** Remove ONE composite ELO (keep off/def diffs)
- Option A: Remove `home_composite_elo`, keep `away_composite_elo` 
  - Rationale: Away ELO + diffs fully determines home ELO
- Option B: Remove `away_composite_elo`, keep `home_composite_elo`
  - Rationale: Home ELO + diffs fully determines away ELO
- Test both in ablation studies

---

### **Group 3: Foul/Free Throw Ecosystem (r=0.75, VIF=7.4)**

#### **`ewma_foul_synergy_home`** - Home Free Throw Rate
**Formula:**
```python
ewma_foul_synergy_home = home_ewma['fta_rate'] * 100
```
**What it measures:** How often home team gets to the line (FTA / FGA)
- Higher = more aggressive driving, draws fouls
- NBA average: ~24%
- Typical range: 18% to 30%

#### **`ewma_foul_synergy_away`** - Away Free Throw Rate  
Same formula for away team.

#### **`total_foul_environment`** - Combined Whistle Density
**Formula:**
```python
total_foul_environment = ewma_foul_synergy_home + ewma_foul_synergy_away
```
**What it measures:** How "whistle-happy" the game will be
- High values = lots of fouls, lots of free throws, slower pace
- Low values = let them play style, faster flow
- **r=0.75 with ewma_foul_synergy_home** - very high correlation

#### **`net_free_throw_advantage`** - Whistle Mismatch (BROKEN?)
**Formula from code:**
```python
net_free_throw_advantage = away_ewma_fta_rate - home_ewma_3p_pct  
```
**What it SHOULD measure:** Free throw rate advantage
**What it ACTUALLY computes:** Away FTA rate minus home 3P%?!
- **This looks like a BUG** - comparing apples to oranges
- Should probably be: `home_fta_rate - away_fta_rate`
- Or: `(home_fta_rate - away_fta_rate) * total_foul_environment`

**The Redundancy:**
- `ewma_foul_synergy_home` = home FTA rate
- `total_foul_environment` = home FTA + away FTA  
- These are 75% correlated because home rate is a component of total
- Plus `net_free_throw_advantage` tries to measure the differential (but is buggy)

**Recommendation:** Consolidate to ONE foul metric
- Option A: Keep `total_foul_environment` only (game flow indicator)
- Option B: Fix `net_free_throw_advantage` formula, use that as differential
- Option C: Create new composite: `foul_matchup = (home_fta - away_fta) * total_fta`

---

## **Feature Calculation Summary Table**

| Feature | Formula | Purpose | Issue |
|---------|---------|---------|-------|
| `ewma_orb_diff` | home_orb% - away_orb% | Rebounding edge | Component of margin |
| `ewma_tov_diff` | away_tov% - home_tov% | Turnover edge | Component of margin |
| `projected_possession_margin` | orb_diff + tov_diff | **Total extra possessions** | **Perfect sum (VIF=999)** |
| `home_composite_elo` | (off_elo + def_elo) / 2 | Home overall strength | Redundant with diffs |
| `away_composite_elo` | (off_elo + def_elo) / 2 | Away overall strength | Redundant with diffs |
| `off_elo_diff` | home_off - away_off | **Offensive mismatch** | **Key predictor (17.6%)** |
| `def_elo_diff` | home_def - away_def | Defensive mismatch | Derived from composites |
| `ewma_foul_synergy_home` | home_fta_rate * 100 | Home whistle rate | Component of total |
| `total_foul_environment` | home_fta + away_fta | **Game whistle density** | **Sum of components** |
| `net_free_throw_advantage` | away_fta - home_3p% (?!) | FT differential | **BROKEN FORMULA** |

---

## **Pruning Strategy for Phase 2**

### **Variant A: Remove Possession Components (22→20 features)**
**Remove:** `ewma_orb_diff`, `ewma_tov_diff`  
**Keep:** `projected_possession_margin`  
**Rationale:** Margin captures the strategic concept without redundant components

### **Variant B1: Remove Home Composite ELO (22→21 features)**
**Remove:** `home_composite_elo`  
**Keep:** `away_composite_elo`, `off_elo_diff`, `def_elo_diff`  
**Math:** home_elo = away_elo + off_diff = away_elo + def_diff  
**Rationale:** Diffs + away fully determine home

### **Variant B2: Remove Away Composite ELO (22→21 features)**
**Remove:** `away_composite_elo`  
**Keep:** `home_composite_elo`, `off_elo_diff`, `def_elo_diff`  
**Math:** away_elo = home_elo - off_diff = home_elo - def_diff  
**Rationale:** Diffs + home fully determine away

### **Variant C: Consolidate Foul Features (22→20 features)**
**Remove:** `ewma_foul_synergy_home`, `net_free_throw_advantage`  
**Keep:** `total_foul_environment`  
**Rationale:** Game flow metric without component redundancy  
**TODO:** Fix net_free_throw_advantage formula and test as replacement

### **Variant D: Full Pruning (22→17 features)**
**Remove:**
1. `ewma_orb_diff` (keep projected_possession_margin)
2. `ewma_tov_diff` (keep projected_possession_margin)  
3. `away_composite_elo` (keep home + diffs) OR `home_composite_elo` (test both)
4. `ewma_foul_synergy_home` (keep total_foul_environment)
5. `net_free_throw_advantage` (broken formula, low importance 2.7%)

**Expected VIF:** All < 10, most < 5

---

## **Testing Protocol**

For each variant:
1. ✅ Train XGBoost with Trial 1306 hyperparameters
2. ✅ Compare CV log loss, accuracy (baseline: 0.6330, 63.89%)
3. ✅ Backtest on 2024-25 season (baseline ROI: 49.7%)
4. ✅ Re-compute VIF (target: all < 10)
5. ✅ Check feature importance shifts
6. ✅ Document any performance degradation

**Success Criteria:**
- Log loss within 0.005 of baseline (0.628-0.638)
- Accuracy within 1% of baseline (62.9%-64.9%)
- ROI within 10% of baseline (44.7%-54.7%)
- All VIF < 10

---

## **Key Insights**

1. **`projected_possession_margin` is redundant by design** - it's literally ewma_orb_diff + ewma_tov_diff
2. **Composite ELO redundancy is mathematical** - fully determined by off/def diffs
3. **Off/def ELO separation provides signal composite doesn't** - specialist teams matter
4. **Foul features measure same phenomenon at different granularities** - consolidation likely safe
5. **`net_free_throw_advantage` formula looks BROKEN** - investigate before using

**Next Steps:** Begin Variant A training (remove orb/tov diffs, keep margin)
