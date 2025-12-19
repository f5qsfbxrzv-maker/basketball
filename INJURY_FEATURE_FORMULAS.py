"""
COMPLETE INJURY FEATURE FORMULAS
=================================

All formulas used in the 43-feature model, extracted from feature_calculator_v5.py
"""

print("=" * 80)
print("INJURY FEATURE FORMULAS - COMPLETE REFERENCE")
print("=" * 80)
print()

print("STEP 1: Calculate Raw PIE-Based Injury Impact")
print("-" * 80)
print("""
For each injured player:
  player_impact = PIE_score * status_weight
  
  Status weights:
    - OUT: 1.0 (fully missing)
    - DOUBTFUL: 0.7 (likely out)
    - QUESTIONABLE: 0.3 (might play)
    - DAY_TO_DAY: 0.1 (minor)

Total team impact:
  home_injury_impact = Σ(player_impact for home injured players)
  away_injury_impact = Σ(player_impact for away injured players)
""")

print("STEP 2: Basic Injury Features")
print("-" * 80)
print("""
injury_impact_diff = home_injury_impact - away_injury_impact
  → Positive = home team more injured (disadvantage)
  → Importance: 6.0

injury_impact_abs = abs(home_injury_impact) + abs(away_injury_impact)
  → Total injury load across both teams
  → Importance: 6.4
""")

print("STEP 3: Injury Shock (EWMA-Based)")
print("-" * 80)
print("""
EWMA calculation (for each team):
  ewma_injury_impact = exponentially_weighted_moving_average(
      injury_impact over last 10 games,
      alpha = 0.3  # decay factor
  )

Shock calculation:
  injury_shock_home = home_injury_impact - ewma_injury_home
  injury_shock_away = away_injury_impact - ewma_injury_away
  injury_shock_diff = injury_shock_home - injury_shock_away

  → Positive shock = worse than usual (new bad news)
  → Negative shock = better than usual (player returned)
  → Importance: shock_diff = 7.0 (HIGHEST), shock_home = 6.7, shock_away = 6.6
""")

print("STEP 4: Star Binary Flags")
print("-" * 80)
print("""
Star threshold = 4.0 PIE points
  (Implies elite starter: ~15+ PER, top 3 rotation player)

home_star_missing = 1 if home_injury_impact >= 4.0 else 0
away_star_missing = 1 if away_injury_impact >= 4.0 else 0
star_mismatch = home_star_missing - away_star_missing

  → Binary "loud signal" for tree models
  → Importance: away_star_missing = 7.7, home_star_missing = 6.1, star_mismatch = 6.2
""")

print("=" * 80)
print("COMPREHENSIVE INJURY MATCHUP FORMULA (TO BE OPTIMIZED)")
print("=" * 80)
print("""
injury_matchup_advantage = (
    w1 * injury_impact_diff          # Baseline talent gap
  + w2 * injury_shock_diff           # Surprise factor
  + w3 * star_mismatch * scale       # Star power (may need scaling)
  + w4 * injury_impact_abs * sign    # Total load (directional)
)

Where:
  w1, w2, w3, w4 = weights to be optimized by regression
  scale = multiplier for star binary (5.0 suggested)
  sign = sign(injury_impact_diff) for directional total load

Current "guess" weights (from importance ratios):
  w1 = 0.40  (impact_diff baseline)
  w2 = 0.30  (shock_diff highest importance)
  w3 = 0.20  (star_mismatch)
  w4 = 0.10  (total load context)

NEXT STEP: Run optimize_injury_weights.py to find optimal weights from data
""")

print("=" * 80)
print("RAW COMPONENT FORMULAS FOR OPTIMIZATION SCRIPT")
print("=" * 80)
print("""
The optimization script needs these raw values:

X1 = injury_impact_diff
   = home_injury_impact - away_injury_impact

X2 = injury_shock_diff
   = (home_injury_impact - ewma_home) - (away_injury_impact - ewma_away)

X3 = star_mismatch
   = (1 if home_injury_impact >= 4.0 else 0) - (1 if away_injury_impact >= 4.0 else 0)

y = home_win (binary: 1 = home won, 0 = away won)

Logistic regression will output coefficients [β1, β2, β3]:
  injury_matchup_advantage = β1*X1 + β2*X2 + β3*X3
""")
