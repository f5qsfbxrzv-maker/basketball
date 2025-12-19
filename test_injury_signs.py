"""
TEST 1: PEN & PAPER SANITY CHECK
=================================

Scenario: Lakers (Home) lose LeBron James right before game
          Celtics (Away) are healthy

Given:
  - LeBron PIE = 5.0
  - Home (LAL): 1 star out (LeBron)
  - Away (BOS): Healthy (0 injured)
"""

print("=" * 80)
print("TEST 1: DISASTER SCENARIO - LEBRON OUT")
print("=" * 80)
print()

# Raw components
pie_diff = 5.0      # Home has 5.0 injured PIE
shock_diff = 5.0    # New injury (not in EWMA baseline)
star_diff = 1       # Home has 1 star out, away has 0

print("Input Components:")
print(f"  pie_diff     = {pie_diff:+.1f}  (Home injured PIE)")
print(f"  shock_diff   = {shock_diff:+.1f}  (NEW injury - market-moving news)")
print(f"  star_diff    = {star_diff:+d}  (Home missing star, away healthy)")
print()

# Apply the EXACT formula from feature_calculator_v5.py
injury_matchup_advantage = (
    0.008127 * pie_diff      # 13% weight: baseline
  - 0.023904 * shock_diff    # 38% weight: shock (NEGATIVE coeff)
  + 0.031316 * star_diff     # 49% weight: star binary
)

print("Formula Calculation:")
print(f"  0.008127 × {pie_diff:.1f}      = {0.008127 * pie_diff:+.6f}")
print(f" -0.023904 × {shock_diff:.1f}      = {-0.023904 * shock_diff:+.6f}")
print(f"  0.031316 × {star_diff}        = {0.031316 * star_diff:+.6f}")
print()
print(f"  Total = {injury_matchup_advantage:+.6f}")
print()

print("=" * 80)
print("RESULT")
print("=" * 80)
print()

if injury_matchup_advantage < 0:
    print(f"✅ NEGATIVE ({injury_matchup_advantage:.6f})")
    print()
    print("Interpretation: HOME TEAM DISADVANTAGED")
    print("  - Lakers missing LeBron = BAD for Lakers")
    print("  - Negative score = Fade home team")
    print("  - Betting recommendation: BET AWAY (Celtics)")
    print()
    print("✅ SIGNS ARE CORRECT! Formula passes sanity check.")
elif injury_matchup_advantage > 0:
    print(f"❌ POSITIVE ({injury_matchup_advantage:.6f})")
    print()
    print("ERROR: This would suggest home team is ADVANTAGED by losing LeBron!")
    print("❌ SIGN ERROR DETECTED - DO NOT USE THIS FORMULA")
else:
    print("⚠️  ZERO (0.0000)")
    print("WARNING: No signal detected")

print()
print("=" * 80)
print("BREAKDOWN: Why is the result small?")
print("=" * 80)
print()
print("The coefficients are small because:")
print("  1. Logistic regression outputs are scaled for log-odds")
print("  2. The raw feature values (PIE=5.0) are already large")
print("  3. What matters is the SIGN (negative = home disadvantage)")
print()
print("In model training, XGBoost will learn the optimal scaling.")
print("Our job here is only to verify the DIRECTION is correct.")
print()

# Test opposite scenario
print("=" * 80)
print("TEST 2: OPPOSITE SCENARIO - CELTICS INJURED")
print("=" * 80)
print()

pie_diff_2 = -5.0    # Away has 5.0 injured PIE
shock_diff_2 = -5.0  # Away new injury
star_diff_2 = -1     # Away missing star, home healthy

injury_matchup_advantage_2 = (
    0.008127 * pie_diff_2
  - 0.023904 * shock_diff_2
  + 0.031316 * star_diff_2
)

print("Input: Away injured (Celtics), Home healthy (Lakers)")
print(f"  pie_diff     = {pie_diff_2:+.1f}")
print(f"  shock_diff   = {shock_diff_2:+.1f}")
print(f"  star_diff    = {star_diff_2:+d}")
print()
print(f"Result: {injury_matchup_advantage_2:+.6f}")
print()

if injury_matchup_advantage_2 > 0:
    print("✅ POSITIVE - Home advantage (Lakers healthier)")
    print("✅ CORRECT: Away injured = favor home team")
elif injury_matchup_advantage_2 < 0:
    print("❌ NEGATIVE - This would be WRONG")
    print("❌ SIGN ERROR: Away injured should help home team")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Expected behavior:")
print("  ✅ Home injured → NEGATIVE score → Fade home")
print("  ✅ Away injured → POSITIVE score → Back home")
print()
print("Test Results:")
if injury_matchup_advantage < 0 and injury_matchup_advantage_2 > 0:
    print("  ✅✅ BOTH TESTS PASSED")
    print("  Formula is correctly signed and ready for production")
else:
    print("  ❌ TESTS FAILED - DO NOT USE")
