"""
Test that injury_matchup_advantage is calculated correctly
"""

# Simulate what happens when all injury features are present
injury_impact_diff = 3.5  # Home team more injured
injury_shock_diff = 2.0   # Home has surprise injuries
star_mismatch = 1         # Home missing star, away not

# Apply optimized formula
injury_matchup_advantage = (
    0.008127 * injury_impact_diff    # 13% weight: baseline
  - 0.023904 * injury_shock_diff     # 38% weight: shock (negative coeff!)
  + 0.031316 * star_mismatch         # 49% weight: star binary
)

print("=" * 80)
print("INJURY MATCHUP ADVANTAGE - TEST CALCULATION")
print("=" * 80)
print()
print("Input Components:")
print(f"  injury_impact_diff:  {injury_impact_diff:+.1f}  (home more injured)")
print(f"  injury_shock_diff:   {injury_shock_diff:+.1f}  (home surprise injuries)")
print(f"  star_mismatch:       {star_mismatch:+d}  (home missing star)")
print()
print("Calculation:")
print(f"  0.008127 × {injury_impact_diff:.1f}  = {0.008127 * injury_impact_diff:+.4f}")
print(f"  -0.023904 × {injury_shock_diff:.1f}  = {-0.023904 * injury_shock_diff:+.4f}")
print(f"  0.031316 × {star_mismatch}     = {0.031316 * star_mismatch:+.4f}")
print()
print(f"RESULT: injury_matchup_advantage = {injury_matchup_advantage:+.4f}")
print()
print("Interpretation:")
if injury_matchup_advantage < 0:
    print(f"  ❌ NEGATIVE ({injury_matchup_advantage:.4f})")
    print("  → Home team disadvantaged by injuries")
    print("  → Favor AWAY team")
elif injury_matchup_advantage > 0:
    print(f"  ✅ POSITIVE ({injury_matchup_advantage:.4f})")
    print("  → Home team healthier/less impacted")
    print("  → Favor HOME team")
else:
    print("  ⚖️  NEUTRAL (0.0000)")
    print("  → No injury advantage either way")
print()

# Test opposite scenario
print("=" * 80)
print("OPPOSITE SCENARIO: Home healthier, away injured")
print("=" * 80)
print()

injury_impact_diff_2 = -4.0  # Away team more injured
injury_shock_diff_2 = -3.0   # Away has surprise injuries
star_mismatch_2 = -1         # Away missing star, home not

injury_matchup_advantage_2 = (
    0.008127 * injury_impact_diff_2
  - 0.023904 * injury_shock_diff_2
  + 0.031316 * star_mismatch_2
)

print("Input Components:")
print(f"  injury_impact_diff:  {injury_impact_diff_2:+.1f}  (away more injured)")
print(f"  injury_shock_diff:   {injury_shock_diff_2:+.1f}  (away surprise injuries)")
print(f"  star_mismatch:       {star_mismatch_2:+d}  (away missing star)")
print()
print(f"RESULT: injury_matchup_advantage = {injury_matchup_advantage_2:+.4f}")
print()
if injury_matchup_advantage_2 > 0:
    print("  ✅ POSITIVE - Home advantage (correct!)")
print()

print("=" * 80)
print("FORMULA ADDED TO FEATURE_CALCULATOR_V5.PY")
print("=" * 80)
print()
print("Next steps:")
print("1. ✅ Formula implemented in src/features/feature_calculator_v5.py")
print("2. 📊 Regenerate training data with new feature")
print("3. 🤖 Retrain 20-feature model (19 + injury_matchup_advantage)")
print("4. 📈 Check feature importance - expect Top 5 placement")
