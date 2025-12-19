"""Show all features organized by category"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.features.feature_calculator_live import FeatureCalculatorV5

# Test game
calc = FeatureCalculatorV5()
home_team = "OKC"
away_team = "SAS"
game_date = "2025-12-16"

print(f"\n{'='*70}")
print(f"FEATURE VALIDATION: {away_team} @ {home_team} on {game_date}")
print(f"{'='*70}\n")

features = calc.calculate_game_features(
    home_team=home_team,
    away_team=away_team,
    game_date=game_date
)

print(f"Total Features: {len(features)}\n")

# Categorize features
elo_features = {k: v for k, v in features.items() if 'elo' in k.lower()}
four_factors = {k: v for k, v in features.items() if any(x in k for x in ['efg', 'tov', 'orb', 'ftr', 'ts_pct'])}
ewma_features = {k: v for k, v in features.items() if k.startswith('ewma_')}
injury_features = {k: v for k, v in features.items() if 'injury' in k or 'shock' in k or 'star' in k.lower()}
rest_features = {k: v for k, v in features.items() if any(x in k for x in ['rest', 'fatigue', 'back_to_back', '3in4'])}
other_features = {k: v for k, v in features.items() if k not in elo_features and k not in four_factors and k not in ewma_features and k not in injury_features and k not in rest_features}

def show_category(name, feat_dict):
    if not feat_dict:
        print(f"{name}: NONE FOUND")
        return
    print(f"{name} ({len(feat_dict)} features):")
    for key, value in sorted(feat_dict.items()):
        # Check for defaults or missing values
        status = ""
        if value == 1500:
            status = " [DEFAULT - POTENTIAL BUG]"
        elif value == 0.0:
            status = " [ZERO - CHECK IF VALID]"
        elif value is None:
            status = " [NONE - MISSING]"
        print(f"  {key:35s} = {value:>10.4f}{status}")
    print()

show_category("1. ELO FEATURES", elo_features)
show_category("2. FOUR FACTORS", four_factors)
show_category("3. EWMA FEATURES", ewma_features)
show_category("4. INJURY FEATURES", injury_features)
show_category("5. REST/FATIGUE", rest_features)
show_category("6. OTHER FEATURES", other_features)

# Final validation
print(f"{'='*70}")
errors = []
if 'away_composite_elo' not in features:
    errors.append("away_composite_elo is MISSING")
elif features['away_composite_elo'] == 0.0:
    errors.append("away_composite_elo is ZERO")
elif features['away_composite_elo'] == 1500:
    errors.append("away_composite_elo is DEFAULT (1500)")

if not four_factors:
    errors.append("Four Factors are MISSING (efg, tov, orb, ftr, ts_pct)")

if not ewma_features:
    errors.append("EWMA features are MISSING")

if errors:
    print("ERRORS FOUND:")
    for err in errors:
        print(f"  - {err}")
else:
    print("ALL FEATURES VALIDATED:")
    print(f"  - away_composite_elo: {features['away_composite_elo']:.2f}")
    print(f"  - Four Factors: {len(four_factors)} features")
    print(f"  - EWMA features: {len(ewma_features)} features")
print(f"{'='*70}\n")
