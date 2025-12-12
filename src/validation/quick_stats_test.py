"""Quick test to see what stats are being loaded"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "feature_calculator_v5",
    project_root / "src" / "features" / "feature_calculator_v5.py"
)
feature_calc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_calc_module)
FeatureCalculatorV5 = feature_calc_module.FeatureCalculatorV5

calc = FeatureCalculatorV5()

# Test loading stats
home_stats = calc.get_team_stats_as_of_date("CLE", "2025-11-15", lookback_games=10)
away_stats = calc.get_team_stats_as_of_date("GSW", "2025-11-15", lookback_games=10)

print("Home Stats (CLE):")
for k, v in sorted(home_stats.items()):
    print(f"  {k:20s}: {v}")

print("\nAway Stats (GSW):")
for k, v in sorted(away_stats.items()):
    print(f"  {k:20s}: {v}")
