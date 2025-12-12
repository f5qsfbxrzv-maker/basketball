"""
Feature Audit - Compare Whitelist vs Actual Generated Features
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import date
import importlib.util

# Direct import from actual file location
spec = importlib.util.spec_from_file_location(
    "feature_calculator_v5",
    project_root / "src" / "features" / "feature_calculator_v5.py"
)
feature_calc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_calc_module)
FeatureCalculatorV5 = feature_calc_module.FeatureCalculatorV5

from config.feature_whitelist import FEATURE_WHITELIST

def audit_features():
    """Compare whitelist features vs actually generated features"""
    
    print("=" * 80)
    print("FEATURE AUDIT: WHITELIST vs GENERATED")
    print("=" * 80)
    
    # Initialize calculator
    calculator = FeatureCalculatorV5()
    
    # Generate features for a recent game
    home_team = "CLE"  # Use abbreviation (database format)
    away_team = "GSW"  # Use abbreviation (database format)
    game_date_str = "2025-11-15"  # Mid-season with plenty of prior games
    
    print(f"\nTest Game: {away_team} @ {home_team} ({game_date_str})")
    
    try:
        # Calculate features (this will return ALL features before whitelist filter)
        # We'll temporarily disable whitelist filtering to see raw output
        from config import feature_whitelist as fw
        original_whitelist = fw.FEATURE_WHITELIST
        fw.FEATURE_WHITELIST = None  # Disable filtering
        
        all_features = calculator.calculate_game_features(
            home_team, away_team, game_date=game_date_str
        )
        
        # Re-enable whitelist
        fw.FEATURE_WHITELIST = original_whitelist
        
        if all_features is None:
            print("\nERROR: No features generated (missing data)")
            return
        
        # Now get filtered features
        filtered_features = calculator.calculate_game_features(
            home_team, away_team, game_date=game_date_str
        )
        
        generated_feature_names = set(all_features.keys())
        whitelist_feature_names = set(FEATURE_WHITELIST)
        filtered_feature_names = set(filtered_features.keys()) if filtered_features else set()
        
        print(f"\nFEATURE COUNTS:")
        print(f"   All Generated: {len(generated_feature_names):3d} features")
        print(f"   Whitelist:     {len(whitelist_feature_names):3d} features")
        print(f"   Filtered:      {len(filtered_feature_names):3d} features")
        
        # Features in whitelist but NOT generated
        missing_from_generation = whitelist_feature_names - generated_feature_names
        
        print(f"\nWHITELIST FEATURES NOT GENERATED ({len(missing_from_generation)}):")
        print("   (These are in whitelist but feature calculator doesn't produce them)")
        for feat in sorted(missing_from_generation):
            print(f"   - {feat}")
        
        # Features generated but NOT in whitelist (eliminated)
        eliminated_features = generated_feature_names - whitelist_feature_names
        
        print(f"\nFEATURES ELIMINATED BY WHITELIST ({len(eliminated_features)}):")
        print("   (These were generated but filtered out as noise)")
        for i, feat in enumerate(sorted(eliminated_features), 1):
            print(f"   {i:3d}. {feat}")
        
        # Features successfully retained
        retained_features = generated_feature_names & whitelist_feature_names
        
        print(f"\nFEATURES SUCCESSFULLY RETAINED ({len(retained_features)}):")
        for feat in sorted(retained_features):
            value = all_features.get(feat, 0.0)
            print(f"   - {feat:40s} = {value:.4f}")
        
        # Analysis summary
        print(f"\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Reduction Rate: {len(eliminated_features)}/{len(generated_feature_names)} = {len(eliminated_features)/len(generated_feature_names)*100:.1f}% eliminated")
        print(f"Retention Rate: {len(retained_features)}/{len(whitelist_feature_names)} = {len(retained_features)/len(whitelist_feature_names)*100:.1f}% of whitelist present")
        
        # Identify major categories eliminated
        print(f"\nMAJOR CATEGORIES ELIMINATED:")
        
        # Group eliminated features by prefix
        categories = {}
        for feat in eliminated_features:
            prefix = feat.split('_')[0] if '_' in feat else feat
            if prefix not in categories:
                categories[prefix] = []
            categories[prefix].append(feat)
        
        for prefix, feats in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"   {prefix:15s}: {len(feats):2d} features")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    audit_features()
