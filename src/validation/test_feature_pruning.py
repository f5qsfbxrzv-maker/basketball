"""
Test Feature Pruning Implementation
Verifies that feature_calculator_v5.py correctly applies FEATURE_WHITELIST
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import date

# Direct import from actual file location
import importlib.util
spec = importlib.util.spec_from_file_location(
    "feature_calculator_v5",
    project_root / "src" / "features" / "feature_calculator_v5.py"
)
feature_calc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_calc_module)
FeatureCalculatorV5 = feature_calc_module.FeatureCalculatorV5

from config.feature_whitelist import FEATURE_WHITELIST

def test_feature_pruning():
    """Test that features are correctly filtered to whitelist"""
    
    print("=" * 70)
    print("FEATURE PRUNING VALIDATION TEST")
    print("=" * 70)
    
    # Initialize feature calculator
    calculator = FeatureCalculatorV5()
    
    # Test game: Recent game with actual data
    home_team = "Cleveland Cavaliers"
    away_team = "Golden State Warriors"
    
    # Use date from Dec 2025 (we just updated game logs)
    from datetime import datetime
    game_date = date(2025, 12, 5)  # Recent game with fresh data
    
    print(f"\nTest Game: {away_team} @ {home_team}")
    print(f"Date: {game_date}")
    
    # Calculate features
    try:
        features = calculator.calculate_game_features(home_team, away_team, game_date)
        
        if features is None:
            print("\n❌ ERROR: calculate_game_features() returned None")
            print("   Likely missing data in database for test teams/date")
            return False
            
        actual_feature_count = len(features)
        expected_feature_count = len(FEATURE_WHITELIST)
        
        print(f"\n✅ Features Calculated Successfully")
        print(f"   Expected: {expected_feature_count} features (from whitelist)")
        print(f"   Actual:   {actual_feature_count} features (returned)")
        
        # Check if pruning worked
        if actual_feature_count == expected_feature_count:
            print(f"\n✅ PRUNING SUCCESSFUL: {actual_feature_count} features match whitelist")
        elif actual_feature_count > expected_feature_count:
            print(f"\n⚠️  WARNING: More features than expected ({actual_feature_count} > {expected_feature_count})")
            extra_features = set(features.keys()) - set(FEATURE_WHITELIST)
            if extra_features:
                print(f"   Extra features: {sorted(extra_features)[:10]}...")
        else:
            print(f"\n⚠️  WARNING: Fewer features than expected ({actual_feature_count} < {expected_feature_count})")
            missing_features = set(FEATURE_WHITELIST) - set(features.keys())
            if missing_features:
                print(f"   Missing features: {sorted(missing_features)[:10]}...")
        
        # Show sample features
        print(f"\n📊 Sample Features (first 10):")
        for i, (feat_name, feat_value) in enumerate(sorted(features.items())[:10], 1):
            print(f"   {i:2d}. {feat_name:30s} = {feat_value:.4f}")
        
        # Verify critical injury features present
        critical_injury_features = ['injury_impact_diff', 'injury_impact_abs', 'injury_elo_interaction']
        present_injury_features = [f for f in critical_injury_features if f in features]
        
        print(f"\n🏥 Critical Injury Features:")
        for feat in critical_injury_features:
            if feat in features:
                print(f"   ✅ {feat:30s} = {features[feat]:.4f}")
            else:
                print(f"   ❌ {feat:30s} = MISSING")
        
        # Verify no redundant features leaked through
        redundant_features = [
            'home_pace', 'away_pace',  # Should only have ewma_pace_diff
            'home_tov', 'away_tov',    # Should only have ewma_tov_diff
            'foul_rate',                # Should only have ewma_foul_synergy
        ]
        leaked_redundant = [f for f in redundant_features if f in features]
        
        if leaked_redundant:
            print(f"\n❌ REDUNDANT FEATURES LEAKED:")
            for feat in leaked_redundant:
                print(f"   - {feat}")
        else:
            print(f"\n✅ No redundant features leaked (good!)")
        
        print("\n" + "=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during feature calculation:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_pruning()
    sys.exit(0 if success else 1)
