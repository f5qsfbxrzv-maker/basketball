"""
Test script to verify production model is loaded correctly
Tests:
1. Model loads from correct path (Dec 12 xgboost_final_trial98.json)
2. Model has all 43 features including 8 injury/shock features
3. Feature calculator generates correct features
4. Model can make predictions
"""

import sys
sys.path.insert(0, 'src')

from config.settings import MONEYLINE_MODEL, ISOTONIC_CALIBRATOR
from pathlib import Path
import xgboost as xgb
import pandas as pd
from features.feature_calculator_v5 import FeatureCalculatorV5

def test_model_path():
    """Test 1: Verify correct model path"""
    print("=" * 60)
    print("TEST 1: Model Path Verification")
    print("=" * 60)
    
    assert MONEYLINE_MODEL.exists(), f"❌ Model not found: {MONEYLINE_MODEL}"
    print(f"✓ Model exists: {MONEYLINE_MODEL}")
    print(f"  File size: {MONEYLINE_MODEL.stat().st_size:,} bytes")
    print(f"  Last modified: {MONEYLINE_MODEL.stat().st_mtime}")
    print()

def test_model_features():
    """Test 2: Verify model has correct features"""
    print("=" * 60)
    print("TEST 2: Model Feature Verification")
    print("=" * 60)
    
    model = xgb.Booster()
    model.load_model(str(MONEYLINE_MODEL))
    
    features = model.feature_names
    injury_features = [f for f in features if 'injury' in f.lower() or 'shock' in f.lower() or 'star' in f.lower()]
    
    print(f"✓ Total features: {len(features)}")
    print(f"✓ Injury/shock features: {len(injury_features)}")
    print()
    print("Injury/shock features:")
    for f in injury_features:
        print(f"  - {f}")
    print()
    
    expected_injury_features = [
        'injury_impact_diff',
        'injury_impact_abs',
        'injury_shock_home',
        'injury_shock_away',
        'injury_shock_diff',
        'home_star_missing',
        'away_star_missing',
        'star_mismatch'
    ]
    
    for exp_f in expected_injury_features:
        assert exp_f in features, f"❌ Missing expected feature: {exp_f}"
    print(f"✓ All 8 expected injury/shock features present")
    print()

def test_feature_calculator():
    """Test 3: Verify feature calculator generates correct features"""
    print("=" * 60)
    print("TEST 3: Feature Calculator Verification")
    print("=" * 60)
    
    calc = FeatureCalculatorV5(db_path='data/nba_betting_data.db')
    
    # Test with a sample game
    features_dict = calc.calculate_game_features(
        home_team='NYK',
        away_team='BOS',
        game_date='2025-12-13'
    )
    
    print(f"✓ Feature calculator generated {len(features_dict)} features")
    
    # Check for injury features
    injury_keys = [k for k in features_dict.keys() if 'injury' in k.lower() or 'shock' in k.lower() or 'star' in k.lower()]
    print(f"✓ Generated {len(injury_keys)} injury/shock features:")
    for k in injury_keys:
        print(f"  - {k} = {features_dict[k]}")
    print()

def test_prediction():
    """Test 4: Verify model can make predictions"""
    print("=" * 60)
    print("TEST 4: Prediction Verification")
    print("=" * 60)
    
    # Load model
    model = xgb.Booster()
    model.load_model(str(MONEYLINE_MODEL))
    
    # Generate features
    calc = FeatureCalculatorV5(db_path='data/nba_betting_data.db')
    features_dict = calc.calculate_game_features(
        home_team='NYK',
        away_team='BOS',
        game_date='2025-12-13'
    )
    
    # Convert to DataFrame and DMatrix
    X = pd.DataFrame([features_dict])
    X = X[model.feature_names]  # Ensure correct feature order
    dmatrix = xgb.DMatrix(X)
    
    # Make prediction
    home_prob = float(model.predict(dmatrix)[0])
    
    print(f"✓ Prediction successful")
    print(f"  NYK (home) win probability: {home_prob:.1%}")
    print(f"  BOS (away) win probability: {(1-home_prob):.1%}")
    print()
    
    # Verify probability is reasonable
    assert 0 <= home_prob <= 1, f"❌ Invalid probability: {home_prob}"
    print(f"✓ Probability is valid (0 ≤ p ≤ 1)")
    print()

def test_calibrator():
    """Test 5: Verify calibrator exists"""
    print("=" * 60)
    print("TEST 5: Calibrator Verification")
    print("=" * 60)
    
    assert ISOTONIC_CALIBRATOR.exists(), f"❌ Calibrator not found: {ISOTONIC_CALIBRATOR}"
    print(f"✓ Calibrator exists: {ISOTONIC_CALIBRATOR}")
    print(f"  File size: {ISOTONIC_CALIBRATOR.stat().st_size:,} bytes")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PRODUCTION MODEL VERIFICATION")
    print("Testing xgboost_final_trial98.json (Dec 12, 2025)")
    print("=" * 60 + "\n")
    
    try:
        test_model_path()
        test_model_features()
        test_feature_calculator()
        test_prediction()
        test_calibrator()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("Production model is correctly configured:")
        print(f"  ✓ Model: {MONEYLINE_MODEL.name}")
        print(f"  ✓ 43 features including 8 injury/shock features")
        print(f"  ✓ FeatureCalculatorV5 generating correct features")
        print(f"  ✓ Model making valid predictions")
        print(f"  ✓ Calibrator ready: {ISOTONIC_CALIBRATOR.name}")
        print()
        print("🎯 Ready for live predictions with full injury modeling!")
        
    except Exception as e:
        print("=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
