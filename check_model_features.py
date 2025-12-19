"""Check what features the production model expects"""
import joblib
import sys

model_path = 'models/production/basket_ats_model_20251121_210643.joblib'

try:
    model_bundle = joblib.load(model_path)
    
    print(f"\n{'='*70}")
    print(f"PRODUCTION MODEL FEATURE CHECK")
    print(f"{'='*70}\n")
    
    print(f"Model: {model_path}")
    
    # Handle different formats
    if isinstance(model_bundle, dict):
        model = model_bundle.get('model')
        features = model_bundle.get('feature_names', [])
    else:
        model = model_bundle
        features = list(model.feature_names_in_)
    
    print(f"Total features: {len(features)}\n")
    
    # Check for away_composite_elo
    
    has_home_elo = 'home_composite_elo' in features
    has_away_elo = 'away_composite_elo' in features
    
    print("ELO Features Check:")
    print(f"  home_composite_elo: {'PRESENT' if has_home_elo else 'MISSING'}")
    print(f"  away_composite_elo: {'PRESENT' if has_away_elo else 'MISSING'}")
    
    if has_home_elo and not has_away_elo:
        print("\n!!! CRITICAL BUG DETECTED:")
        print("     Model was trained with home_composite_elo but NOT away_composite_elo")
        print("     This means the model never learned the away team's raw ELO signal")
        print("     RECOMMENDATION: Retrain model with corrected feature set\n")
    
    print("\nFull Feature List:")
    for i, f in enumerate(features, 1):
        marker = ""
        if f == 'home_composite_elo':
            marker = " ← HOME ELO"
        elif f == 'away_composite_elo':
            marker = " ← AWAY ELO"
        print(f"  {i:2d}. {f}{marker}")
    
    print(f"\n{'='*70}\n")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
