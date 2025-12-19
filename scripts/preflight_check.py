"""
Pre-flight check before running Optuna tuning
Validates data, features, and runs a quick 5-trial test
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"c:\Users\d76do\OneDrive\Documents\New Basketball Model")
DATA_PATH = PROJECT_ROOT / "data" / "training_data_matchup_with_injury_advantage.csv"

FEATURES = [
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'net_fatigue_score', 'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff',
    'ewma_orb_diff', 'ewma_vol_3p_diff', 'injury_impact_diff', 'injury_shock_diff',
    'star_mismatch', 'ewma_chaos_home', 'ewma_foul_synergy_home', 'total_foul_environment',
    'league_offensive_context', 'season_progress', 'pace_efficiency_interaction',
    'projected_possession_margin', 'three_point_matchup', 'net_free_throw_advantage',
    'star_power_leverage', 'offense_vs_defense_matchup', 'injury_matchup_advantage'
]

TARGET = 'target_moneyline_win'


def check_data():
    """Validate data file and structure"""
    print("=" * 70)
    print("1. DATA VALIDATION")
    print("=" * 70)
    
    if not DATA_PATH.exists():
        print(f"❌ FAIL: Data file not found: {DATA_PATH}")
        return False
    print(f"✅ Data file exists: {DATA_PATH.name}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df):,} games")
    
    # Check target
    if TARGET not in df.columns:
        print(f"❌ FAIL: Target '{TARGET}' not in dataset")
        return False
    print(f"✅ Target present: {TARGET} (balance: {df[TARGET].mean():.3f})")
    
    # Check features
    missing = set(FEATURES) - set(df.columns)
    if missing:
        print(f"❌ FAIL: Missing features: {missing}")
        return False
    print(f"✅ All 25 features present")
    
    # Check for NaNs
    nan_counts = df[FEATURES + [TARGET]].isna().sum()
    if nan_counts.sum() > 0:
        print(f"❌ FAIL: NaN values found:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"   {col}: {count} NaNs")
        return False
    print(f"✅ No NaN values")
    
    # Check injury_matchup_advantage
    injury_col = df['injury_matchup_advantage']
    print(f"\ninjury_matchup_advantage stats:")
    print(f"  Mean:   {injury_col.mean():.6f}")
    print(f"  Std:    {injury_col.std():.6f}")
    print(f"  Range:  [{injury_col.min():.4f}, {injury_col.max():.4f}]")
    print(f"  Zeros:  {(injury_col == 0).sum():,} ({(injury_col == 0).mean()*100:.1f}%)")
    
    return True


def quick_train_test():
    """Run quick XGBoost test with conservative params"""
    print("\n" + "=" * 70)
    print("2. QUICK TRAINING TEST")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[TARGET].values
    
    # Conservative test params
    params = {
        'max_depth': 4,
        'min_child_weight': 50,
        'gamma': 5.0,
        'learning_rate': 0.01,
        'n_estimators': 100,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 10.0,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'random_state': 42
    }
    
    print(f"Training with conservative params...")
    print(f"  max_depth={params['max_depth']}, min_child_weight={params['min_child_weight']}")
    print(f"  learning_rate={params['learning_rate']}, n_estimators={params['n_estimators']}")
    
    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X, y, verbose=False)
        print(f"✅ Training successful")
        
        # Check predictions
        y_pred = model.predict_proba(X)[:, 1]
        print(f"✅ Predictions generated")
        print(f"   Mean: {y_pred.mean():.3f}")
        print(f"   Range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
        
        # Feature importance
        importance = model.feature_importances_
        top_features = sorted(zip(FEATURES, importance), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 features:")
        for feat, imp in top_features:
            print(f"  {feat:35s}: {imp:.1f}")
        
        # Check injury feature
        injury_importance = dict(zip(FEATURES, importance))['injury_matchup_advantage']
        injury_rank = sorted(importance, reverse=True).index(injury_importance) + 1
        print(f"\ninjury_matchup_advantage:")
        print(f"  Rank: #{injury_rank}/25")
        print(f"  Importance: {injury_importance:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Training error: {e}")
        return False


def check_optuna():
    """Check Optuna installation"""
    print("\n" + "=" * 70)
    print("3. DEPENDENCIES")
    print("=" * 70)
    
    try:
        import optuna
        print(f"✅ Optuna installed (version {optuna.__version__})")
    except ImportError:
        print(f"❌ FAIL: Optuna not installed")
        print(f"   Install: pip install optuna")
        return False
    
    try:
        import sklearn
        print(f"✅ scikit-learn installed (version {sklearn.__version__})")
    except ImportError:
        print(f"❌ FAIL: scikit-learn not installed")
        return False
    
    return True


def estimate_runtime():
    """Estimate runtime for 3000 trials"""
    print("\n" + "=" * 70)
    print("4. RUNTIME ESTIMATE")
    print("=" * 70)
    
    print("Conservative estimate (5-fold CV per trial):")
    print("  Trial time: ~10-30 seconds")
    print("  3000 trials: ~8-25 hours")
    print()
    print("Recommendation:")
    print("  - Run overnight or during free time")
    print("  - Monitor first 10 trials to confirm speed")
    print("  - Can stop early if convergence looks good")
    

def main():
    """Run all checks"""
    print("\n" + "=" * 70)
    print("PRE-FLIGHT CHECK - Optuna 25-Feature Tuning")
    print("=" * 70)
    
    checks = [
        ("Data Validation", check_data),
        ("Quick Training Test", quick_train_test),
        ("Dependencies", check_optuna),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception: {e}")
            results.append(False)
    
    estimate_runtime()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all(results):
        print("✅ ALL CHECKS PASSED - Ready to run optimization!")
        print()
        print("To start tuning:")
        print("  python scripts/optuna_tune_25features.py")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before tuning")
        return 1


if __name__ == "__main__":
    sys.exit(main())
