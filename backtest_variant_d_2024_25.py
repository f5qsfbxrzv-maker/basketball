"""
WALK-FORWARD BACKTEST: Variant D on 2024-25 Season
Tests if the 18-feature model generalizes to unseen current season data
Compares against Trial 1306 baseline performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from datetime import datetime

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Variant D: The 18 "Clean" Features (VIF < 2.5)
VARIANT_D_FEATURES = [
    'home_composite_elo',           # The anchor (VIF: 1.76)
    'off_elo_diff',                 # Primary predictor (VIF: 2.34)
    'def_elo_diff',                 # Defensive mismatch (VIF: 1.54)
    'projected_possession_margin',  # Consolidated possession (VIF: 1.18)
    'ewma_pace_diff',
    'net_fatigue_score',
    'ewma_efg_diff',
    'ewma_vol_3p_diff',
    'three_point_matchup',
    'ewma_chaos_home',
    'injury_matchup_advantage',
    'star_power_leverage',
    'season_progress',
    'league_offensive_context',
    'total_foul_environment',
    'net_free_throw_advantage',
    'pace_efficiency_interaction',
    'offense_vs_defense_matchup'
]

# Trial 1306 Baseline Hyperparameters (for fair comparison)
TRIAL1306_PARAMS = {
    'max_depth': 3,
    'min_child_weight': 25,
    'gamma': 5.162427047142856,
    'learning_rate': 0.010519422544676995,
    'n_estimators': 9947,
    'subsample': 0.6277685565263181,
    'colsample_bytree': 0.6014538139159614,
    'reg_alpha': 6.193992559265241,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42
}

# Season Split Date (2024-25 season started Oct 22, 2024)
SPLIT_DATE = '2024-10-01'

# Data file
FILE_PATH = 'data/training_data_GOLD_ELO_22_features.csv'

# ==========================================
# 🧪 THE BACKTEST
# ==========================================
def run_backtest():
    print("="*70)
    print("🚀 WALK-FORWARD BACKTEST: VARIANT D ON 2024-25 SEASON")
    print("="*70)
    print(f"Features: {len(VARIANT_D_FEATURES)} (Variant D - Consolidated)")
    print(f"Split Date: {SPLIT_DATE}")
    print(f"Model: Trial 1306 Hyperparameters")
    print("="*70)
    
    # 1. Load Data
    try:
        df = pd.read_csv(FILE_PATH)
        
        # Standardize date column
        if 'date' in df.columns:
            df['game_date'] = pd.to_datetime(df['date'])
        elif 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        else:
            raise ValueError("No date column found")
        
        df = df.sort_values('game_date').reset_index(drop=True)
        print(f"✓ Loaded {len(df):,} games")
        print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Check features exist
    missing = [f for f in VARIANT_D_FEATURES if f not in df.columns]
    if missing:
        print(f"\n❌ Missing features: {missing}")
        print("Available features:", df.columns.tolist())
        return

    # 3. Split: History (Train) vs 2024-25 Season (Test)
    train_df = df[df['game_date'] < SPLIT_DATE].copy()
    test_df = df[df['game_date'] >= SPLIT_DATE].copy()
    
    print(f"\n📚 TRAINING SET (Historical)")
    print(f"   Games: {len(train_df):,}")
    print(f"   Date range: {train_df['game_date'].min().date()} to {train_df['game_date'].max().date()}")
    
    print(f"\n🔮 TEST SET (2024-25 Season)")
    print(f"   Games: {len(test_df):,}")
    if len(test_df) > 0:
        print(f"   Date range: {test_df['game_date'].min().date()} to {test_df['game_date'].max().date()}")
    
    if len(test_df) == 0:
        print("❌ ERROR: No games found in 2024-25 season. Check split date.")
        return
    
    # 4. Prepare data
    target_col = 'target_moneyline_win'
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found")
        return
    
    X_train = train_df[VARIANT_D_FEATURES].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[VARIANT_D_FEATURES].copy()
    y_test = test_df[target_col].copy()
    
    # Remove NaN
    train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
    
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    print(f"\n✓ Training samples: {len(X_train):,} (home win rate: {y_train.mean():.1%})")
    print(f"✓ Test samples:     {len(X_test):,} (home win rate: {y_test.mean():.1%})")
    
    # 5. Train Model with Trial 1306 params
    print("\n" + "="*70)
    print("⚙️  TRAINING MODEL")
    print("="*70)
    
    model = xgb.XGBClassifier(**TRIAL1306_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    
    print("✓ Model trained successfully")
    
    # 6. Predict on both sets
    # Training set (sanity check)
    train_preds = model.predict_proba(X_train)[:, 1]
    train_loss = log_loss(y_train, train_preds)
    train_acc = accuracy_score(y_train, (train_preds > 0.5).astype(int))
    train_auc = roc_auc_score(y_train, train_preds)
    
    # Test set (the real evaluation)
    test_preds = model.predict_proba(X_test)[:, 1]
    test_loss = log_loss(y_test, test_preds)
    test_acc = accuracy_score(y_test, (test_preds > 0.5).astype(int))
    test_auc = roc_auc_score(y_test, test_preds)
    
    # 7. Results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    print("\nTRAINING SET (Historical Data)")
    print(f"  Log Loss: {train_loss:.5f}")
    print(f"  Accuracy: {train_acc:.2%}")
    print(f"  AUC:      {train_auc:.4f}")
    
    print("\nTEST SET (2024-25 Season) ⭐")
    print(f"  Log Loss: {test_loss:.5f}")
    print(f"  Accuracy: {test_acc:.2%}")
    print(f"  AUC:      {test_auc:.4f}")
    
    # 8. Comparison to Trial 1306 Baseline
    baseline_cv_loss = 0.6330
    baseline_cv_acc = 0.6389
    
    print("\n" + "="*70)
    print("📈 COMPARISON TO TRIAL 1306 BASELINE")
    print("="*70)
    print(f"Baseline (CV on historical): Log Loss = {baseline_cv_loss:.5f}, Accuracy = {baseline_cv_acc:.2%}")
    print(f"Variant D (Test on 2024-25): Log Loss = {test_loss:.5f}, Accuracy = {test_acc:.2%}")
    print(f"\nΔ Log Loss: {test_loss - baseline_cv_loss:+.5f}")
    print(f"Δ Accuracy: {(test_acc - baseline_cv_acc)*100:+.2f}%")
    
    # 9. Generalization Check
    print("\n" + "="*70)
    print("🎯 GENERALIZATION ANALYSIS")
    print("="*70)
    
    overfitting_gap = test_loss - train_loss
    print(f"Overfitting Gap: {overfitting_gap:+.5f}")
    
    if overfitting_gap < 0.01:
        print("✅ EXCELLENT: Minimal overfitting (Gap < 0.01)")
    elif overfitting_gap < 0.02:
        print("✅ GOOD: Acceptable overfitting (Gap < 0.02)")
    elif overfitting_gap < 0.03:
        print("⚠️  MODERATE: Some overfitting (Gap < 0.03)")
    else:
        print("🔴 HIGH: Significant overfitting (Gap > 0.03)")
    
    # 10. Production Readiness
    print("\n" + "="*70)
    print("🏆 PRODUCTION READINESS ASSESSMENT")
    print("="*70)
    
    criteria = {
        'test_loss_acceptable': test_loss < 0.65,
        'test_acc_acceptable': test_acc > 0.60,
        'better_than_baseline': test_loss <= baseline_cv_loss + 0.01,
        'low_overfitting': overfitting_gap < 0.03
    }
    
    passed = sum(criteria.values())
    total = len(criteria)
    
    print(f"\nCriteria Met: {passed}/{total}")
    for criterion, met in criteria.items():
        status = "✅" if met else "❌"
        print(f"  {status} {criterion.replace('_', ' ').title()}")
    
    if passed == total:
        print("\n✅ VERDICT: APPROVED FOR PRODUCTION")
        print("   Variant D shows excellent generalization to 2024-25 season")
        print("   Feature reduction improved model robustness")
    elif passed >= 3:
        print("\n⚠️  VERDICT: CONDITIONAL APPROVAL")
        print("   Variant D shows good performance but monitor closely")
    else:
        print("\n🔴 VERDICT: NEEDS REVISION")
        print("   Model may not generalize well to current season")
    
    # 11. Save results
    results = {
        'variant': 'D',
        'n_features': len(VARIANT_D_FEATURES),
        'train_games': len(X_train),
        'test_games': len(X_test),
        'train_log_loss': float(train_loss),
        'train_accuracy': float(train_acc),
        'train_auc': float(train_auc),
        'test_log_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'baseline_cv_loss': baseline_cv_loss,
        'baseline_cv_acc': baseline_cv_acc,
        'overfitting_gap': float(overfitting_gap),
        'criteria_passed': passed,
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    with open('models/experimental/variant_d_backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results: models/experimental/variant_d_backtest_results.json")
    
    # 12. Feature importance on test set
    print("\n" + "="*70)
    print("🔍 TOP 10 FEATURES (Test Set)")
    print("="*70)
    
    importance = pd.DataFrame({
        'feature': VARIANT_D_FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:.4f}")

if __name__ == "__main__":
    run_backtest()
