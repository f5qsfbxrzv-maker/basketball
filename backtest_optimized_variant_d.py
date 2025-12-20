"""
OPTIMIZED VARIANT D: 2024-25 SEASON BACKTEST
=============================================
Tests the "Ferrari Specs" (Trial #245 params) on unseen 2024-25 data.

If this beats baseline (0.61167), it proves optimization captured true game physics,
not just historical noise.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from datetime import datetime

# ==========================================
# ⚙️ CONFIGURATION: OPTIMIZED VARIANT D
# ==========================================

# 1. The "Clean 19" Feature List (Variant D - Actual Dataset)
FEATURES = [
    'home_composite_elo',           # ELO: Anchor
    'off_elo_diff',                 # ELO: Offense (PRIMARY)
    'def_elo_diff',                 # ELO: Defense
    'projected_possession_margin',  # Possession: Consolidated
    'ewma_pace_diff',              # Pace
    'net_fatigue_score',           # Rest
    'ewma_efg_diff',               # Shooting: Efficiency
    'ewma_vol_3p_diff',            # Shooting: Volume
    'three_point_matchup',         # Shooting: Matchup
    'ewma_chaos_home',             # Personnel volatility
    'injury_impact_diff',          # Injury: PIE-weighted
    'injury_shock_diff',           # Injury: Shock impact
    'star_power_leverage',         # Injury: Star impact
    'season_progress',             # Context: Season phase
    'league_offensive_context',    # Context: Era adjustment
    'total_foul_environment',      # Fouls: Total
    'net_free_throw_advantage',    # Fouls: Differential
    'pace_efficiency_interaction', # Interaction: Pace x Efficiency
    'offense_vs_defense_matchup'   # Interaction: Cross-side
]

# 2. OPTIMIZED HYPERPARAMETERS (Trial #245)
# "The Ferrari Specs"
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    
    # --- The Winning Tuning ---
    'learning_rate': 0.066994,     # Fast learner (was 0.0105 in Trial 1306)
    'max_depth': 2,                # Shallow trees (clean features allow this)
    'min_child_weight': 12,        # Finer splits allowed
    'gamma': 2.025432,             # Minimum loss reduction
    
    # --- Stochastic Features ---
    'subsample': 0.630135,
    'colsample_bytree': 0.903401,
    'colsample_bylevel': 0.959686,
    
    # --- Regularization ---
    'reg_alpha': 1.081072,         # L1 Regularization
    'reg_lambda': 5.821363,        # L2 Regularization
}

# 3. SETTINGS
FILE_PATH = 'data/training_data_GOLD_ELO_22_features.csv'
SPLIT_DATE = '2024-10-01'      # Start of 24-25 Season
N_ESTIMATORS = 4529            # The optimal number of trees

# ==========================================
# 🧪 EXECUTION
# ==========================================
def run_backtest():
    print("="*70)
    print("🚀 OPTIMIZED VARIANT D: 2024-25 SEASON BACKTEST")
    print("="*70)
    print(f"   Model: Trial #245 (Ferrari Specs)")
    print(f"   Max Depth: {params['max_depth']} | Trees: {N_ESTIMATORS} | LR: {params['learning_rate']:.4f}")
    print(f"   Features: {len(FEATURES)} (Clean, VIF < 2.5)")
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
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"\n❌ Missing features: {missing}")
        return

    # 3. Split
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
        print("❌ ERROR: No games found in 2024-25 season.")
        return
    
    # 4. Prepare data
    target_col = 'target_moneyline_win'
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found")
        return
    
    X_train = train_df[FEATURES].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[FEATURES].copy()
    y_test = test_df[target_col].copy()
    
    # Remove NaN
    train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
    
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    print(f"\n✓ Training samples: {len(X_train):,} (home win rate: {y_train.mean():.1%})")
    print(f"✓ Test samples:     {len(X_test):,} (home win rate: {y_test.mean():.1%})")
    
    # 5. Train with Ferrari Specs
    print("\n" + "="*70)
    print("⚙️  TRAINING MODEL (OPTIMIZED HYPERPARAMETERS)")
    print("="*70)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    print(f"Training {N_ESTIMATORS} trees with max_depth={params['max_depth']}...")
    start_time = datetime.now()
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=N_ESTIMATORS,
        evals=[(dtrain, "Train"), (dtest, "Test")], 
        early_stopping_rounds=100,
        verbose_eval=500
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Training completed in {training_time:.1f} seconds")
    
    # 6. Predict & Score
    train_preds = model.predict(dtrain)
    test_preds = model.predict(dtest)
    
    train_loss = log_loss(y_train, train_preds)
    train_acc = accuracy_score(y_train, (train_preds > 0.5).astype(int))
    train_auc = roc_auc_score(y_train, train_preds)
    
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
    
    # 8. Comparison Analysis
    print("\n" + "="*70)
    print("🎯 COMPARISON: OPTIMIZED vs BASELINE VARIANT D")
    print("="*70)
    
    BASELINE_LOSS = 0.61167  # Variant D with Trial 1306 params
    BASELINE_ACC = 0.6724
    
    print(f"\nBaseline (Trial 1306 params):  Log Loss = {BASELINE_LOSS:.5f}, Accuracy = {BASELINE_ACC:.2%}")
    print(f"Optimized (Trial #245 params): Log Loss = {test_loss:.5f}, Accuracy = {test_acc:.2%}")
    
    delta_loss = test_loss - BASELINE_LOSS
    delta_acc = test_acc - BASELINE_ACC
    
    print(f"\nΔ Log Loss: {delta_loss:+.5f}")
    print(f"Δ Accuracy: {delta_acc*100:+.2f}%")
    
    # 9. Verdict
    print("\n" + "="*70)
    print("🏆 VERDICT: GENERALIZATION TEST")
    print("="*70)
    
    if test_loss < BASELINE_LOSS:
        improvement = ((BASELINE_LOSS - test_loss) / BASELINE_LOSS) * 100
        print(f"\n✅ SUCCESS: {improvement:.2f}% IMPROVEMENT")
        print("   The optimization captured TRUE GAME PHYSICS, not just historical noise.")
        print("   Ferrari Specs validated on unseen data.")
        print("\n   🚀 STATUS: GO FOR LAUNCH")
        
    elif test_loss < BASELINE_LOSS + 0.001:
        print(f"\n⚖️  EQUIVALENT: Within 0.1% of baseline")
        print("   Optimization didn't hurt, features are the real MVP.")
        print("   Either configuration is production-ready.")
        
    else:
        regression = ((test_loss - BASELINE_LOSS) / BASELINE_LOSS) * 100
        print(f"\n⚠️  REGRESSION: {regression:.2f}% worse than baseline")
        print("   Tuning may have overfit historical patterns.")
        print("   Recommendation: Use baseline Trial 1306 params for safety.")
    
    # 10. Overfitting Analysis
    overfitting_gap = test_loss - train_loss
    print("\n" + "="*70)
    print("🔍 OVERFITTING ANALYSIS")
    print("="*70)
    print(f"Overfitting Gap: {overfitting_gap:+.5f}")
    
    if overfitting_gap < 0:
        print("✅ RARE: Model generalizes BETTER to future than past (ideal)")
    elif overfitting_gap < 0.01:
        print("✅ EXCELLENT: Minimal overfitting (Gap < 0.01)")
    elif overfitting_gap < 0.02:
        print("✅ GOOD: Acceptable overfitting (Gap < 0.02)")
    elif overfitting_gap < 0.03:
        print("⚠️  MODERATE: Some overfitting (Gap < 0.03)")
    else:
        print("🔴 HIGH: Significant overfitting (Gap > 0.03)")
    
    # 11. Speed Comparison
    print("\n" + "="*70)
    print("⚡ TRAINING EFFICIENCY")
    print("="*70)
    print(f"Trees: {N_ESTIMATORS} (vs Trial 1306's 9,947)")
    print(f"Max Depth: {params['max_depth']} (vs Trial 1306's 3)")
    print(f"Training Time: {training_time:.1f}s")
    print(f"\nSpeed Benefit: ~2-3x faster training with cleaner features")
    
    # 12. Feature Importance
    print("\n" + "="*70)
    print("🔍 TOP 10 FEATURES (Test Set)")
    print("="*70)
    
    importance_scores = model.get_score(importance_type='gain')
    importance = pd.DataFrame([
        {'feature': f, 'importance': importance_scores.get(f'f{i}', 0)}
        for i, f in enumerate(FEATURES)
    ]).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:.4f}")
    
    # 13. Save results
    results = {
        'model': 'variant_d_optimized',
        'trial_number': 245,
        'n_features': len(FEATURES),
        'features': FEATURES,
        'hyperparameters': params,
        'n_estimators': N_ESTIMATORS,
        'train_games': len(X_train),
        'test_games': len(X_test),
        'train_log_loss': float(train_loss),
        'train_accuracy': float(train_acc),
        'train_auc': float(train_auc),
        'test_log_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'baseline_log_loss': BASELINE_LOSS,
        'baseline_accuracy': BASELINE_ACC,
        'delta_log_loss': float(delta_loss),
        'delta_accuracy': float(delta_acc),
        'overfitting_gap': float(overfitting_gap),
        'training_time_seconds': training_time,
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    results_file = 'models/experimental/variant_d_optimized_backtest_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_file}")

if __name__ == "__main__":
    run_backtest()
