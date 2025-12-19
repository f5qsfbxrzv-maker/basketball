"""
PROPER COMPARISON: Trial 1306 hyperparameters on Gold Standard ELO data
Same 22 features, same model, only difference is ELO calculation method
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
import json

# Trial 1306 exact hyperparameters
TRIAL_1306_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 25,
    "gamma": 5.162427047142856,
    "learning_rate": 0.010519422544676995,
    "n_estimators": 9947,
    "subsample": 0.6277685565263181,
    "colsample_bytree": 0.6014538139159614,
    "reg_alpha": 6.193992559265241,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42
}

TRIAL_1306_FEATURES = [
    "home_composite_elo",
    "away_composite_elo",
    "off_elo_diff",
    "def_elo_diff",
    "ewma_efg_diff",
    "ewma_pace_diff",
    "ewma_tov_diff",
    "ewma_orb_diff",
    "ewma_vol_3p_diff",
    "ewma_chaos_home",
    "injury_matchup_advantage",
    "net_fatigue_score",
    "ewma_foul_synergy_home",
    "total_foul_environment",
    "league_offensive_context",
    "season_progress",
    "pace_efficiency_interaction",
    "projected_possession_margin",
    "three_point_matchup",
    "net_free_throw_advantage",
    "star_power_leverage",
    "offense_vs_defense_matchup"
]

print("=" * 80)
print("APPLES-TO-APPLES COMPARISON")
print("=" * 80)
print("\nTrial 1306 Hyperparameters on Gold Standard ELO Data\n")

# Load data
print("1. Loading Gold Standard ELO training data...")
df = pd.read_csv(r"data\training_data_GOLD_ELO_22_features.csv")
print(f"   Total games: {len(df)}")

# Prepare features and target
X = df[TRIAL_1306_FEATURES].values
y = df['target_moneyline_win'].values

print(f"\n2. Feature verification:")
print(f"   Features: {len(TRIAL_1306_FEATURES)}")
print(f"   Samples: {len(X)}")
print(f"   Target distribution: {y.mean():.3f} home wins")

# Time-series split for validation
print(f"\n3. Training with TimeSeriesSplit (5 folds)...")
tscv = TimeSeriesSplit(n_splits=5)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"\n   Fold {fold}/5:")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"      Train: {len(X_train)} games, Val: {len(X_val)} games")
    
    model = xgb.XGBClassifier(**TRIAL_1306_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    ll = log_loss(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"      Log-Loss: {ll:.4f}")
    print(f"      Accuracy: {acc:.1%}")
    
    fold_results.append({
        'fold': fold,
        'log_loss': ll,
        'accuracy': acc
    })

# Average results
avg_ll = np.mean([r['log_loss'] for r in fold_results])
avg_acc = np.mean([r['accuracy'] for r in fold_results])

print(f"\n4. CROSS-VALIDATION RESULTS")
print("-" * 80)
print(f"   Average Log-Loss: {avg_ll:.4f}")
print(f"   Average Accuracy: {avg_acc:.1%}")

# Train final model on all data
print(f"\n5. Training final model on all data...")
final_model = xgb.XGBClassifier(**TRIAL_1306_PARAMS)
final_model.fit(X, y, verbose=False)

# Final predictions
y_pred_proba_final = final_model.predict_proba(X)[:, 1]
train_ll = log_loss(y, y_pred_proba_final)
train_acc = accuracy_score(y, (y_pred_proba_final > 0.5).astype(int))

print(f"   Train Log-Loss: {train_ll:.4f}")
print(f"   Train Accuracy: {train_acc:.1%}")

# Feature importance
feature_importance = final_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': TRIAL_1306_FEATURES,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n6. FEATURE IMPORTANCE")
print("-" * 80)
for i, row in importance_df.head(10).iterrows():
    print(f"   {row['feature']:30s} {row['importance']:.4f}")

# Save model
final_model.save_model('models/trial1306_gold_elo.json')
print(f"\n7. Model saved to: models/trial1306_gold_elo.json")

# Save results
results = {
    'model': 'trial1306_gold_elo',
    'hyperparameters': TRIAL_1306_PARAMS,
    'features': TRIAL_1306_FEATURES,
    'cv_log_loss': float(avg_ll),
    'cv_accuracy': float(avg_acc),
    'train_log_loss': float(train_ll),
    'train_accuracy': float(train_acc),
    'n_samples': len(X),
    'fold_results': fold_results,
    'feature_importance': importance_df.to_dict('records')
}

with open('models/trial1306_gold_elo_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"   Results saved to: models/trial1306_gold_elo_results.json")

# THE MOMENT OF TRUTH
print("\n" + "=" * 80)
print("THE VERDICT")
print("=" * 80)
print(f"""
Old Model (Trial 1306) on OLD ELO (K=32):
   Validation Log-Loss: 0.6222
   Training Log-Loss: 0.5978

New Model (Trial 1306) on GOLD ELO (K=15):
   Validation Log-Loss: {avg_ll:.4f}
   Training Log-Loss: {train_ll:.4f}

Difference: {avg_ll - 0.6222:.4f}
""")

if avg_ll < 0.6222:
    improvement = (0.6222 - avg_ll) / 0.6222 * 100
    print(f"✓ GOLD STANDARD IS BETTER by {improvement:.1f}%!")
    print("  The new ELO system (K=15, WIN_WEIGHT=30) improves predictions")
    print("  even though it correctly ranks Brooklyn lower.")
elif avg_ll > 0.6222:
    degradation = (avg_ll - 0.6222) / 0.6222 * 100
    print(f"✗ OLD SYSTEM WAS BETTER by {degradation:.1f}%")
    print("  The old ELO (K=32) had better predictive power")
    print("  despite the Brooklyn ghost ranking issue.")
else:
    print("= EXACTLY THE SAME performance")
    print("  The ELO change didn't affect predictive accuracy.")

print("\nInterpretation:")
if avg_ll < 0.6222:
    print("   The Gold Standard fixes the ranking issue WITHOUT hurting predictions.")
    print("   This is the best outcome - more accurate ELO AND better predictions.")
else:
    print("   Trade-off detected: Better rankings but worse predictions.")
    print("   You may need to decide: accurate rankings vs predictive power?")
