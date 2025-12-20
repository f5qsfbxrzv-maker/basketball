"""
ISOTONIC CALIBRATION TRAINER
============================
Generates out-of-sample predictions using K-Fold CV and trains isotonic regression
to map raw XGBoost probabilities to true win rates.

This fixes the overconfidence issue where the model predicts 70% but teams only win 48%.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import joblib
import sys
import os

# Import Ferrari Specs
try:
    from production_config import ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS, DATA_PATH, CALIBRATOR_PATH
except ImportError:
    print("❌ CRITICAL: production_config.py missing.")
    print("   Make sure production_config.py is in the same directory.")
    sys.exit(1)

def train_calibrator():
    print("="*80)
    print("⚖️  ISOTONIC CALIBRATION TRAINER")
    print("="*80)
    print("Goal: Fix model overconfidence by mapping raw probabilities to actual win rates")
    print("Method: K-Fold Cross-Validation for unbiased out-of-sample predictions")
    print("="*80)
    
    # 1. Load Data
    print(f"\n📥 Loading training data from {DATA_PATH}...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: {DATA_PATH} not found!")
        print(f"   Expected location: {os.path.abspath(DATA_PATH)}")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df):,} games")
    
    # Check for required columns
    if 'target_moneyline_win' not in df.columns:
        print("❌ ERROR: 'target_moneyline_win' column not found!")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    # Check for required features
    missing_features = [f for f in ACTIVE_FEATURES if f not in df.columns]
    if missing_features:
        print(f"❌ ERROR: Missing features: {missing_features}")
        return
    
    # Prepare features and target
    X = df[ACTIVE_FEATURES].copy()
    y = df['target_moneyline_win'].copy()
    
    # Remove NaN rows
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"✓ Clean data: {len(X):,} games with {len(ACTIVE_FEATURES)} features")
    
    # 2. Generate Out-of-Fold Predictions (The "Unbiased" View)
    # We use 5-Fold Cross Validation to predict every game in history "blind".
    # This prevents overfitting the calibrator to the training data.
    
    print(f"\n🔄 Running 5-Fold Cross-Validation...")
    print("   Generating out-of-sample predictions for entire history...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    raw_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n   Fold {fold+1}/5:")
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        # Train XGBoost model for this fold
        model = xgb.XGBClassifier(**XGB_PARAMS, n_estimators=N_ESTIMATORS)
        model.fit(X_train, y_train, verbose=False)
        
        # Predict on validation fold (these games were NOT in training)
        raw_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        print(f"      ✓ Trained on {len(train_idx):,} games, predicted on {len(val_idx):,} games")

    print(f"\n✓ Generated unbiased predictions for all {len(raw_preds):,} games")
    
    # 3. Train Isotonic Regression
    # Maps raw XGBoost probability → true win rate
    print("\n📈 Fitting Isotonic Regression...")
    print("   Learning the correction mapping...")
    
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(raw_preds, y)
    
    print("✓ Isotonic calibrator trained")
    
    # 4. Apply Calibration and Measure Improvement
    calibrated_preds = iso_reg.predict(raw_preds)
    
    # Calculate metrics
    brier_raw = brier_score_loss(y, raw_preds)
    brier_cal = brier_score_loss(y, calibrated_preds)
    logloss_raw = log_loss(y, raw_preds)
    logloss_cal = log_loss(y, calibrated_preds)
    
    # Calculate calibration curves (10 bins)
    prob_true_raw, prob_pred_raw = calibration_curve(y, raw_preds, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y, calibrated_preds, n_bins=10)
    
    # 5. Display Results
    print("\n" + "="*80)
    print("🏆 CALIBRATION RESULTS")
    print("="*80)
    
    print("\n📊 Performance Metrics:")
    print(f"   Raw Brier Score:        {brier_raw:.6f}")
    print(f"   Calibrated Brier Score: {brier_cal:.6f}")
    print(f"   Improvement:            {(brier_raw - brier_cal)/brier_raw*100:.2f}%")
    print()
    print(f"   Raw Log Loss:           {logloss_raw:.6f}")
    print(f"   Calibrated Log Loss:    {logloss_cal:.6f}")
    print(f"   Improvement:            {(logloss_raw - logloss_cal)/logloss_raw*100:.2f}%")
    
    # 6. Show Correction Mapping
    print("\n" + "-"*80)
    print("🔍 CORRECTION MAPPING (The 'S' Curve)")
    print("-"*80)
    print("   Raw Input → Calibrated Output (Difference)")
    print()
    
    test_probs = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    corrected = iso_reg.predict(test_probs)
    
    for raw, cal in zip(test_probs, corrected):
        diff = cal - raw
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        color = "🟢" if abs(diff) < 0.02 else "🟡" if abs(diff) < 0.05 else "🔴"
        print(f"   {color} {raw:.0%} → {cal:.1%} ({diff:+.1%}) {arrow}")
    
    print()
    if max(abs(corrected - test_probs)) > 0.10:
        print("⚠️  LARGE CORRECTIONS DETECTED (>10%)")
        print("   Model has significant calibration issues - this fix is critical!")
    elif max(abs(corrected - test_probs)) > 0.05:
        print("⚠️  MODERATE CORRECTIONS (5-10%)")
        print("   Calibration will meaningfully improve betting accuracy")
    else:
        print("✅ SMALL CORRECTIONS (<5%)")
        print("   Model is relatively well-calibrated, but still benefits from tuning")
    
    # 7. Show Calibration Quality by Bin
    print("\n" + "-"*80)
    print("📊 CALIBRATION BY PROBABILITY BIN")
    print("-"*80)
    print(f"{'Predicted':<12} {'Actual (Raw)':<15} {'Actual (Cal)':<15} {'Games':<8}")
    print("-"*80)
    
    # Bin both raw and calibrated predictions
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    df_eval = pd.DataFrame({
        'raw_prob': raw_preds,
        'cal_prob': calibrated_preds,
        'actual': y
    })
    
    df_eval['bin'] = pd.cut(df_eval['cal_prob'], bins=bins)
    
    for bin_range, group in df_eval.groupby('bin', observed=False):
        if len(group) == 0:
            continue
        avg_pred = group['cal_prob'].mean()
        actual_raw = group['actual'].mean()
        count = len(group)
        print(f"{bin_range!s:<12} {actual_raw:.1%}             {avg_pred:.1%}             {count:<8}")
    
    # 8. Save Calibrator
    print("\n" + "="*80)
    print("💾 SAVING CALIBRATOR")
    print("="*80)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(CALIBRATOR_PATH), exist_ok=True)
    
    joblib.dump(iso_reg, CALIBRATOR_PATH)
    print(f"✓ Saved to: {CALIBRATOR_PATH}")
    
    # 9. Create visualization (save to file)
    print("\n📊 Generating calibration curve visualization...")
    
    plt.figure(figsize=(10, 8))
    
    # Plot perfect calibration
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot raw model
    plt.plot(prob_pred_raw, prob_true_raw, 's-', label='Raw XGBoost', 
             color='#e74c3c', linewidth=2, markersize=8)
    
    # Plot calibrated model
    plt.plot(prob_pred_cal, prob_true_cal, 'o-', label='Isotonic Calibrated', 
             color='#2ecc71', linewidth=2, markersize=8)
    
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Actual Win Rate', fontsize=12)
    plt.title('Model Calibration: Raw vs Isotonic Corrected', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = 'models/calibration_curve.png'
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Saved calibration curve: {plot_path}")
    
    print("\n" + "="*80)
    print("✅ CALIBRATION COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review the correction mapping above")
    print("2. Check calibration_curve.png for visual validation")
    print("3. Run optimize_edge_calibrated.py to find new betting thresholds")
    print("4. Update production_config.py with the optimal thresholds")
    print("5. Deploy to daily picks!")
    print("="*80)

if __name__ == "__main__":
    train_calibrator()
