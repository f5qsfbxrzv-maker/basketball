"""
🎯 ISOTONIC CALIBRATION TRAINER FOR MDP ENGINE
===============================================
Trains an isotonic regression calibrator to fix overconfidence at high edges.

Process:
1. Generate unbiased predictions via 5-fold CV
2. Train isotonic regressor: raw_prob -> actual_win_rate
3. Save calibrator for production use

This fixes the autopsy finding: 10%+ edges only winning ~48% (should be 55-65%)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
import joblib
import sys

# CONFIG
DATA_PATH = 'data/training_data_MDP_with_margins.csv'
CALIBRATOR_PATH = 'models/nba_mdp_isotonic.joblib'
NBA_STD_DEV = 13.42  # Your tuned RMSE

# Load Config features/params
try:
    from production_config_mdp import ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS
except ImportError:
    print("❌ Config missing. Ensure production_config_mdp.py exists.")
    sys.exit()

def train_calibrator():
    print("⚖️  TRAINING ISOTONIC CALIBRATOR FOR MDP ENGINE...")
    print("=" * 70)
    
    # 1. Load Data
    print(f"📂 Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"   ✓ Loaded {len(df):,} games")
    
    if 'margin_target' not in df.columns:
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['margin_target'] = df['home_score'] - df['away_score']
            print("   ✓ Created margin_target from scores")
        else:
            print("❌ Error: Need 'margin_target' or 'home_score'/'away_score' columns.")
            return

    X = df[ACTIVE_FEATURES]
    y_margin = df['margin_target']
    y_class = (df['margin_target'] > 0).astype(int)  # Win/Loss for calibration

    # 2. Generate Unbiased Predictions (Cross-Validation)
    # We need predictions on data the model hasn't seen to train the calibrator correctly.
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    raw_mdp_probs = np.zeros(len(df))
    
    print(f"\n🔄 Running 5-Fold CV to generate raw probabilities...")
    print("   (This ensures calibrator sees unbiased predictions)")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_margin)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_margin.iloc[train_idx], y_margin.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val)
        
        # Train MDP Regressor
        model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=False)
        
        # Predict Margin
        margins = model.predict(dval)
        
        # Convert to Prob (The "Raw" Prob using NBA_STD_DEV)
        probs = norm.cdf(margins / NBA_STD_DEV)
        raw_mdp_probs[val_idx] = probs
        
        print(f"   ✓ Fold {fold+1}/5: {len(val_idx):,} predictions generated")

    # 3. Train Isotonic Regression
    # Input: Raw MDP Probabilities -> Output: Actual Win Rates
    print("\n📈 Fitting Isotonic Regression...")
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso_reg.fit(raw_mdp_probs, y_class)
    print("   ✓ Isotonic regressor trained")
    
    # 4. Analyze the Correction
    print("\n🔍 CALIBRATION CORRECTION MAP")
    print("   (What the raw model says -> What the calibrator corrects to)")
    print("-" * 70)
    
    test_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    corrected = iso_reg.predict(test_points)
    
    for raw, cal in zip(test_points, corrected):
        diff = cal - raw
        arrow = "↑" if diff > 0 else "↓"
        if abs(diff) < 0.01: arrow = "→"
        status = "🚨" if abs(diff) > 0.05 else "✅" if abs(diff) < 0.02 else "⚠️"
        print(f"   Raw {raw:.0%} -> Calibrated {cal:.1%} ({diff:+.1%}) {arrow} {status}")

    # 5. Evaluate Improvement
    calibrated_probs = iso_reg.predict(raw_mdp_probs)
    
    raw_loss = log_loss(y_class, raw_mdp_probs)
    cal_loss = log_loss(y_class, calibrated_probs)
    raw_brier = brier_score_loss(y_class, raw_mdp_probs)
    cal_brier = brier_score_loss(y_class, calibrated_probs)
    
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE METRICS")
    print("-" * 70)
    print(f"Raw Log Loss:        {raw_loss:.5f}")
    print(f"Calibrated Log Loss: {cal_loss:.5f}")
    print(f"Improvement:         {(raw_loss - cal_loss)/raw_loss*100:+.2f}%")
    print()
    print(f"Raw Brier Score:     {raw_brier:.5f}")
    print(f"Calibrated Brier:    {cal_brier:.5f}")
    print(f"Improvement:         {(raw_brier - cal_brier)/raw_brier*100:+.2f}%")
    
    # 6. Calibration Buckets
    print("\n📊 CALIBRATION BY CONFIDENCE BUCKET")
    print("-" * 70)
    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in buckets:
        mask = (raw_mdp_probs >= low) & (raw_mdp_probs < high)
        if mask.sum() > 0:
            raw_acc = y_class[mask].mean()
            cal_acc = y_class[mask].mean()  # Compare same outcomes
            raw_pred = raw_mdp_probs[mask].mean()
            cal_pred = calibrated_probs[mask].mean()
            n = mask.sum()
            
            raw_err = abs(raw_pred - raw_acc) * 100
            cal_err = abs(cal_pred - cal_acc) * 100
            
            print(f"{low:.0%}-{high:.0%}: {n:4d} games | Actual: {raw_acc:.1%} | "
                  f"Raw: {raw_pred:.1%} (Err: {raw_err:.1f}%) | "
                  f"Cal: {cal_pred:.1%} (Err: {cal_err:.1f}%)")
    
    # 7. Save
    print("\n" + "=" * 70)
    joblib.dump(iso_reg, CALIBRATOR_PATH)
    print(f"💾 Calibrator saved to {CALIBRATOR_PATH}")
    print("=" * 70)
    print("✅ CALIBRATION TRAINING COMPLETE")
    print()
    print("🎯 Next Steps:")
    print("   1. Update production_config_mdp.py with zero-edge thresholds")
    print("   2. Apply calibrator in prediction pipeline: iso_reg.predict(raw_prob)")
    print("   3. Backtest with calibrated probabilities")

if __name__ == "__main__":
    train_calibrator()
