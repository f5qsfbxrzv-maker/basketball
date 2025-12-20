"""
⚖️ MODERN ERA ISOTONIC CALIBRATOR (2023-2025)
==============================================
Strategy: Train base model on history, but calibrate ONLY on modern NBA.

Hypothesis: The NBA has fundamentally changed (pace, variance, 3-point volume).
A calibrator trained on 2018 data is useless - it's calibrating a ghost.

Approach:
1. Base Model: Trained on all history before Oct 2023 (learns physics)
2. Calibrator: Trained ONLY on 2023-2025 (learns modern bias)

Goal: See "dampening" in 70-90% range (fixing overconfidence on "locks")
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
import joblib
import sys

# CONFIG
DATA_PATH = 'data/training_data_MDP_with_margins.csv'
CALIBRATOR_PATH = 'models/nba_modern_isotonic.joblib'
NBA_STD_DEV = 13.42 
MODERN_ERA_START = '2023-10-01'  # The cutoff for the "New NBA"

try:
    from production_config_mdp import ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS
except ImportError:
    print("❌ Config missing. Ensure production_config_mdp.py exists.")
    sys.exit()

def train_modern_calibrator():
    print("⚖️  TRAINING MODERN ERA ISOTONIC CALIBRATOR (2023-2025)...")
    print("=" * 70)
    print("Strategy: Base model on history, calibrator on modern NBA only")
    print("=" * 70)
    
    # 1. Load Data
    print(f"\n📂 Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['game_date'] = pd.to_datetime(df['date'])  # Use 'date' column
    print(f"   ✓ Loaded {len(df):,} games")
    
    if 'margin_target' not in df.columns:
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['margin_target'] = df['home_score'] - df['away_score']
            print("   ✓ Created margin_target from scores")
        else:
            print("❌ Error: Need 'margin_target' or 'home_score'/'away_score'")
            return

    X = df[ACTIVE_FEATURES]
    y_margin = df['margin_target']
    y_class = (df['margin_target'] > 0).astype(int)

    # 2. Split: Train Base Model on History vs Calibrate on Modern Era
    # We want the model to predict the Modern Era "Out of Sample"
    modern_cutoff = pd.Timestamp(MODERN_ERA_START)
    train_mask = df['game_date'] < modern_cutoff
    calib_mask = df['game_date'] >= modern_cutoff
    
    print(f"\n🔄 DATA SPLIT")
    print(f"   Base Model Training (History):    {train_mask.sum():,} games (before {MODERN_ERA_START})")
    print(f"   Calibrator Training (Modern Era): {calib_mask.sum():,} games (since {MODERN_ERA_START})")
    
    if calib_mask.sum() < 500:
        print("   ⚠️ WARNING: Less than 500 modern games. Calibrator may be unstable.")
    
    # 3. Train Base Model on Historical Data
    print(f"\n🏋️ Training base model on {train_mask.sum():,} historical games...")
    dtrain = xgb.DMatrix(X[train_mask], label=y_margin[train_mask])
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    print("   ✓ Base model trained")
    
    # 4. Generate Raw Predictions for Modern Era (Out-of-Sample)
    print(f"\n🔮 Generating out-of-sample predictions for modern era...")
    dcalib = xgb.DMatrix(X[calib_mask])
    pred_margins = model.predict(dcalib)
    raw_probs = norm.cdf(pred_margins / NBA_STD_DEV)
    actual_outcomes = y_class[calib_mask].values
    
    print(f"   ✓ {len(raw_probs):,} predictions generated")
    print(f"   Raw prob range: {raw_probs.min():.3f} to {raw_probs.max():.3f}")
    print(f"   Actual win rate: {actual_outcomes.mean():.3f}")
    
    # 5. Train Isotonic Regression (Modern Era Only)
    # Map Raw Prob -> Modern Reality
    print(f"\n📈 Fitting isotonic regression on modern era...")
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso_reg.fit(raw_probs, actual_outcomes)
    print("   ✓ Isotonic regressor trained")
    
    # 6. THE CORRECTION MAP (The Verdict)
    print("\n" + "=" * 70)
    print("🔍 MODERN ERA CORRECTION MAP")
    print("   (Goal: See 'DAMPENING' at high confidence to fix overconfidence)")
    print("=" * 70)
    
    # Check specifically the "Toxic" zones we found (70-90%)
    test_points = np.arange(0.50, 0.96, 0.05)
    corrected = iso_reg.predict(test_points)
    
    for raw, cal in zip(test_points, corrected):
        diff = cal - raw
        status = "✅ OK"
        arrow = "→"
        
        if diff > 0.03: 
            status = "⚠️ BOOSTING (Risky)"
            arrow = "↑"
        if diff < -0.03: 
            status = "🛡️ DAMPENING (Fixing Overconfidence)"
            arrow = "↓"
        
        print(f"   Input {raw:.0%} -> Output {cal:.1%} ({diff:+.1%}) {arrow} {status}")
    
    # 7. Detailed Calibration Buckets (Modern Era)
    print("\n" + "=" * 70)
    print("📊 CALIBRATION BY CONFIDENCE BUCKET (Modern Era)")
    print("=" * 70)
    
    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in buckets:
        mask = (raw_probs >= low) & (raw_probs < high)
        if mask.sum() > 0:
            actual_rate = actual_outcomes[mask].mean()
            raw_pred = raw_probs[mask].mean()
            cal_pred = iso_reg.predict(raw_probs[mask]).mean()
            n = mask.sum()
            
            raw_err = abs(raw_pred - actual_rate) * 100
            cal_err = abs(cal_pred - actual_rate) * 100
            improvement = raw_err - cal_err
            
            status = "✅" if cal_err < raw_err else "❌"
            
            print(f"{low:.0%}-{high:.0%}: {n:4d} games | Actual: {actual_rate:.1%}")
            print(f"       Raw: {raw_pred:.1%} (Err: {raw_err:+.1f}pp) | Cal: {cal_pred:.1%} (Err: {cal_err:+.1f}pp) | Δ: {improvement:+.1f}pp {status}")
    
    # 8. Evaluation
    calibrated_probs = iso_reg.predict(raw_probs)
    
    raw_loss = log_loss(actual_outcomes, raw_probs)
    cal_loss = log_loss(actual_outcomes, calibrated_probs)
    raw_brier = brier_score_loss(actual_outcomes, raw_probs)
    cal_brier = brier_score_loss(actual_outcomes, calibrated_probs)
    
    print("\n" + "=" * 70)
    print("📊 MODERN ERA PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Raw Log Loss:        {raw_loss:.5f}")
    print(f"Calibrated Log Loss: {cal_loss:.5f}")
    
    if cal_loss < raw_loss:
        improvement = (raw_loss - cal_loss) / raw_loss * 100
        print(f"✅ SUCCESS: Improved by {improvement:+.2f}%")
    else:
        worsening = (cal_loss - raw_loss) / raw_loss * 100
        print(f"❌ WORSENED: Degraded by {worsening:+.2f}%")
    
    print(f"\nRaw Brier Score:     {raw_brier:.5f}")
    print(f"Calibrated Brier:    {cal_brier:.5f}")
    
    if cal_brier < raw_brier:
        improvement = (raw_brier - cal_brier) / raw_brier * 100
        print(f"✅ SUCCESS: Improved by {improvement:+.2f}%")
    
    # 9. Decision: Save or Reject
    print("\n" + "=" * 70)
    
    if cal_loss < raw_loss:
        print("✅ CALIBRATION SUCCESSFUL - Saving calibrator")
        joblib.dump(iso_reg, CALIBRATOR_PATH)
        print(f"💾 Saved to {CALIBRATOR_PATH}")
        print("=" * 70)
        print("\n🎯 Next Steps:")
        print("   1. Run backtest with modern calibrator")
        print("   2. Check if 70-90% dampening translates to profit")
        print("   3. If successful, deploy zero-edge strategy with modern calibrator")
    else:
        print("❌ CALIBRATION FAILED - Model does NOT improve on modern era")
        print("   Modern NBA patterns may already be captured in base model")
        print("   Recommendation: Stick with optimized thresholds (1.5% fav / 8.0% dog)")
        print("=" * 70)

if __name__ == "__main__":
    train_modern_calibrator()
