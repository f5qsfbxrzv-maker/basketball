"""
Optimize Injury Feature Weights using Logistic Regression
===========================================================

This script finds the optimal weights for combining injury features into
a single comprehensive 'injury_matchup_advantage' metric.

Usage:
    python scripts/optimize_injury_weights.py

Output:
    - Optimal coefficients for injury_impact_diff, injury_shock_diff, star_mismatch
    - Formula to add to feature_calculator_v5.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = PROJECT_ROOT / 'data' / 'training_data_with_injury_shock.csv'
TARGET_COLUMN = 'target_moneyline_win'  # Home team win indicator
STAR_THRESHOLD = 4.0  # PIE threshold for star player

def load_and_prepare_data():
    """Load training data and construct raw injury components"""
    print("=" * 80)
    print("🧪 INJURY WEIGHT OPTIMIZATION")
    print("=" * 80)
    print()
    print(f"📂 Loading data from: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        print(f"❌ Error: Data file not found: {DATA_PATH}")
        print("\nExpected columns:")
        print("  - home_injury_impact, away_injury_impact")
        print("  - injury_shock_home, injury_shock_away")
        print("  - home_star_missing, away_star_missing")
        print("  - home_win (target)")
        return None
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} games")
    print()
    
    return df

def construct_injury_features(df):
    """Construct the raw injury components from data"""
    print("📊 Constructing Raw Injury Signals...")
    print()
    
    features_needed = [
        'injury_impact_diff',
        'injury_shock_diff', 
        'star_mismatch',
        TARGET_COLUMN
    ]
    
    # Check if all features exist
    missing = [f for f in features_needed if f not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        print("\nAttempting to construct from available columns...")
        
        # Try to construct from alternative column names
        if 'injury_impact_diff' in df.columns:
            print("   Found: injury_impact_diff")
        if 'injury_shock_diff' in df.columns:
            print("   Found: injury_shock_diff")
        if 'star_mismatch' in df.columns:
            print("   Found: star_mismatch")
            
        # If we have the diff columns, we can work with those directly
        if all(col in df.columns for col in ['injury_impact_diff', 'injury_shock_diff', 'star_mismatch', TARGET_COLUMN]):
            print("✅ Using pre-calculated differentials")
            return df[['injury_impact_diff', 'injury_shock_diff', 'star_mismatch', TARGET_COLUMN]].copy()
    
    # Calculate differentials
    X_data = pd.DataFrame()
    
    # 1. Baseline PIE Differential
    if 'injury_impact_diff' in df.columns:
        X_data['injury_impact_diff'] = df['injury_impact_diff']
    else:
        X_data['injury_impact_diff'] = df['home_injury_impact'] - df['away_injury_impact']
    
    # 2. Shock Differential (New News)
    if 'injury_shock_diff' in df.columns:
        X_data['injury_shock_diff'] = df['injury_shock_diff']
    else:
        X_data['injury_shock_diff'] = df['injury_shock_home'] - df['injury_shock_away']
    
    # 3. Star Mismatch (Binary)
    if 'star_mismatch' in df.columns:
        X_data['star_mismatch'] = df['star_mismatch']
    else:
        X_data['star_mismatch'] = df['home_star_missing'] - df['away_star_missing']
    
    X_data['home_win'] = df[TARGET_COLUMN]
    
    return X_data

def optimize_weights(data):
    """Run logistic regression to find optimal weights"""
    print("🧮 Running Logistic Regression (No Regularization)...")
    print()
    
    # Prepare features and target
    features = ['injury_impact_diff', 'injury_shock_diff', 'star_mismatch']
    X = data[features].fillna(0)
    y = data['home_win']
    
    # Check for variance
    print("Feature Statistics:")
    print(X.describe())
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit logistic regression with NO regularization (penalty=None)
    # This gives us the raw, unbiased coefficients
    model = LogisticRegression(penalty=None, fit_intercept=True, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("=" * 80)
    print("🏆 OPTIMIZATION RESULTS")
    print("=" * 80)
    print()
    print(f"Test AUC: {auc:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print()
    
    # Extract coefficients
    coeffs = model.coef_[0]
    
    print("Raw Coefficients:")
    print("-" * 40)
    for feat, coeff in zip(features, coeffs):
        print(f"{feat:25s}: {coeff:+.6f}")
    print()
    
    # Normalize to make baseline = 1.0 for readability
    base_w = abs(coeffs[0]) if abs(coeffs[0]) > 0 else 1.0
    
    print("Relative Importance (Normalized to Baseline = 1.0):")
    print("-" * 40)
    print(f"Baseline (PIE Diff):     1.00x")
    print(f"Shock (News):            {abs(coeffs[1]/base_w):.2f}x")
    print(f"Star Mismatch:           {abs(coeffs[2]/base_w):.2f}x")
    print()
    
    # Generate formula
    print("=" * 80)
    print("📋 COPY/PASTE INTO FEATURE_CALCULATOR_V5.PY")
    print("=" * 80)
    print()
    print("# Optimized injury matchup formula (derived from logistic regression)")
    print("injury_matchup_advantage = (")
    print(f"    {coeffs[0]:.6f} * injury_impact_diff")
    print(f"  + {coeffs[1]:.6f} * injury_shock_diff")
    print(f"  + {coeffs[2]:.6f} * star_mismatch")
    print(")")
    print()
    print("# Alternative: Normalized weights (if you prefer 0-1 scale)")
    total_abs = sum(abs(c) for c in coeffs)
    w1, w2, w3 = [abs(c)/total_abs for c in coeffs]
    print(f"# injury_matchup_advantage = (")
    print(f"#     {w1:.4f} * injury_impact_diff")
    print(f"#   + {w2:.4f} * injury_shock_diff")
    print(f"#   + {w3:.4f} * star_mismatch")
    print(f"# )")
    print()
    
    return model, coeffs, auc

def main():
    """Main execution"""
    # Load data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Construct features
    data = construct_injury_features(df)
    if data is None:
        return
    
    # Optimize
    model, coeffs, auc = optimize_weights(data)
    
    print("=" * 80)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Copy the formula above into feature_calculator_v5.py")
    print("2. Retrain your 20-feature model (19 + injury_matchup_advantage)")
    print("3. Validate on test set")
    print()

if __name__ == "__main__":
    main()
