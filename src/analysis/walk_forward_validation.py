"""
RIGOROUS WALK-FORWARD VALIDATION (Time Series Split)

This script simulates EXACTLY what would happen if you deployed the model
and retrained it every day/week for the 2023-24 season.

CRITICAL RULES:
1. Train on ALL games BEFORE date X
2. Predict ONLY games ON date X  
3. Move forward in time (never look ahead)
4. Features MUST be calculated as-of prediction date (no future peeking)

If accuracy > 75% after this, the leak is in FEATURE CONSTRUCTION.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import pickle
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.append('src')


def load_training_data(data_path: str) -> pd.DataFrame:
    """Load and prepare the full dataset"""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Find date column (could be 'game_date' or 'date')
    date_col = None
    for col in ['game_date', 'date', 'Date', 'GAME_DATE']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError(f"No date column found! Columns: {df.columns.tolist()[:10]}")
    
    # Rename to standard 'game_date'
    if date_col != 'game_date':
        df = df.rename(columns={date_col: 'game_date'})
    
    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # CRITICAL: Sort by date to respect time
    df = df.sort_values('game_date').reset_index(drop=True)
    
    print(f"  Loaded {len(df)} games")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get feature columns, EXCLUDING any potential leakers.
    
    EXCLUDED:
    - Outcomes: home_win, home_points, away_points, point_diff
    - Game identifiers: game_id, game_date, home_team, away_team, season
    - Suspicious: plus_minus, net_rating_game (could contain outcome data)
    """
    exclude_patterns = [
        'home_win', 'home_points', 'away_points', 'point_diff',
        'game_id', 'game_date', 'home_team', 'away_team', 'season',
        'plus_minus', 'net_rating_game', 'final_margin',
        'outcome', 'result', 'winner', 'loser',
        'target_'  # Exclude ALL target columns (these are outcomes!)
    ]
    
    feature_cols = []
    for col in df.columns:
        # Skip if matches any exclude pattern
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
        # Skip if non-numeric
        if df[col].dtype not in ['int64', 'float64']:
            continue
        feature_cols.append(col)
    
    print(f"\n[OK] Selected {len(feature_cols)} feature columns")
    print(f"First 10: {feature_cols[:10]}")
    
    return feature_cols


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'home_win',
    start_date: str = '2023-10-24',
    retrain_frequency: int = 7,  # Retrain every N days (1=daily, 7=weekly)
    min_train_samples: int = 1000
):
    """
    Expanding window walk-forward validation.
    
    Args:
        df: Full dataset (sorted by date)
        feature_cols: List of feature column names
        target_col: Target variable name
        start_date: When to start testing (train on all data before this)
        retrain_frequency: How often to retrain (1=daily, 7=weekly)
        min_train_samples: Minimum training samples required
    
    Returns:
        DataFrame with predictions and actual outcomes
    """
    start_date = pd.to_datetime(start_date)
    end_date = df['game_date'].max()
    
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD VALIDATION")
    print(f"{'='*70}")
    print(f"Training Start: {df['game_date'].min()}")
    print(f"Testing Start:  {start_date}")
    print(f"Testing End:    {end_date}")
    print(f"Retrain Freq:   Every {retrain_frequency} days")
    print(f"{'='*70}\n")
    
    # Get unique test dates
    test_dates = df[df['game_date'] >= start_date]['game_date'].unique()
    test_dates = sorted(test_dates)
    
    print(f"Total test dates: {len(test_dates)}")
    
    all_predictions = []
    model = None
    last_train_date = None
    
    for i, current_date in enumerate(test_dates):
        # A. SPLIT: Train on EVERYTHING before current_date
        train_mask = df['game_date'] < current_date
        test_mask = df['game_date'] == current_date
        
        train_data = df[train_mask]
        test_data = df[test_mask]
        
        # Skip if not enough training data
        if len(train_data) < min_train_samples:
            print(f"  [{i+1}/{len(test_dates)}] {current_date}: SKIP (not enough training data)")
            continue
        
        # Skip if no test games on this date
        if len(test_data) == 0:
            continue
        
        # B. RETRAIN MODEL (if needed)
        should_retrain = (
            model is None or 
            last_train_date is None or
            (current_date - last_train_date).days >= retrain_frequency
        )
        
        if should_retrain:
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            
            # Use Optuna best params (from your previous optimization)
            model = xgb.XGBClassifier(
                max_depth=11,
                min_child_weight=43,
                subsample=0.59,
                colsample_bytree=0.94,
                colsample_bylevel=0.86,
                colsample_bynode=0.73,
                learning_rate=0.003,
                n_estimators=3731,
                gamma=3.85,
                reg_alpha=9.60,
                reg_lambda=0.89,
                scale_pos_weight=1.04,
                max_delta_step=2,
                random_state=42,
                n_jobs=-1
            )
            
            print(f"  🔄 RETRAINING on {len(train_data)} games (up to {current_date})")
            model.fit(X_train, y_train, verbose=False)
            last_train_date = current_date
        
        # C. PREDICT on current_date ONLY
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        preds_proba = model.predict_proba(X_test)[:, 1]
        
        # D. STORE RESULTS
        results = test_data[['game_date', 'home_team', 'away_team', target_col]].copy()
        results['model_prob'] = preds_proba
        results['model_pred'] = (preds_proba > 0.5).astype(int)
        
        all_predictions.append(results)
        
        # Progress update
        acc_today = accuracy_score(y_test, (preds_proba > 0.5).astype(int))
        print(f"  [{i+1}/{len(test_dates)}] {current_date}: {len(test_data)} games | Acc: {acc_today:.3f}")
    
    # Combine all predictions
    full_results = pd.concat(all_predictions, ignore_index=True)
    
    return full_results


def analyze_results(results: pd.DataFrame, target_col: str = 'home_win'):
    """Analyze walk-forward results and check for leakage"""
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD RESULTS (TRUE OUT-OF-TIME VALIDATION)")
    print(f"{'='*70}")
    
    # Overall metrics
    acc = accuracy_score(results[target_col], results['model_pred'])
    ll = log_loss(results[target_col], results['model_prob'])
    brier = brier_score_loss(results[target_col], results['model_prob'])
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Total Games:     {len(results)}")
    print(f"  Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Log Loss:        {ll:.4f}")
    print(f"  Brier Score:     {brier:.4f}")
    
    # Leakage detection
    print(f"\n🔍 LEAKAGE DETECTION:")
    if acc > 0.75:
        print(f"  🚨 CRITICAL: Accuracy > 75%!")
        print(f"  This is SUSPICIOUSLY HIGH for NBA betting.")
        print(f"  Leak likely exists in FEATURE CONSTRUCTION.")
        print(f"\n  Possible causes:")
        print(f"    1. EWMA features not properly shifted (using current game)")
        print(f"    2. Season averages include future games")
        print(f"    3. Opponent stats include current matchup")
        print(f"    4. Injury data includes late scratches after feature calc")
    elif acc > 0.65:
        print(f"  ⚠️  WARNING: Accuracy {acc:.1%} is high but possible.")
        print(f"  Verify EWMA shift logic and feature calculation dates.")
    else:
        print(f"  ✅ Accuracy {acc:.1%} is realistic for NBA.")
        print(f"  No obvious data leakage detected.")
    
    # Monthly breakdown
    results['month'] = pd.to_datetime(results['game_date']).dt.to_period('M')
    monthly = results.groupby('month').apply(
        lambda x: pd.Series({
            'games': len(x),
            'accuracy': accuracy_score(x[target_col], x['model_pred']),
            'log_loss': log_loss(x[target_col], x['model_prob']) if len(x) > 0 else np.nan
        })
    )
    
    print(f"\n📅 MONTHLY BREAKDOWN:")
    print(monthly.to_string())
    
    # Save results
    output_path = 'output/walk_forward_results.csv'
    results.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    return {
        'accuracy': acc,
        'log_loss': ll,
        'brier_score': brier,
        'total_games': len(results)
    }


def main():
    """Run rigorous walk-forward validation"""
    
    # Load data - try multiple possible paths
    possible_paths = [
        'data/training_data_with_features.csv',
        'data/processed/training_features_30.csv',
        'data/training_data_full.csv'
    ]
    
    data_path = None
    for path in possible_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if data_path is None:
        print(f"❌ ERROR: No training data found in:")
        for path in possible_paths:
            print(f"  - {path}")
        print(f"\nPlease generate training data first or update paths.")
        return
    
    df = load_training_data(data_path)
    
    # Get feature columns (excluding leakers)
    feature_cols = get_feature_columns(df)
    
    # Run walk-forward validation
    results = walk_forward_validation(
        df=df,
        feature_cols=feature_cols,
        target_col='target_moneyline_win',
        start_date='2023-10-24',  # Start of 2023-24 season
        retrain_frequency=7,      # Retrain weekly (faster than daily)
        min_train_samples=1000
    )
    
    # Analyze results
    metrics = analyze_results(results, target_col='target_moneyline_win')
    
    # Final verdict
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT")
    print(f"{'='*70}")
    
    if metrics['accuracy'] > 0.75:
        print(f"❌ LEAKAGE DETECTED")
        print(f"\nYou must rebuild features with strict temporal separation:")
        print(f"  1. Use .shift(1) before any rolling calculations")
        print(f"  2. Verify EWMA uses: df[df['game_date'] < as_of_date]")
        print(f"  3. Check injury data is as-of game_date (not retroactive)")
        print(f"  4. Ensure no 'future' opponent stats leak into features")
    elif metrics['accuracy'] > 0.65:
        print(f"⚠️  HIGH BUT POSSIBLE")
        print(f"\nDouble-check feature construction logic, but this could be legitimate.")
    else:
        print(f"✅ CLEAN MODEL")
        print(f"\nAccuracy {metrics['accuracy']:.1%} is realistic for NBA prediction.")
        print(f"Proceed to Kelly criterion backtest with confidence!")


if __name__ == "__main__":
    main()
