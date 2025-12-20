"""
🎯 BACKTEST: ISOTONIC-CALIBRATED MDP WITH ZERO-EDGE THRESHOLDS
================================================================
Tests the isotonic calibration fix for high-edge overconfidence.

Strategy Changes:
- Zero-edge thresholds (0.1% for both fav/dog)
- Isotonic calibration applied to all predictions
- Removed pricing filters (MAX_FAV_ODDS)
- Kept only physics filter (FILTER_MIN_OFF_ELO = -90)

Hypothesis: Isotonic calibration fixes the 10%+ edge underperformance
observed in autopsy (48-50% win rate when expecting 55-65%)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
import joblib
from datetime import datetime

# Import config
from production_config_mdp import (
    ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS, NBA_STD_DEV,
    MIN_EDGE_FAVORITE, MIN_EDGE_UNDERDOG, FILTER_MIN_OFF_ELO,
    CALIBRATOR_PATH
)

def american_to_prob(odds):
    """Convert American odds to implied probability (with vig)"""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def remove_vig(yes_price, no_price):
    """Remove vig to get fair probabilities"""
    yes_prob = american_to_prob(yes_price)
    no_prob = american_to_prob(no_price)
    total = yes_prob + no_prob
    
    if total > 1:  # Has vig
        return yes_prob / total
    return yes_prob  # Already fair

def calc_edge(model_prob, market_odds):
    """Calculate edge with vig removal"""
    # Get both sides odds (assume symmetric vig for now)
    home_implied = american_to_prob(market_odds)
    away_implied = 1 - home_implied
    
    # Simple vig adjustment
    total_implied = home_implied + away_implied
    if total_implied > 1:
        fair_prob = home_implied / total_implied
    else:
        fair_prob = home_implied
    
    return model_prob - fair_prob

def backtest_isotonic_zero_edge():
    print("🎯 BACKTESTING MDP WITH ISOTONIC CALIBRATION (ZERO-EDGE MODE)")
    print("=" * 80)
    
    # 1. Load Data
    print("📂 Loading data...")
    mdp_data = pd.read_csv('data/training_data_MDP_with_margins.csv')
    mdp_data['game_date'] = pd.to_datetime(mdp_data['date'])
    
    # Load odds
    odds_2024 = pd.read_csv('data/closing_odds_2024_25_CLEANED.csv')
    odds_2024['game_date'] = pd.to_datetime(odds_2024['game_date'])
    
    print(f"   ✓ MDP data: {len(mdp_data):,} games")
    print(f"   ✓ 2024-25 odds: {len(odds_2024):,} games")
    
    # 2. Load Isotonic Calibrator
    print(f"\n📈 Loading isotonic calibrator from {CALIBRATOR_PATH}...")
    try:
        iso_calibrator = joblib.load(CALIBRATOR_PATH)
        print("   ✓ Calibrator loaded")
    except FileNotFoundError:
        print("   ❌ Calibrator not found! Run train_mdp_isotonic.py first.")
        return
    
    # 3. Define Season Split
    season_start = pd.Timestamp('2024-10-22')
    
    # Training: All data before season start
    train_mask = mdp_data['game_date'] < season_start
    train_data = mdp_data[train_mask]
    
    # Test: 2024-25 season
    test_mask = mdp_data['game_date'] >= season_start
    test_data = mdp_data[test_mask]
    
    print(f"\n🔄 Training on {len(train_data):,} games before {season_start.date()}")
    print(f"🔄 Testing on {len(test_data):,} games from 2024-25 season")
    
    # 4. Train Model
    print("\n🏋️ Training MDP model...")
    X_train = train_data[ACTIVE_FEATURES]
    y_train = train_data['margin_target']
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    print("   ✓ Model trained")
    
    # 5. Generate Predictions
    print("\n🔮 Generating predictions...")
    X_test = test_data[ACTIVE_FEATURES]
    dtest = xgb.DMatrix(X_test)
    
    # Predict margins
    margins = model.predict(dtest)
    
    # Convert to raw probabilities
    raw_probs = norm.cdf(margins / NBA_STD_DEV)
    
    # Apply isotonic calibration
    calibrated_probs = iso_calibrator.predict(raw_probs)
    
    # Add to test data
    test_predictions = test_data.copy()
    test_predictions['predicted_margin'] = margins
    test_predictions['raw_prob'] = raw_probs
    test_predictions['calibrated_prob'] = calibrated_probs
    
    print(f"   ✓ {len(test_predictions):,} predictions generated")
    print(f"   Raw prob range: {raw_probs.min():.3f} to {raw_probs.max():.3f}")
    print(f"   Calibrated range: {calibrated_probs.min():.3f} to {calibrated_probs.max():.3f}")
    
    # 6. Merge with Odds
    print("\n🔗 Merging with closing odds...")
    merged = test_predictions.merge(
        odds_2024[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
        on=['game_date', 'home_team', 'away_team'],
        how='inner'
    )
    print(f"   ✓ Matched {len(merged):,} games with odds")
    
    # 7. Calculate Edges (using CALIBRATED probabilities)
    merged['market_prob'] = merged['home_ml_odds'].apply(american_to_prob)
    merged['edge'] = merged['calibrated_prob'] - merged['market_prob']
    merged['is_favorite'] = merged['home_ml_odds'] < 0
    
    # 8. Apply Filters
    print("\n🚫 Applying filters...")
    
    # Physics filter: Broken offense
    physics_filter = merged['off_elo_diff'] >= FILTER_MIN_OFF_ELO
    print(f"   Physics (off_elo > {FILTER_MIN_OFF_ELO}): {physics_filter.sum():,} pass")
    
    # Apply filter
    filtered = merged[physics_filter].copy()
    
    # 9. Identify Bets (Zero-Edge Mode)
    print(f"\n💰 Identifying bets (zero-edge mode)...")
    print(f"   MIN_EDGE_FAVORITE: {MIN_EDGE_FAVORITE:.3%}")
    print(f"   MIN_EDGE_UNDERDOG: {MIN_EDGE_UNDERDOG:.3%}")
    
    # Mark all bets first
    filtered['won'] = filtered['margin_target'] > 0
    
    # Calculate stake and profit for all
    filtered['stake'] = 1.0
    
    def calc_profit(row):
        if row['won']:
            if row['home_ml_odds'] < 0:
                return 100 / abs(row['home_ml_odds'])
            else:
                return row['home_ml_odds'] / 100
        else:
            return -1.0
    
    filtered['profit'] = filtered.apply(calc_profit, axis=1)
    
    # Now filter by edge
    favorite_bets = filtered[
        filtered['is_favorite'] & (filtered['edge'] >= MIN_EDGE_FAVORITE)
    ].copy()
    
    underdog_bets = filtered[
        ~filtered['is_favorite'] & (filtered['edge'] >= MIN_EDGE_UNDERDOG)
    ].copy()
    
    all_bets = pd.concat([favorite_bets, underdog_bets])
    
    print(f"   ✓ Favorites: {len(favorite_bets):,} bets")
    print(f"   ✓ Underdogs: {len(underdog_bets):,} bets")
    print(f"   ✓ Total: {len(all_bets):,} bets")
    
    # 10. Results already calculated above
    
    # 11. Print Results
    print("\n" + "=" * 80)
    print("📊 ISOTONIC ZERO-EDGE BACKTEST RESULTS (2024-25)")
    print("=" * 80)
    
    total_bets = len(all_bets)
    total_wins = all_bets['won'].sum()
    win_rate = total_wins / total_bets if total_bets > 0 else 0
    total_profit = all_bets['profit'].sum()
    total_stake = all_bets['stake'].sum()
    roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
    
    print(f"\n🏆 COMBINED RESULTS")
    print(f"Total Bets:   {total_bets:,}")
    print(f"Wins:         {total_wins:,}")
    print(f"Win Rate:     {win_rate:.1%}")
    print(f"Total Profit: {total_profit:+.2f}u")
    print(f"ROI:          {roi:+.1f}%")
    
    # Favorites
    fav_wins = favorite_bets['won'].sum()
    fav_profit = favorite_bets['profit'].sum()
    fav_roi = (fav_profit / len(favorite_bets) * 100) if len(favorite_bets) > 0 else 0
    
    print(f"\n🦁 FAVORITES")
    print(f"Bets:         {len(favorite_bets):,}")
    print(f"Win Rate:     {fav_wins / len(favorite_bets):.1%}" if len(favorite_bets) > 0 else "N/A")
    print(f"Profit:       {fav_profit:+.2f}u")
    print(f"ROI:          {fav_roi:+.1f}%")
    
    # Underdogs
    dog_wins = underdog_bets['won'].sum()
    dog_profit = underdog_bets['profit'].sum()
    dog_roi = (dog_profit / len(underdog_bets) * 100) if len(underdog_bets) > 0 else 0
    
    print(f"\n🐶 UNDERDOGS")
    print(f"Bets:         {len(underdog_bets):,}")
    print(f"Win Rate:     {dog_wins / len(underdog_bets):.1%}" if len(underdog_bets) > 0 else "N/A")
    print(f"Profit:       {dog_profit:+.2f}u")
    print(f"ROI:          {dog_roi:+.1f}%")
    
    # 12. Edge Bucket Analysis
    print("\n" + "=" * 80)
    print("📊 EDGE BUCKET ANALYSIS (Calibrated Edge)")
    print("=" * 80)
    
    edge_buckets = [
        (0.000, 0.025, "0-2.5%"),
        (0.025, 0.050, "2.5-5%"),
        (0.050, 0.100, "5-10%"),
        (0.100, 0.150, "10-15%"),
        (0.150, 1.000, "15%+")
    ]
    
    print(f"{'EDGE RANGE':<12} | {'BETS':<6} | {'WIN %':<8} | {'PROFIT':<10} | {'ROI':<8}")
    print("-" * 70)
    
    for low, high, label in edge_buckets:
        bucket = all_bets[(all_bets['edge'] >= low) & (all_bets['edge'] < high)]
        if len(bucket) > 0:
            wins = bucket['won'].sum()
            win_pct = wins / len(bucket) * 100
            profit = bucket['profit'].sum()
            roi = profit / len(bucket) * 100
            print(f"{label:<12} | {len(bucket):<6} | {win_pct:>6.1f}% | {profit:>+9.2f}u | {roi:>+6.1f}%")
    
    # 13. Compare Raw vs Calibrated
    print("\n" + "=" * 80)
    print("🔬 RAW vs CALIBRATED PROBABILITY ANALYSIS")
    print("=" * 80)
    
    all_bets['raw_edge'] = all_bets['raw_prob'] - all_bets['market_prob']
    
    print(f"\nAverage Raw Prob:        {all_bets['raw_prob'].mean():.3f}")
    print(f"Average Calibrated Prob: {all_bets['calibrated_prob'].mean():.3f}")
    print(f"Average Actual Win Rate: {all_bets['won'].mean():.3f}")
    
    print(f"\nAverage Raw Edge:        {all_bets['raw_edge'].mean():.4f} ({all_bets['raw_edge'].mean()*100:.2f}%)")
    print(f"Average Calibrated Edge: {all_bets['edge'].mean():.4f} ({all_bets['edge'].mean()*100:.2f}%)")
    
    # High confidence calibration check
    high_conf = all_bets[all_bets['calibrated_prob'] > 0.7]
    if len(high_conf) > 0:
        print(f"\n📊 HIGH CONFIDENCE (>70%) CHECK")
        print(f"Count:         {len(high_conf):,}")
        print(f"Raw Prob Avg:  {high_conf['raw_prob'].mean():.1%}")
        print(f"Cal Prob Avg:  {high_conf['calibrated_prob'].mean():.1%}")
        print(f"Actual WR:     {high_conf['won'].mean():.1%}")
        print(f"Profit:        {high_conf['profit'].sum():+.2f}u")
        print(f"ROI:           {(high_conf['profit'].sum() / len(high_conf) * 100):+.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETE")
    print("=" * 80)
    
    # Save detailed results
    output_file = 'results_isotonic_zero_edge_2024_25.csv'
    all_bets.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to {output_file}")

if __name__ == "__main__":
    backtest_isotonic_zero_edge()
