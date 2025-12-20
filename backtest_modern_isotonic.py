"""
🧪 BACKTEST: MODERN ISOTONIC CALIBRATION (2024-25)
===================================================
The Proof: Does the "boosting" behavior make money or lose it?

Strategy: Zero Edge Mode with Modern Era Calibrator
- Bet ANY game where Calibrated Prob > Implied Prob
- Trust the calibrator completely (no thresholds)

Benchmark: +79.18u from optimized thresholds (1.5% fav / 8.0% dog)

If Modern Isotonic beats this, the "boosting" was finding real signal.
If it loses money, we have definitive proof to abandon calibration.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
import joblib
import sys

# CONFIG
DATA_PATH = 'data/training_data_MDP_with_margins.csv'
ODDS_PATH = 'data/closing_odds_2024_25_CLEANED.csv'
CALIBRATOR_PATH = 'models/nba_modern_isotonic.joblib'
NBA_STD_DEV = 13.42
START_DATE_24 = '2024-10-22'

# Load Features/Params
try:
    from production_config_mdp import ACTIVE_FEATURES, XGB_PARAMS, N_ESTIMATORS
except ImportError:
    print("❌ Config missing. Ensure production_config_mdp.py exists.")
    sys.exit()

def american_to_prob(odds):
    """Convert American odds to implied probability"""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def american_to_payout(odds):
    """Convert American odds to payout multiplier"""
    if odds < 0:
        return 100 / abs(odds)
    else:
        return odds / 100

def run_backtest():
    print("🧪 BACKTESTING MODERN ISOTONIC CALIBRATION (2024-25)...")
    print("=" * 70)
    print("Strategy: Zero Edge (Trusting the Calibrated Probability)")
    print(f"Benchmark: +79.18u from optimized thresholds (1.5% fav / 8.0% dog)")
    print("=" * 70)

    # 1. Load Data
    print("\n📂 Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
        df['game_date'] = pd.to_datetime(df['date'])
        
        odds_df = pd.read_csv(ODDS_PATH)
        odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])
        
        # Load Calibrator
        calibrator = joblib.load(CALIBRATOR_PATH)
        print(f"   ✓ Loaded {len(df):,} games from MDP data")
        print(f"   ✓ Loaded {len(odds_df):,} games with odds")
        print(f"   ✓ Loaded modern isotonic calibrator")
        
    except Exception as e:
        print(f"❌ Setup Error: {e}")
        return

    # 2. Train Base Model on History Before 2024-25
    print(f"\n🏋️ Training base model on history before {START_DATE_24}...")
    season_cutoff = pd.Timestamp(START_DATE_24)
    train_df = df[df['game_date'] < season_cutoff].copy()
    test_df = df[df['game_date'] >= season_cutoff].copy()
    
    if 'margin_target' not in train_df.columns:
        train_df['margin_target'] = train_df['home_score'] - train_df['away_score']
    
    print(f"   Training: {len(train_df):,} games")
    print(f"   Testing:  {len(test_df):,} games")
    
    dtrain = xgb.DMatrix(train_df[ACTIVE_FEATURES], label=train_df['margin_target'])
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    print("   ✓ Model trained")
    
    # 3. Generate Predictions for 2024-25
    print("\n🔮 Generating predictions...")
    dtest = xgb.DMatrix(test_df[ACTIVE_FEATURES])
    pred_margins = model.predict(dtest)
    raw_probs = norm.cdf(pred_margins / NBA_STD_DEV)
    
    # 4. APPLY MODERN ISOTONIC CALIBRATION
    print("📈 Applying modern era calibration...")
    calibrated_probs = calibrator.predict(raw_probs)
    
    test_df['raw_prob'] = raw_probs
    test_df['calibrated_prob'] = calibrated_probs
    
    print(f"   ✓ {len(test_df):,} predictions generated")
    print(f"   Raw prob range: {raw_probs.min():.3f} to {raw_probs.max():.3f}")
    print(f"   Calibrated range: {calibrated_probs.min():.3f} to {calibrated_probs.max():.3f}")
    
    # 5. Merge with Odds
    print("\n🔗 Merging with closing odds...")
    merged = test_df.merge(
        odds_df[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
        on=['game_date', 'home_team', 'away_team'],
        how='inner'
    )
    print(f"   ✓ Matched {len(merged):,} games with odds")
    
    # 6. Simulate Betting (Zero Edge Strategy)
    print("\n💰 Simulating zero-edge betting strategy...")
    bets = []
    
    for idx, row in merged.iterrows():
        if pd.isna(row['home_ml_odds']): 
            continue
        
        # Get implied probability from market
        home_implied = american_to_prob(row['home_ml_odds'])
        away_implied = american_to_prob(row['away_ml_odds'])
        
        # Calculate edge using CALIBRATED probability
        home_edge = row['calibrated_prob'] - home_implied
        away_edge = (1 - row['calibrated_prob']) - away_implied
        
        # ZERO EDGE RULE: Bet if ANY positive edge exists
        bet_made = False
        
        if home_edge > 0 and home_edge >= away_edge:
            # Bet Home
            pick_home = True
            edge = home_edge
            ml_odds = row['home_ml_odds']
            conf = row['calibrated_prob']
            raw_conf = row['raw_prob']
            bet_made = True
            
        elif away_edge > 0 and away_edge > home_edge:
            # Bet Away
            pick_home = False
            edge = away_edge
            ml_odds = row['away_ml_odds']
            conf = 1 - row['calibrated_prob']
            raw_conf = 1 - row['raw_prob']
            bet_made = True
        
        if bet_made:
            # Determine outcome
            actual_home_win = row['margin_target'] > 0
            won = actual_home_win if pick_home else not actual_home_win
            
            # Calculate profit
            if won:
                profit = american_to_payout(ml_odds)
            else:
                profit = -1.0
            
            # Classify bet type (is this a vegas favorite or underdog?)
            is_favorite = ml_odds < 0
            
            bets.append({
                'type': 'Favorite' if is_favorite else 'Underdog',
                'raw_conf': raw_conf,
                'cal_conf': conf,
                'edge': edge,
                'ml_odds': ml_odds,
                'won': won,
                'profit': profit,
                'boosted': conf > raw_conf  # Did calibrator boost confidence?
            })
    
    # 7. Report Results
    results = pd.DataFrame(bets)
    
    print("\n" + "=" * 70)
    print("🏆 FINAL RESULTS - MODERN ISOTONIC ZERO-EDGE")
    print("=" * 70)
    
    total_bets = len(results)
    total_wins = results['won'].sum()
    win_rate = total_wins / total_bets if total_bets > 0 else 0
    total_profit = results['profit'].sum()
    roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
    
    print(f"\n📊 OVERALL")
    print(f"Total Bets:   {total_bets:,}")
    print(f"Wins:         {total_wins:,}")
    print(f"Win Rate:     {win_rate:.1%}")
    print(f"Total Profit: {total_profit:+.2f}u")
    print(f"ROI:          {roi:+.1f}%")
    
    print(f"\n🎯 COMPARISON TO BENCHMARK")
    benchmark = 79.18
    diff = total_profit - benchmark
    if diff > 0:
        print(f"   Benchmark (Optimized Thresholds): +{benchmark:.2f}u")
        print(f"   Modern Isotonic (Zero-Edge):      {total_profit:+.2f}u")
        print(f"   ✅ BEATS BENCHMARK by {diff:+.2f}u ({diff/benchmark*100:+.1f}%)")
    else:
        print(f"   Benchmark (Optimized Thresholds): +{benchmark:.2f}u")
        print(f"   Modern Isotonic (Zero-Edge):      {total_profit:+.2f}u")
        print(f"   ❌ UNDERPERFORMS by {diff:.2f}u ({diff/benchmark*100:.1f}%)")
    
    # 8. Breakdown by Type
    print("\n" + "=" * 70)
    print("🦁 FAVORITES vs UNDERDOGS")
    print("=" * 70)
    
    for btype in ['Favorite', 'Underdog']:
        subset = results[results['type'] == btype]
        if len(subset) > 0:
            wins = subset['won'].sum()
            wr = wins / len(subset)
            profit = subset['profit'].sum()
            roi_pct = profit / len(subset) * 100
            
            print(f"{btype:<10} | Bets: {len(subset):>4} | Win%: {wr:>5.1%} | "
                  f"Profit: {profit:>+7.2f}u | ROI: {roi_pct:>+6.1f}%")
    
    # 9. Boosting Analysis
    print("\n" + "=" * 70)
    print("🔍 IMPACT OF CALIBRATION BOOSTING")
    print("=" * 70)
    
    boosted_bets = results[results['boosted']]
    unboosted_bets = results[~results['boosted']]
    
    print("\nBoosted Bets (Calibrator Increased Confidence):")
    if len(boosted_bets) > 0:
        print(f"   Count:  {len(boosted_bets):,}")
        print(f"   Win%:   {boosted_bets['won'].mean():.1%}")
        print(f"   Profit: {boosted_bets['profit'].sum():+.2f}u")
        print(f"   ROI:    {(boosted_bets['profit'].sum() / len(boosted_bets) * 100):+.1f}%")
    
    print("\nUnboosted/Dampened Bets:")
    if len(unboosted_bets) > 0:
        print(f"   Count:  {len(unboosted_bets):,}")
        print(f"   Win%:   {unboosted_bets['won'].mean():.1%}")
        print(f"   Profit: {unboosted_bets['profit'].sum():+.2f}u")
        print(f"   ROI:    {(unboosted_bets['profit'].sum() / len(unboosted_bets) * 100):+.1f}%")
    
    # 10. High Confidence Analysis
    print("\n" + "=" * 70)
    print("🎲 HIGH CONFIDENCE BETS (>70% Calibrated)")
    print("=" * 70)
    
    high_conf = results[results['cal_conf'] > 0.70]
    if len(high_conf) > 0:
        print(f"Count:          {len(high_conf):,}")
        print(f"Avg Raw Conf:   {high_conf['raw_conf'].mean():.1%}")
        print(f"Avg Cal Conf:   {high_conf['cal_conf'].mean():.1%}")
        print(f"Actual Win%:    {high_conf['won'].mean():.1%}")
        print(f"Profit:         {high_conf['profit'].sum():+.2f}u")
        print(f"ROI:            {(high_conf['profit'].sum() / len(high_conf) * 100):+.1f}%")
        print(f"\n💡 Did the extra confidence pay off?")
        if high_conf['profit'].sum() > 0:
            print("   ✅ YES - High confidence bets were profitable")
        else:
            print("   ❌ NO - High confidence bets lost money")
    
    # 11. Edge Bucket Analysis
    print("\n" + "=" * 70)
    print("📊 EDGE BUCKET ANALYSIS (Calibrated Edge)")
    print("=" * 70)
    
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
        bucket = results[(results['edge'] >= low) & (results['edge'] < high)]
        if len(bucket) > 0:
            wins = bucket['won'].sum()
            win_pct = wins / len(bucket) * 100
            profit = bucket['profit'].sum()
            roi = profit / len(bucket) * 100
            print(f"{label:<12} | {len(bucket):<6} | {win_pct:>6.1f}% | {profit:>+9.2f}u | {roi:>+6.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ BACKTEST COMPLETE")
    print("=" * 70)
    
    # Final verdict
    print("\n🎯 VERDICT:")
    if total_profit > benchmark:
        print("   ✅ Modern isotonic calibration BEATS optimized thresholds")
        print("   The 'boosting' behavior found real signal in the modern era")
        print("   RECOMMENDATION: Deploy zero-edge strategy with modern calibrator")
    else:
        print("   ❌ Modern isotonic calibration UNDERPERFORMS optimized thresholds")
        print("   The 'boosting' behavior was chasing noise, not signal")
        print("   RECOMMENDATION: Stick with optimized thresholds (1.5% fav / 8.0% dog)")

if __name__ == "__main__":
    run_backtest()
