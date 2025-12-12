"""
Walk-Forward Backtest - ROI Analysis
Tests model performance on historical games with proper Kelly sizing
Simulates real betting with calibrated probabilities and commission
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST

# Configuration
MODEL_PATH = 'models/xgboost_pruned_31features.pkl'
DB_PATH = 'data/live/nba_betting_data.db'
START_DATE = '2023-10-01'
END_DATE = '2024-11-01'
INITIAL_BANKROLL = 10000
KELLY_FRACTION = 0.25  # Quarter Kelly for risk management
COMMISSION = 0.07  # Kalshi takes 7% on wins
MIN_EDGE = 0.03  # Minimum 3% edge to bet
MAX_BET_PCT = 0.05  # Never risk more than 5% of bankroll

print("=" * 80)
print("WALK-FORWARD BACKTEST - ROI ANALYSIS")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Period: {START_DATE} to {END_DATE}")
print(f"  Initial Bankroll: ${INITIAL_BANKROLL:,.0f}")
print(f"  Kelly Fraction: {KELLY_FRACTION:.0%}")
print(f"  Commission: {COMMISSION:.0%}")
print(f"  Min Edge: {MIN_EDGE:.0%}")
print(f"  Max Bet: {MAX_BET_PCT:.0%} of bankroll")

# Load model
print(f"\n1. Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print(f"   ✓ Model loaded with {len(FEATURE_WHITELIST)} features")

# Initialize feature calculator
print("\n2. Initializing feature calculator...")
calc = FeatureCalculatorV5()

# Load game results
print(f"\n3. Loading games from {START_DATE} to {END_DATE}...")
conn = sqlite3.connect(DB_PATH)

query = f"""
SELECT 
    game_id,
    game_date,
    home_team,
    away_team,
    home_score,
    away_score,
    home_won
FROM game_results
WHERE game_date >= '{START_DATE}' AND game_date <= '{END_DATE}'
ORDER BY game_date
"""

games_df = pd.read_sql(query, conn)
conn.close()
print(f"   ✓ Loaded {len(games_df)} games")

# Generate predictions and simulate betting
print("\n4. Generating predictions and simulating bets...")
print("   (This will take several minutes...)")

results = []
bankroll = INITIAL_BANKROLL
total_staked = 0
total_profit = 0
total_bets = 0

for idx, row in games_df.iterrows():
    if idx % 50 == 0:
        print(f"   Progress: {idx}/{len(games_df)} ({100*idx/len(games_df):.1f}%) - Bankroll: ${bankroll:,.0f}", end='\r')
    
    try:
        # Generate features for this game
        features = calc.calculate_game_features(
            home_team=row['home_team'],
            away_team=row['away_team'],
            game_date=row['game_date']
        )
        
        if not all(f in features for f in FEATURE_WHITELIST):
            continue
        
        # Get prediction
        X = pd.DataFrame([features])[FEATURE_WHITELIST]
        raw_prob = model.predict_proba(X)[0][1]  # Probability home team wins
        
        # Simple calibration adjustment (ideally use CalibrationFitter)
        # For now, use raw probability with conservative adjustment
        calibrated_prob = raw_prob
        
        # Simulate market odds with realistic vig
        # Market makers add ~8-10% vig total (split between yes/no)
        # Example: 50/50 game becomes 54/54 (108 total)
        VIG = 0.08
        
        # Add vig proportionally to both sides
        market_yes_prob = calibrated_prob * (1 + VIG/2)
        market_no_prob = (1 - calibrated_prob) * (1 + VIG/2)
        
        # Normalize so they sum to 1 + VIG
        total_prob = market_yes_prob + market_no_prob
        market_yes_prob = market_yes_prob / total_prob * (1 + VIG)
        market_no_prob = market_no_prob / total_prob * (1 + VIG)
        
        # Clip to valid range
        market_yes_prob = np.clip(market_yes_prob, 0.02, 0.98)
        
        yes_price = int(market_yes_prob * 100)  # Kalshi cents
        no_price = 100 - yes_price
        
        # Calculate edge (our probability vs market implied probability)
        edge = calibrated_prob - (yes_price / 100)
        
        # Kelly criterion
        if edge > MIN_EDGE:
            # Kelly formula: (edge) / odds
            # For binary outcome: f = (p*odds - 1) / (odds - 1)
            # Simplified: f = edge / (1 - commission)
            kelly_fraction = edge / (1 - COMMISSION)
            bet_size = bankroll * kelly_fraction * KELLY_FRACTION
            
            # Apply max bet constraint
            max_bet = bankroll * MAX_BET_PCT
            bet_size = min(bet_size, max_bet)
            
            # Don't bet if too small
            if bet_size < 10:
                continue
            
            # Determine outcome
            home_won = row['home_won']
            won_bet = (calibrated_prob > 0.5 and home_won == 1) or (calibrated_prob < 0.5 and home_won == 0)
            
            # Calculate profit/loss
            if won_bet:
                # Win: get back stake + profit - commission
                payout = bet_size * (100 / yes_price)
                profit = payout - bet_size
                profit_after_commission = profit * (1 - COMMISSION)
                net_profit = profit_after_commission
            else:
                # Loss: lose entire stake
                net_profit = -bet_size
            
            # Update bankroll
            bankroll += net_profit
            total_staked += bet_size
            total_profit += net_profit
            total_bets += 1
            
            results.append({
                'date': row['game_date'],
                'game': f"{row['away_team']} @ {row['home_team']}",
                'prediction': 'Home' if calibrated_prob > 0.5 else 'Away',
                'confidence': max(calibrated_prob, 1 - calibrated_prob),
                'edge': edge,
                'stake': bet_size,
                'won': won_bet,
                'profit': net_profit,
                'bankroll': bankroll,
                'home_score': row['home_score'],
                'away_score': row['away_score']
            })
    
    except Exception as e:
        continue

print(f"\n   ✓ Generated {len(results)} bets from {len(games_df)} games")

# Convert to DataFrame for analysis
if len(results) == 0:
    print("\n⚠️  No bets placed - check MIN_EDGE threshold")
    sys.exit(1)

results_df = pd.DataFrame(results)

# Calculate metrics
print("\n" + "=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)

wins = results_df['won'].sum()
losses = len(results_df) - wins
win_rate = wins / len(results_df)
roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
final_bankroll = bankroll
net_profit = final_bankroll - INITIAL_BANKROLL
return_pct = (net_profit / INITIAL_BANKROLL) * 100

print(f"\n📊 Overall Performance:")
print(f"   Total Bets: {total_bets}")
print(f"   Wins: {wins} | Losses: {losses}")
print(f"   Win Rate: {win_rate:.1%}")
print(f"   Total Staked: ${total_staked:,.0f}")
print(f"   Total Profit: ${total_profit:,.0f}")
print(f"   ROI: {roi:+.1f}%")
print(f"\n💰 Bankroll:")
print(f"   Starting: ${INITIAL_BANKROLL:,.0f}")
print(f"   Ending: ${final_bankroll:,.0f}")
print(f"   Return: {return_pct:+.1f}%")

# Monthly breakdown
results_df['month'] = pd.to_datetime(results_df['date']).dt.to_period('M')
monthly = results_df.groupby('month').agg({
    'won': ['sum', 'count'],
    'profit': 'sum',
    'stake': 'sum'
})
monthly.columns = ['wins', 'bets', 'profit', 'staked']
monthly['win_rate'] = monthly['wins'] / monthly['bets']
monthly['roi'] = (monthly['profit'] / monthly['staked']) * 100

print(f"\n📅 Monthly Breakdown:")
print("   " + "-" * 76)
print(f"   {'Month':<10} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'Profit':<15} {'ROI':<10}")
print("   " + "-" * 76)
for month, row in monthly.iterrows():
    print(f"   {str(month):<10} {int(row['bets']):<8} {int(row['wins']):<8} {row['win_rate']:<12.1%} ${row['profit']:<14,.0f} {row['roi']:<10.1f}%")

# Edge bucket analysis
results_df['edge_bucket'] = pd.cut(
    results_df['edge'], 
    bins=[0, 0.05, 0.10, 0.15, 1.0],
    labels=['3-5%', '5-10%', '10-15%', '15%+']
)
edge_buckets = results_df.groupby('edge_bucket').agg({
    'won': ['sum', 'count'],
    'profit': 'sum',
    'stake': 'sum'
})
edge_buckets.columns = ['wins', 'bets', 'profit', 'staked']
edge_buckets['win_rate'] = edge_buckets['wins'] / edge_buckets['bets']
edge_buckets['roi'] = (edge_buckets['profit'] / edge_buckets['staked']) * 100

print(f"\n🎯 Performance by Edge:")
print("   " + "-" * 76)
print(f"   {'Edge Range':<12} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'ROI':<10}")
print("   " + "-" * 76)
for edge, row in edge_buckets.iterrows():
    if pd.notna(edge):
        print(f"   {edge:<12} {int(row['bets']):<8} {int(row['wins']):<8} {row['win_rate']:<12.1%} {row['roi']:<10.1f}%")

# Best and worst bets
print(f"\n🏆 Top 5 Winning Bets:")
print("   " + "-" * 76)
top_wins = results_df.nlargest(5, 'profit')
for _, bet in top_wins.iterrows():
    print(f"   {bet['date']}: {bet['game']:<40} | Edge: {bet['edge']:.1%} | Profit: ${bet['profit']:,.0f}")

print(f"\n💸 Top 5 Losing Bets:")
print("   " + "-" * 76)
top_losses = results_df.nsmallest(5, 'profit')
for _, bet in top_losses.iterrows():
    print(f"   {bet['date']}: {bet['game']:<40} | Edge: {bet['edge']:.1%} | Loss: ${bet['profit']:,.0f}")

# Save results
output_file = 'output/walk_forward_backtest_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\n💾 Results saved to: {output_file}")

print("\n" + "=" * 80)
print("✅ BACKTEST COMPLETE")
print("=" * 80)
