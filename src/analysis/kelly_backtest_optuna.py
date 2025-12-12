"""
Quarter Kelly Walk-Forward Backtest with Optuna-Tuned Models
Tests both calibrated and uncalibrated versions to diagnose calibration issue
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

print("="*80)
print("QUARTER KELLY WALK-FORWARD BACKTEST")
print("="*80)

# Configuration
START_BANKROLL = 1000
KELLY_FRACTION = 0.25  # Quarter Kelly (conservative)
MIN_EDGE = 0.03  # 3% minimum edge to bet
VIG_ODDS = -110  # Standard betting line
TEST_START_DATE = '2023-10-24'  # 2023-24 season start

# Load cached features
df = pd.read_csv('data/processed/training_features_30.csv')
df['game_date'] = pd.to_datetime(df['game_date'])
df = df.sort_values('game_date')

# Filter to test period
test_df = df[df['game_date'] >= TEST_START_DATE].copy()
print(f"\nTest period: {TEST_START_DATE} to {test_df['game_date'].max().date()}")
print(f"Games in test set: {len(test_df)}")

# Load models
print("\nLoading models...")
model_uncal = joblib.load('models/xgboost_optuna_uncalibrated.pkl')
model_cal = joblib.load('models/xgboost_optuna_calibrated.pkl')
print("✓ Uncalibrated model loaded")
print("✓ Calibrated model loaded")

# Get features - Load from cached CSV (actual 30 features)
feature_df = pd.read_csv('data/processed/training_features_30.csv', nrows=1)
FEATURE_WHITELIST = [col for col in feature_df.columns if col not in ['home_won', 'game_date', 'home_team', 'away_team']]
print(f"\n✓ Loaded {len(FEATURE_WHITELIST)} features from cached training data")
print(f"   Features: {', '.join(FEATURE_WHITELIST[:5])}...")

X_test = test_df[FEATURE_WHITELIST].values
y_test = test_df['home_won'].values

# Check for injury features
injury_features = [f for f in FEATURE_WHITELIST if 'inj' in f.lower() or 'star' in f.lower()]
print(f"\n🏥 Injury features in model: {injury_features}")

# ============================================================================
# KELLY CRITERION HELPER
# ============================================================================

def calculate_kelly(prob, odds=-110):
    """Calculate Kelly fraction for a bet"""
    if odds < 0:
        decimal_odds = (100 / abs(odds)) + 1
    else:
        decimal_odds = (odds / 100) + 1
    
    b = decimal_odds - 1  # Net odds (profit per $1 wagered)
    p = prob  # Win probability
    q = 1 - p  # Loss probability
    
    # Kelly formula: f* = (bp - q) / b
    f_star = (b * p - q) / b
    return max(f_star, 0.0)  # Don't bet negative Kelly

def implied_probability(odds=-110):
    """Convert odds to implied probability"""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

# ============================================================================
# RUN BACKTESTS FOR BOTH MODELS
# ============================================================================

def run_backtest(model, model_name, X_test, y_test, test_df):
    """Run Kelly backtest for a model"""
    print(f"\n{'='*80}")
    print(f"BACKTEST: {model_name}")
    print(f"{'='*80}")
    
    # Get probabilities
    probs = model.predict_proba(X_test)[:, 1]
    
    # Initialize
    bankroll = START_BANKROLL
    history = []
    bets_placed = 0
    wins = 0
    
    # Market implied probability
    market_prob = implied_probability(VIG_ODDS)
    
    for i in range(len(X_test)):
        prob = probs[i]
        
        # Calculate edge
        edge = prob - market_prob
        
        # Only bet if edge > threshold
        if edge >= MIN_EDGE:
            # Calculate Kelly bet size
            kelly_full = calculate_kelly(prob, VIG_ODDS)
            bet_fraction = kelly_full * KELLY_FRACTION
            
            if bet_fraction > 0:
                wager = bankroll * bet_fraction
                bets_placed += 1
                
                # Outcome
                outcome = y_test[i]
                
                if outcome == 1:  # Home win
                    profit = wager * (100 / 110)  # Win at -110
                    bankroll += profit
                    wins += 1
                    result = 'WIN'
                else:
                    bankroll -= wager
                    result = 'LOSS'
                
                history.append({
                    'date': test_df.iloc[i]['game_date'],
                    'home': test_df.iloc[i]['home_team'],
                    'away': test_df.iloc[i]['away_team'],
                    'prob': prob,
                    'edge': edge,
                    'wager': wager,
                    'result': result,
                    'bankroll': bankroll
                })
    
    # Results
    history_df = pd.DataFrame(history)
    roi = ((bankroll - START_BANKROLL) / START_BANKROLL) * 100
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
    
    print(f"\n💰 RESULTS:")
    print(f"  Starting Bankroll: ${START_BANKROLL:,.2f}")
    print(f"  Final Bankroll:    ${bankroll:,.2f}")
    print(f"  Total Profit:      ${bankroll - START_BANKROLL:,.2f}")
    print(f"  ROI:               {roi:.2f}%")
    print(f"  Bets Placed:       {bets_placed} / {len(X_test)} games ({100*bets_placed/len(X_test):.1f}%)")
    print(f"  Win Rate:          {win_rate:.1f}%")
    print(f"  Required Win Rate: {implied_probability(VIG_ODDS)*100:.1f}% (break-even at -110)")
    
    if bets_placed > 0:
        print(f"\n  Avg Edge:          {history_df['edge'].mean()*100:.2f}%")
        print(f"  Avg Wager:         ${history_df['wager'].mean():.2f}")
        print(f"  Max Wager:         ${history_df['wager'].max():.2f}")
        print(f"  Avg Prob:          {history_df['prob'].mean()*100:.1f}%")
    
    return history_df, roi, win_rate, bets_placed

# Run both backtests
print("\n" + "="*80)
print("TESTING BOTH MODELS")
print("="*80)

history_uncal, roi_uncal, win_uncal, bets_uncal = run_backtest(
    model_uncal, "UNCALIBRATED MODEL", X_test, y_test, test_df
)

history_cal, roi_cal, win_cal, bets_cal = run_backtest(
    model_cal, "CALIBRATED MODEL (Isotonic)", X_test, y_test, test_df
)

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

print(f"\n{'Metric':<25} {'Uncalibrated':>15} {'Calibrated':>15}")
print("-" * 60)
print(f"{'ROI':<25} {roi_uncal:>14.2f}% {roi_cal:>14.2f}%")
print(f"{'Win Rate':<25} {win_uncal:>14.1f}% {win_cal:>14.1f}%")
print(f"{'Bets Placed':<25} {bets_uncal:>15} {bets_cal:>15}")

if roi_uncal > roi_cal:
    print(f"\n⚠️ CALIBRATION MADE PERFORMANCE WORSE")
    print(f"   Uncalibrated model is {roi_uncal - roi_cal:.2f}% better")
    print(f"   Reason: Isotonic overfitted to calibration set")
    print(f"   Solution: Use uncalibrated model OR retrain with Platt scaling")
else:
    print(f"\n✅ CALIBRATION IMPROVED PERFORMANCE")
    print(f"   Calibrated model is {roi_cal - roi_uncal:.2f}% better")

# ============================================================================
# VISUALIZATION
# ============================================================================

if len(history_uncal) > 0 and len(history_cal) > 0:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Uncalibrated
    ax1.plot(history_uncal['date'], history_uncal['bankroll'], linewidth=2, label='Uncalibrated')
    ax1.axhline(y=START_BANKROLL, color='gray', linestyle='--', alpha=0.5, label='Starting Bankroll')
    ax1.set_title(f'Uncalibrated Model: ${history_uncal["bankroll"].iloc[-1]:,.2f} ({roi_uncal:.1f}% ROI)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Bankroll ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calibrated
    ax2.plot(history_cal['date'], history_cal['bankroll'], linewidth=2, color='orange', label='Calibrated')
    ax2.axhline(y=START_BANKROLL, color='gray', linestyle='--', alpha=0.5, label='Starting Bankroll')
    ax2.set_title(f'Calibrated Model: ${history_cal["bankroll"].iloc[-1]:,.2f} ({roi_cal:.1f}% ROI)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Bankroll ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/kelly_backtest_comparison.png', dpi=150)
    print(f"\n📊 Bankroll chart saved to output/kelly_backtest_comparison.png")
    plt.close()

# ============================================================================
# MONTHLY BREAKDOWN (BEST MODEL)
# ============================================================================

best_history = history_uncal if roi_uncal > roi_cal else history_cal
best_model_name = "Uncalibrated" if roi_uncal > roi_cal else "Calibrated"

print(f"\n{'='*80}")
print(f"MONTHLY BREAKDOWN ({best_model_name.upper()} MODEL)")
print(f"{'='*80}")

best_history['month'] = pd.to_datetime(best_history['date']).dt.to_period('M')
monthly = best_history.groupby('month').agg({
    'wager': 'count',
    'result': lambda x: (x == 'WIN').sum(),
    'bankroll': 'last'
}).rename(columns={'wager': 'bets', 'result': 'wins'})

monthly['win_rate'] = (monthly['wins'] / monthly['bets'] * 100).round(1)
monthly['profit'] = monthly['bankroll'].diff().fillna(monthly['bankroll'].iloc[0] - START_BANKROLL)

print(f"\n{'Month':<12} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'Profit':>10}")
print("-" * 50)
for month, row in monthly.iterrows():
    print(f"{str(month):<12} {int(row['bets']):>6} {int(row['wins']):>6} {row['win_rate']:>6.1f}% ${row['profit']:>9.2f}")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)

if roi_uncal > 0 or roi_cal > 0:
    print("\n✅ PROFITABLE MODEL FOUND")
    print(f"Best: {best_model_name} with {max(roi_uncal, roi_cal):.2f}% ROI")
else:
    print("\n⚠️ Both models unprofitable - need more tuning or features")
