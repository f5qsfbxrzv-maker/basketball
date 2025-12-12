"""
Fixed-Stake Backtest for Optuna Model
=====================================
More realistic backtest using fixed $100 units per bet instead of Kelly compounding.
This shows linear profit accumulation which is easier to evaluate.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_START_DATE = pd.Timestamp('2023-10-24')  # Start of 2023-24 season
VIG_ODDS = -110  # Standard American odds (52.4% implied)
MIN_EDGE = 0.03  # 3% minimum edge to bet
UNIT_SIZE = 100  # Fixed $100 per bet

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def implied_probability(odds=-110):
    """Convert American odds to implied probability"""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def calculate_profit(wager, odds=-110, won=True):
    """Calculate profit for a bet"""
    if won:
        if odds < 0:
            return wager * (100 / abs(odds))
        else:
            return wager * (odds / 100)
    else:
        return -wager

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("FIXED-STAKE BACKTEST (Realistic Evaluation)")
print("="*80)

df = pd.read_csv('data/processed/training_features_30.csv', parse_dates=['game_date'])
df = df.sort_values('game_date')

# Filter to test period
test_df = df[df['game_date'] >= TEST_START_DATE].copy()
print(f"\nTest period: {TEST_START_DATE.date()} to {test_df['game_date'].max().date()}")
print(f"Games in test set: {len(test_df)}")

# Load models
print("\nLoading models...")
model_uncal = joblib.load('models/xgboost_optuna_uncalibrated.pkl')
model_cal = joblib.load('models/xgboost_optuna_calibrated.pkl')
print("✓ Uncalibrated model loaded")
print("✓ Calibrated model loaded")

# Get features
feature_df = pd.read_csv('data/processed/training_features_30.csv', nrows=1)
FEATURE_WHITELIST = [col for col in feature_df.columns if col not in ['home_won', 'game_date', 'home_team', 'away_team']]

X_test = test_df[FEATURE_WHITELIST].values
y_test = test_df['home_won'].values

# ============================================================================
# BACKTEST FUNCTION
# ============================================================================

def run_fixed_stake_backtest(model, model_name, X_test, y_test, test_df):
    """Run backtest with fixed stake per bet"""
    print(f"\n{'='*80}")
    print(f"BACKTEST: {model_name}")
    print(f"{'='*80}")
    
    # Get probabilities
    probs = model.predict_proba(X_test)[:, 1]
    
    # Initialize
    total_profit = 0
    history = []
    bets_placed = 0
    wins = 0
    total_wagered = 0
    
    # Market implied probability
    market_prob = implied_probability(VIG_ODDS)
    
    edges = []
    probs_list = []
    
    for i in range(len(X_test)):
        prob = probs[i]
        
        # Calculate edge
        edge = prob - market_prob
        
        # Only bet if edge > threshold
        if edge >= MIN_EDGE:
            bets_placed += 1
            total_wagered += UNIT_SIZE
            
            # Outcome
            outcome = y_test[i]
            profit = calculate_profit(UNIT_SIZE, VIG_ODDS, won=(outcome == 1))
            total_profit += profit
            
            if outcome == 1:
                wins += 1
                result = 'WIN'
            else:
                result = 'LOSS'
            
            edges.append(edge)
            probs_list.append(prob)
            
            history.append({
                'date': test_df.iloc[i]['game_date'],
                'home': test_df.iloc[i]['home_team'],
                'away': test_df.iloc[i]['away_team'],
                'prob': prob,
                'edge': edge,
                'wager': UNIT_SIZE,
                'profit': profit,
                'result': result,
                'cumulative_profit': total_profit
            })
    
    # Results
    if bets_placed == 0:
        print("⚠️  No bets placed (no edges > threshold)")
        return None
    
    history_df = pd.DataFrame(history)
    win_rate = (wins / bets_placed * 100)
    roi = (total_profit / total_wagered) * 100
    avg_edge = np.mean(edges)
    avg_prob = np.mean(probs_list)
    
    print(f"\n💰 RESULTS:")
    print(f"  Unit Size:         ${UNIT_SIZE:,.2f}")
    print(f"  Total Wagered:     ${total_wagered:,.2f}")
    print(f"  Total Profit:      ${total_profit:,.2f}")
    print(f"  ROI:               {roi:.2f}%")
    print(f"  Bets Placed:       {bets_placed} / {len(test_df)} games ({bets_placed/len(test_df)*100:.1f}%)")
    print(f"  Win Rate:          {win_rate:.1f}%")
    print(f"  Required Win Rate: 52.4% (break-even at -110)")
    print(f"")
    print(f"  Avg Edge:          {avg_edge*100:.2f}%")
    print(f"  Avg Prob:          {avg_prob*100:.1f}%")
    
    # Monthly breakdown
    history_df['month'] = pd.to_datetime(history_df['date']).dt.to_period('M')
    monthly = history_df.groupby('month').agg({
        'wager': 'count',
        'result': lambda x: (x == 'WIN').sum(),
        'profit': 'sum'
    })
    monthly.columns = ['bets', 'wins', 'profit']
    monthly['win_rate'] = (monthly['wins'] / monthly['bets'] * 100).round(1)
    
    return {
        'name': model_name,
        'history': history_df,
        'monthly': monthly,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': win_rate,
        'bets_placed': bets_placed
    }

# ============================================================================
# RUN BACKTESTS
# ============================================================================

print("\n" + "="*80)
print("TESTING BOTH MODELS")
print("="*80)

results_uncal = run_fixed_stake_backtest(model_uncal, "UNCALIBRATED MODEL", X_test, y_test, test_df)
results_cal = run_fixed_stake_backtest(model_cal, "CALIBRATED MODEL (Isotonic)", X_test, y_test, test_df)

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Metric': ['Total Profit', 'ROI', 'Win Rate', 'Bets Placed'],
    'Uncalibrated': [
        f"${results_uncal['total_profit']:,.2f}",
        f"{results_uncal['roi']:.2f}%",
        f"{results_uncal['win_rate']:.1f}%",
        results_uncal['bets_placed']
    ],
    'Calibrated': [
        f"${results_cal['total_profit']:,.2f}",
        f"{results_cal['roi']:.2f}%",
        f"{results_cal['win_rate']:.1f}%",
        results_cal['bets_placed']
    ]
})

print()
print(comparison.to_string(index=False))

# Determine winner
if results_cal['total_profit'] > results_uncal['total_profit']:
    diff = results_cal['total_profit'] - results_uncal['total_profit']
    print(f"\n✅ CALIBRATION IMPROVED PERFORMANCE")
    print(f"   Calibrated model earned ${diff:,.2f} more")
else:
    diff = results_uncal['total_profit'] - results_cal['total_profit']
    print(f"\n⚠️  CALIBRATION HURT PERFORMANCE")
    print(f"   Uncalibrated model earned ${diff:,.2f} more")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Equity curves
axes[0].plot(results_uncal['history']['date'], results_uncal['history']['cumulative_profit'], 
             label='Uncalibrated', linewidth=2, alpha=0.8)
axes[0].plot(results_cal['history']['date'], results_cal['history']['cumulative_profit'], 
             label='Calibrated', linewidth=2, alpha=0.8)
axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[0].set_title('Cumulative Profit (Fixed $100 Stakes)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Profit ($)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Monthly comparison
months = sorted(set(results_uncal['monthly'].index) | set(results_cal['monthly'].index))
month_labels = [str(m) for m in months]
x = np.arange(len(months))
width = 0.35

uncal_monthly = [results_uncal['monthly'].loc[m, 'profit'] if m in results_uncal['monthly'].index else 0 for m in months]
cal_monthly = [results_cal['monthly'].loc[m, 'profit'] if m in results_cal['monthly'].index else 0 for m in months]

axes[1].bar(x - width/2, uncal_monthly, width, label='Uncalibrated', alpha=0.8)
axes[1].bar(x + width/2, cal_monthly, width, label='Calibrated', alpha=0.8)
axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[1].set_title('Monthly Profit Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Profit ($)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(month_labels, rotation=45)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('output/fixed_stake_backtest_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Equity curve saved to output/fixed_stake_backtest_comparison.png")

# ============================================================================
# MONTHLY BREAKDOWN
# ============================================================================

print("\n" + "="*80)
print("MONTHLY BREAKDOWN (CALIBRATED MODEL)")
print("="*80)

print(f"\n{'Month':<15} {'Bets':>5} {'Wins':>5} {'Win%':>8} {'Profit':>12}")
print("-" * 50)
for month, row in results_cal['monthly'].iterrows():
    print(f"{str(month):<15} {int(row['bets']):>5} {int(row['wins']):>5} {row['win_rate']:>7.1f}% ${row['profit']:>11,.2f}")

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)

if results_cal['roi'] > 0 or results_uncal['roi'] > 0:
    best = results_cal if results_cal['roi'] > results_uncal['roi'] else results_uncal
    print(f"\n✅ PROFITABLE MODEL FOUND")
    print(f"Best: {best['name']} with {best['roi']:.2f}% ROI (${best['total_profit']:,.2f} profit)")
    print(f"\n🎯 RECOMMENDATION: Deploy this model for live betting")
else:
    print(f"\n❌ MODELS NOT PROFITABLE")
    print(f"   Both models lost money on fixed-stake backtest")
    print(f"   Recommendation: Re-calibrate or increase minimum edge threshold")
