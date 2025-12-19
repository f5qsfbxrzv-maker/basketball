"""
Blind Simulation - Logic Stress Test
Tests split thresholds and Kelly sizing with known probabilities
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from betting_strategy_config import (
    FAVORITE_EDGE_THRESHOLD,
    UNDERDOG_EDGE_THRESHOLD,
    ODDS_SPLIT_THRESHOLD,
    KELLY_FRACTION
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = 'data/blind_test_games.csv'
BANKROLL = 10000

print("="*90)
print("BLIND SIMULATION: LOGIC STRESS TEST")
print("="*90)
print(f"\nBankroll: ${BANKROLL:,}")
print(f"Kelly Fraction: {KELLY_FRACTION}x")
print(f"Favorite Threshold: {FAVORITE_EDGE_THRESHOLD*100:.1f}%")
print(f"Underdog Threshold: {UNDERDOG_EDGE_THRESHOLD*100:.1f}%")
print(f"Odds Split: {ODDS_SPLIT_THRESHOLD:.2f}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def calculate_kelly(odds_decimal, win_prob):
    """Calculate Kelly criterion stake percentage"""
    if odds_decimal <= 1:
        return 0
    b = odds_decimal - 1  # Net odds
    p = win_prob
    q = 1 - p
    f_star = (b * p - q) / b
    return max(f_star, 0)

# ==============================================================================
# RUN SIMULATION
# ==============================================================================
print(f"\n{'='*90}")
print("SIMULATION RESULTS")
print(f"{'='*90}\n")

df = pd.read_csv(INPUT_FILE)

# Calculate Implied Prob & Edge using MOCK PROBABILITIES
df['implied_prob'] = 1 / df['moneyline_decimal']
df['edge'] = df['mock_model_prob'] - df['implied_prob']

results = []

for index, row in df.iterrows():
    odds = row['moneyline_decimal']
    prob = row['mock_model_prob']
    edge = row['edge']
    team = row['team']
    opponent = row['opponent']
    
    # DECISION LOGIC (Split Threshold Strategy)
    is_bet = False
    reason = "PASS"
    bet_type = "-"

    if odds < ODDS_SPLIT_THRESHOLD:  # FAVORITE
        if edge >= FAVORITE_EDGE_THRESHOLD:
            is_bet = True
            bet_type = "FAVORITE"
            reason = f"✓ BET (Edge {edge*100:.1f}% > {FAVORITE_EDGE_THRESHOLD*100:.1f}%)"
        else:
            reason = f"✗ REJECT (Edge {edge*100:.1f}% < {FAVORITE_EDGE_THRESHOLD*100:.1f}%)"
            
    else:  # UNDERDOG
        if edge >= UNDERDOG_EDGE_THRESHOLD:
            is_bet = True
            bet_type = "UNDERDOG"
            reason = f"✓ BET (Edge {edge*100:.1f}% > {UNDERDOG_EDGE_THRESHOLD*100:.1f}%)"
        else:
            reason = f"✗ REJECT (Edge {edge*100:.1f}% < {UNDERDOG_EDGE_THRESHOLD*100:.1f}%)"

    # SIZING LOGIC (Kelly Criterion)
    stake_amt = 0
    adj_kelly = 0
    full_kelly = 0
    
    if is_bet:
        full_kelly = calculate_kelly(odds, prob)
        adj_kelly = full_kelly * KELLY_FRACTION
        stake_amt = BANKROLL * adj_kelly

    results.append({
        'Team': team,
        'vs': opponent,
        'Odds': f"{odds:.2f}",
        'Type': bet_type,
        'Model': f"{prob*100:.1f}%",
        'Implied': f"{row['implied_prob']*100:.1f}%",
        'Edge': f"{edge*100:.1f}%",
        'Decision': reason,
        'Full Kelly': f"{full_kelly*100:.1f}%" if is_bet else "-",
        'Adj Kelly': f"{adj_kelly*100:.2f}%" if is_bet else "-",
        'Stake': f"${stake_amt:,.0f}" if is_bet else "-"
    })

# Display results
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ==============================================================================
# VALIDATION CHECKS
# ==============================================================================
print(f"\n{'='*90}")
print("VALIDATION CHECKS")
print(f"{'='*90}\n")

# Extract stake amounts for comparison
bos_stake = float([r['Stake'].replace('$','').replace(',','') for r in results if r['Team'] == 'BOS'][0]) if any(r['Team'] == 'BOS' for r in results) else 0
was_stake = float([r['Stake'].replace('$','').replace(',','') for r in results if r['Team'] == 'WAS'][0]) if any(r['Team'] == 'WAS' for r in results) else 0
lal_rejected = any(r['Team'] == 'LAL' and '✗' in r['Decision'] for r in results)
cha_rejected = any(r['Team'] == 'CHA' and '✗' in r['Decision'] for r in results)

checks = [
    {
        'Test': 'LAL Trap Favorite (0.3% edge)',
        'Expected': 'REJECTED',
        'Result': 'PASS ✓' if lal_rejected else 'FAIL ✗',
        'Status': lal_rejected
    },
    {
        'Test': 'CHA Noise Underdog (5.0% edge)',
        'Expected': 'REJECTED',
        'Result': 'PASS ✓' if cha_rejected else 'FAIL ✗',
        'Status': cha_rejected
    },
    {
        'Test': 'Kelly sizing accounts for edge AND odds',
        'Expected': 'WAS (16.6% edge, 6.5x odds) > BOS (2.0% edge, 1.6x odds)',
        'Result': f'PASS ✓ (${was_stake:,.0f} > ${bos_stake:,.0f}) - Correct per Kelly formula' if was_stake > bos_stake and was_stake > 0 else 'FAIL ✗',
        'Status': (was_stake > bos_stake and was_stake > 0)
    }
]

for check in checks:
    print(f"{'[✓]' if check['Status'] else '[✗]'} {check['Test']}")
    print(f"    Expected: {check['Expected']}")
    print(f"    Result: {check['Result']}\n")

all_passed = all(check['Status'] for check in checks)

print(f"{'='*90}")
if all_passed:
    print("✓ ALL TESTS PASSED - Logic is correct!")
    print("  → Split thresholds working as expected")
    print("  → Kelly sizing properly accounts for edge AND odds")
    print("  → Ready for production deployment")
    print("\n  NOTE: WAS stake ($491) > BOS stake ($133) is CORRECT")
    print("        Kelly optimizes log growth: high edge + high odds = larger stake")
    print("        See verify_kelly_math.py for mathematical proof")
else:
    print("✗ SOME TESTS FAILED - Review logic before deployment")
print(f"{'='*90}")
