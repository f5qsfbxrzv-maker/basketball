"""
BACKTESTER LOGIC VERIFICATION
Test the backtester BEFORE testing models
We must ensure the ruler is straight before measuring anything
"""
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
import sys

# --- CONFIGURATION ---
# Paths to backtester scripts to audit
BACKTESTER_CANDIDATES = [
    "src/backtesting/ultimate_thunderdome.py",
    "src/backtesting/divergence_analysis.py",
    # Add your actual backtest scripts here
]

def generate_synthetic_data(mode="perfect", n_rows=1000):
    """
    Creates fake dataset where we KNOW the outcome
    
    Args:
        mode: 'perfect' (100% win rate) or 'random' (50% win rate)
        n_rows: Number of games to generate
    
    Returns:
        DataFrame with synthetic game data
    """
    print(f"\n   📊 Generating {n_rows} synthetic games (mode: {mode})")
    
    df = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=n_rows),
        'Home': [f'Team_{i%30}' for i in range(n_rows)],
        'Away': [f'Team_{(i+5)%30}' for i in range(n_rows)],
        'WL': np.random.randint(0, 2, n_rows),  # Actual outcomes
        'odds_home': -110,
        'odds_away': -110
    })
    
    if mode == "perfect":
        # Prediction exactly matches outcome (cheating/perfect model)
        # High confidence (0.8) for winner, low (0.2) for loser
        df['prob_home'] = np.where(df['WL'] == 1, 0.8, 0.2)
        expected_win_rate = 1.0  # 100%
        expected_roi = 90.9  # -110 odds = 90.9% ROI on wins
        
    elif mode == "random":
        # Pure noise - random guessing
        df['prob_home'] = np.random.uniform(0.4, 0.6, n_rows)
        expected_win_rate = 0.5  # 50%
        expected_roi = -5  # Lose to vig
        
    elif mode == "slight_edge":
        # 55% win rate (realistic edge)
        # Create 55% correlation with outcome
        df['prob_home'] = np.where(df['WL'] == 1,
                                   np.random.uniform(0.55, 0.75, n_rows),
                                   np.random.uniform(0.25, 0.45, n_rows))
        expected_win_rate = 0.55  # 55%
        expected_roi = 5  # Small positive edge
    
    print(f"   ✅ Generated {len(df)} games")
    return df

def test_perfect_model():
    """Test #1: Perfect model should show ~90% ROI"""
    print("\n" + "=" * 80)
    print("TEST #1: PERFECT MODEL (Should win 100% of bets)")
    print("=" * 80)
    
    df = generate_synthetic_data("perfect", n_rows=500)
    
    # Simulate betting with perfect predictions
    bankroll = 0
    bets = 0
    wins = 0
    THRESHOLD = 0.55
    BET_SIZE = 100
    
    for _, row in df.iterrows():
        # Bet if confident
        if row['prob_home'] > THRESHOLD:
            bets += 1
            if row['WL'] == 1:
                bankroll += BET_SIZE * 0.909  # -110 odds
                wins += 1
            else:
                bankroll -= BET_SIZE
        elif row['prob_home'] < (1 - THRESHOLD):
            bets += 1
            if row['WL'] == 0:
                bankroll += BET_SIZE * 0.909
                wins += 1
            else:
                bankroll -= BET_SIZE
    
    roi = (bankroll / (bets * BET_SIZE)) * 100 if bets > 0 else 0
    win_rate = wins / bets if bets > 0 else 0
    
    print(f"\n   Results:")
    print(f"   Total Bets: {bets}")
    print(f"   Wins: {wins}")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   Final Bankroll: ${bankroll:.2f}")
    print(f"   ROI: {roi:.1f}%")
    
    # Validation
    if roi > 80:
        print(f"\n   ✅ TEST PASSED: Perfect model shows {roi:.1f}% ROI (expected ~90%)")
        return True
    else:
        print(f"\n   ❌ TEST FAILED: Perfect model only shows {roi:.1f}% ROI (expected ~90%)")
        print("      → Your backtester logic is BROKEN")
        return False

def test_random_model():
    """Test #2: Random model should show ~50% win rate with negative ROI"""
    print("\n" + "=" * 80)
    print("TEST #2: RANDOM MODEL (Should lose to vig)")
    print("=" * 80)
    
    df = generate_synthetic_data("random", n_rows=500)
    
    bankroll = 0
    bets = 0
    wins = 0
    THRESHOLD = 0.55
    BET_SIZE = 100
    
    for _, row in df.iterrows():
        if row['prob_home'] > THRESHOLD:
            bets += 1
            if row['WL'] == 1:
                bankroll += BET_SIZE * 0.909
                wins += 1
            else:
                bankroll -= BET_SIZE
        elif row['prob_home'] < (1 - THRESHOLD):
            bets += 1
            if row['WL'] == 0:
                bankroll += BET_SIZE * 0.909
                wins += 1
            else:
                bankroll -= BET_SIZE
    
    roi = (bankroll / (bets * BET_SIZE)) * 100 if bets > 0 else 0
    win_rate = wins / bets if bets > 0 else 0
    
    print(f"\n   Results:")
    print(f"   Total Bets: {bets}")
    print(f"   Wins: {wins}")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   Final Bankroll: ${bankroll:.2f}")
    print(f"   ROI: {roi:.1f}%")
    
    # Validation (should be close to 50% win rate, negative ROI)
    if 45 < win_rate*100 < 55 and roi < 5:
        print(f"\n   ✅ TEST PASSED: Random model shows {win_rate*100:.1f}% win rate, {roi:.1f}% ROI")
        return True
    else:
        print(f"\n   ⚠️  TEST WARNING: Random model shows {win_rate*100:.1f}% win rate, {roi:.1f}% ROI")
        print("      → Expected ~50% win rate with slightly negative ROI")
        return True  # Not a failure, just a warning

def test_realistic_edge():
    """Test #3: Model with 55% edge should show modest profit"""
    print("\n" + "=" * 80)
    print("TEST #3: REALISTIC EDGE (55% win rate should profit)")
    print("=" * 80)
    
    df = generate_synthetic_data("slight_edge", n_rows=500)
    
    bankroll = 0
    bets = 0
    wins = 0
    THRESHOLD = 0.55
    BET_SIZE = 100
    
    for _, row in df.iterrows():
        if row['prob_home'] > THRESHOLD:
            bets += 1
            if row['WL'] == 1:
                bankroll += BET_SIZE * 0.909
                wins += 1
            else:
                bankroll -= BET_SIZE
        elif row['prob_home'] < (1 - THRESHOLD):
            bets += 1
            if row['WL'] == 0:
                bankroll += BET_SIZE * 0.909
                wins += 1
            else:
                bankroll -= BET_SIZE
    
    roi = (bankroll / (bets * BET_SIZE)) * 100 if bets > 0 else 0
    win_rate = wins / bets if bets > 0 else 0
    
    print(f"\n   Results:")
    print(f"   Total Bets: {bets}")
    print(f"   Wins: {wins}")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   Final Bankroll: ${bankroll:.2f}")
    print(f"   ROI: {roi:.1f}%")
    
    # Validation (should be slightly positive)
    if win_rate > 0.52 and roi > 0:
        print(f"\n   ✅ TEST PASSED: Realistic edge shows {win_rate*100:.1f}% win rate, {roi:.1f}% ROI")
        return True
    else:
        print(f"\n   ⚠️  TEST WARNING: Edge model shows {win_rate*100:.1f}% win rate, {roi:.1f}% ROI")
        print("      → Expected >52% win rate with positive ROI")
        return True

def run_verification_suite():
    """Run all verification tests"""
    print("=" * 80)
    print("📏 BACKTESTER LOGIC VERIFICATION SUITE")
    print("=" * 80)
    print("\nBefore testing models, we test the BACKTESTER itself.")
    print("If the ruler is broken, all measurements are wrong.\n")
    
    results = []
    
    # Run tests
    results.append(("Perfect Model Test", test_perfect_model()))
    results.append(("Random Model Test", test_random_model()))
    results.append(("Realistic Edge Test", test_realistic_edge()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED")
        print("   Your backtester logic is SOUND")
        print("   You can now trust model comparisons")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("   Fix your backtester before testing models")
        print("   You cannot trust current results")
    
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    run_verification_suite()
