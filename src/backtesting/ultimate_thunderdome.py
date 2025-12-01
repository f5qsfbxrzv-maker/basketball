"""
ULTIMATE THUNDERDOME
Combined betting simulation with divergence analysis and risk metrics
This is the master script that runs EVERYTHING
"""
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtesting.visualize_equity import plot_thunderdome_results, print_risk_report

# --- CONFIGURATION ---
MODEL_A_PATH = "models/production/best_model.joblib"
MODEL_B_PATH = "models/staging/candidate_model.joblib"
TEST_DATA_PATH = "data/processed/training_data_final.csv"
CONFIDENCE_THRESHOLD = 0.55
BET_SIZE = 100

def calculate_roi(bankroll_history, bet_size=BET_SIZE):
    """Calculate ROI from bankroll history"""
    if not bankroll_history:
        return 0
    total_bets = len(bankroll_history)
    final_profit = bankroll_history[-1]
    total_wagered = total_bets * bet_size
    return (final_profit / total_wagered) * 100 if total_wagered > 0 else 0

def simulate_season(df, model, name):
    """
    Simulate a full season of betting with a model
    
    Returns:
        bankroll: List of cumulative profit after each bet
        probs: Array of predicted probabilities
        wins: Number of winning bets
        bets: Total number of bets placed
    """
    print(f"   🏃 Running simulation for {name}...")
    
    # Align features
    if hasattr(model, "feature_names_in_"):
        features = [c for c in model.feature_names_in_ if c in df.columns]
        X = df[features].fillna(0)
    else:
        X = df.select_dtypes(include=[np.number]).drop(columns=['WL'], errors='ignore').fillna(0)
    
    probs = model.predict_proba(X)[:, 1]
    df[f'Prob_{name}'] = probs
    
    bankroll = []
    current_balance = 0
    wins = 0
    bets = 0
    
    for i, prob in enumerate(probs):
        outcome = df['WL'].iloc[i]
        bet_placed = False
        
        # Logic: Bet Home if Prob > Threshold
        if prob > CONFIDENCE_THRESHOLD:
            bet_placed = True
            bets += 1
            if outcome == 1:
                current_balance += (BET_SIZE * 0.909)  # -110 odds = 90.9% return
                wins += 1
            else:
                current_balance -= BET_SIZE
        
        # Logic: Bet Away if Prob < (1-Threshold)
        elif prob < (1 - CONFIDENCE_THRESHOLD):
            bet_placed = True
            bets += 1
            if outcome == 0:
                current_balance += (BET_SIZE * 0.909)
                wins += 1
            else:
                current_balance -= BET_SIZE
        
        if bet_placed:
            bankroll.append(current_balance)
    
    return bankroll, probs, wins, bets

def run_thunderdome():
    """The main Thunderdome execution"""
    print("=" * 80)
    print("🥊 ULTIMATE THUNDERDOME INITIALIZED")
    print("=" * 80)
    
    # Load Assets
    try:
        print(f"\nLoading Model A: {MODEL_A_PATH}")
        model_a = joblib.load(MODEL_A_PATH)
        print(f"Loading Model B: {MODEL_B_PATH}")
        model_b = joblib.load(MODEL_B_PATH)
        print(f"Loading Test Data: {TEST_DATA_PATH}")
        df = pd.read_csv(TEST_DATA_PATH).dropna(subset=['WL'])
        print(f"✅ Loaded {len(df):,} games for testing")
    except Exception as e:
        print(f"❌ Failed to load files: {e}")
        return
    
    # --- ROUND 1: PERFORMANCE ---
    print("\n" + "=" * 80)
    print("📊 ROUND 1: PROFIT & RISK ANALYSIS")
    print("=" * 80)
    
    hist_a, probs_a, wins_a, bets_a = simulate_season(df, model_a, "Model_A")
    hist_b, probs_b, wins_b, bets_b = simulate_season(df, model_b, "Model_B")
    
    roi_a = calculate_roi(hist_a)
    roi_b = calculate_roi(hist_b)
    
    print(f"\n🏛️  MODEL A (Production):")
    print(f"   Profit: ${hist_a[-1]:.2f}")
    print(f"   ROI: {roi_a:.2f}%")
    print(f"   Win Rate: {wins_a/bets_a*100:.1f}% ({wins_a}/{bets_a} bets)")
    
    print(f"\n⚔️  MODEL B (Challenger):")
    print(f"   Profit: ${hist_b[-1]:.2f}")
    print(f"   ROI: {roi_b:.2f}%")
    print(f"   Win Rate: {wins_b/bets_b*100:.1f}% ({wins_b}/{bets_b} bets)")
    
    # Risk metrics
    print_risk_report(hist_a, "Model A")
    print_risk_report(hist_b, "Model B")
    
    # --- ROUND 2: DIVERGENCE ---
    print("\n" + "=" * 80)
    print("⚡ ROUND 2: DIVERGENCE ANALYSIS")
    print("=" * 80)
    
    # Calculate signals: 1 = Bet Home, -1 = Bet Away, 0 = Pass
    sig_a = np.where(probs_a > CONFIDENCE_THRESHOLD, 1, 
                     np.where(probs_a < (1-CONFIDENCE_THRESHOLD), -1, 0))
    sig_b = np.where(probs_b > CONFIDENCE_THRESHOLD, 1, 
                     np.where(probs_b < (1-CONFIDENCE_THRESHOLD), -1, 0))
    
    disagreements = np.sum(sig_a != sig_b)
    print(f"\nTotal Disagreements: {disagreements:,} games ({disagreements/len(df)*100:.1f}%)")
    
    # Head to Head (Opposite bets)
    h2h = np.sum((sig_a == 1) & (sig_b == -1)) + np.sum((sig_a == -1) & (sig_b == 1))
    print(f"🔥 HEAD-TO-HEAD CLASHES (Opposite Bets): {h2h:,} games")
    
    # Who won the disagreements?
    if h2h > 0:
        df['Signal_A'] = sig_a
        df['Signal_B'] = sig_b
        
        conflicts = df[((sig_a == 1) & (sig_b == -1)) | ((sig_a == -1) & (sig_b == 1))].copy()
        
        conflicts['A_Correct'] = ((conflicts['Signal_A'] == 1) & (conflicts['WL'] == 1)) | \
                                 ((conflicts['Signal_A'] == -1) & (conflicts['WL'] == 0))
        
        conflicts['B_Correct'] = ((conflicts['Signal_B'] == 1) & (conflicts['WL'] == 1)) | \
                                 ((conflicts['Signal_B'] == -1) & (conflicts['WL'] == 0))
        
        a_wins_h2h = conflicts['A_Correct'].sum()
        b_wins_h2h = conflicts['B_Correct'].sum()
        
        print(f"\n🥊 Disagreement Results:")
        print(f"   Model A Correct: {a_wins_h2h} ({a_wins_h2h/h2h*100:.1f}%)")
        print(f"   Model B Correct: {b_wins_h2h} ({b_wins_h2h/h2h*100:.1f}%)")
    
    # --- FINAL VERDICT ---
    print("\n" + "=" * 80)
    print("🏆 FINAL VERDICT")
    print("=" * 80)
    
    # Multi-criteria decision
    points_a = 0
    points_b = 0
    
    # Criterion 1: Higher profit
    if hist_a[-1] > hist_b[-1]:
        points_a += 1
        print("✓ Model A wins on PROFIT")
    else:
        points_b += 1
        print("✓ Model B wins on PROFIT")
    
    # Criterion 2: Higher ROI
    if roi_a > roi_b:
        points_a += 1
        print("✓ Model A wins on ROI")
    else:
        points_b += 1
        print("✓ Model B wins on ROI")
    
    # Criterion 3: Better Sharpe (lower risk)
    from src.backtesting.visualize_equity import calculate_risk_metrics
    sharpe_a, _, _ = calculate_risk_metrics(hist_a)
    sharpe_b, _, _ = calculate_risk_metrics(hist_b)
    
    if sharpe_a > sharpe_b:
        points_a += 1
        print("✓ Model A wins on SHARPE RATIO (lower risk)")
    else:
        points_b += 1
        print("✓ Model B wins on SHARPE RATIO (lower risk)")
    
    print("\n" + "=" * 80)
    if points_a > points_b:
        print("👑 WINNER: MODEL A RETAINS THE CROWN")
        print(f"   Victory Score: {points_a}-{points_b}")
    elif points_b > points_a:
        print("👑 WINNER: MODEL B IS THE NEW CHAMPION")
        print(f"   Victory Score: {points_b}-{points_a}")
        print("\n⚠️  ACTION REQUIRED: Promote Model B to production")
    else:
        print("⚖️  TIE: Models are equally matched")
    print("=" * 80)
    
    # --- VISUALIZATION ---
    print("\n📊 Generating equity curve visualization...")
    plot_thunderdome_results(
        hist_a, 
        hist_b, 
        labels=(f'Model A (ROI {roi_a:.1f}%)', f'Model B (ROI {roi_b:.1f}%)')
    )
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Model': ['Model A', 'Model B'],
        'Final_Profit': [hist_a[-1], hist_b[-1]],
        'ROI_%': [roi_a, roi_b],
        'Win_Rate_%': [wins_a/bets_a*100, wins_b/bets_b*100],
        'Total_Bets': [bets_a, bets_b],
        'Sharpe_Ratio': [sharpe_a, sharpe_b]
    })
    
    output_path = "logs/thunderdome_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"💾 Detailed results saved to: {output_path}")
    
    print("\n✅ Thunderdome complete!")

if __name__ == "__main__":
    run_thunderdome()
