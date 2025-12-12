"""
Kelly Criterion Backtest with Star Player Veto

Implements the "safety valve" - skips bets when:
1. Model is confident but ignoring a star injury (shock > 3.0)
2. Star player scratched late (star_missing=1 + high shock)

This prevents the model from betting on teams just because their
"past stats look good" when their best player is actually sitting.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
sys.path.append('src')

from features.feature_calculator_v5 import FeatureCalculatorV5


def calculate_kelly(prob, odds=-110, commission=0.07):
    """
    Calculate Kelly fraction for a bet.
    
    Args:
        prob: Model probability (calibrated)
        odds: American odds (default -110)
        commission: Kalshi commission (7%)
    
    Returns:
        Kelly fraction (0 to 1)
    """
    # Convert to decimal odds
    if odds < 0:
        decimal_odds = (100 / abs(odds)) + 1
    else:
        decimal_odds = (odds / 100) + 1
    
    # Kelly formula: f* = (bp - q) / b
    b = decimal_odds - 1  # Net odds (profit per $1)
    p = prob              # Win probability
    q = 1 - p             # Lose probability
    
    # Adjust for commission (reduces edge)
    edge = (b * p - q) - commission
    
    # Kelly fraction
    if edge <= 0:
        return 0.0
    
    f_star = edge / b
    return max(f_star, 0.0)


def should_veto_bet(features: dict, model_prob: float, 
                    shock_threshold: float = 3.0,
                    confidence_threshold: float = 0.60) -> tuple[bool, str]:
    """
    Star Player Veto: Skip bet if model is ignoring a fresh injury.
    
    Args:
        features: Feature dictionary for the game
        model_prob: Model's predicted probability (home win)
        shock_threshold: Minimum injury_shock to trigger veto (default 3.0)
        confidence_threshold: Model confidence to trigger veto (default 0.60)
    
    Returns:
        (should_veto: bool, reason: str)
    """
    home_star_out = features.get('home_star_missing', 0) == 1
    away_star_out = features.get('away_star_missing', 0) == 1
    
    home_shock = features.get('injury_shock_home', 0)
    away_shock = features.get('injury_shock_away', 0)
    
    # VETO CONDITION 1: Model predicts home win, but home star JUST injured
    if model_prob > confidence_threshold:  # Model confident in HOME
        if home_star_out and home_shock > shock_threshold:
            return True, f"HOME star out + shock={home_shock:.1f} (model ignoring fresh injury)"
    
    # VETO CONDITION 2: Model predicts away win, but away star JUST injured
    if model_prob < (1 - confidence_threshold):  # Model confident in AWAY
        if away_star_out and away_shock > shock_threshold:
            return True, f"AWAY star out + shock={away_shock:.1f} (model ignoring fresh injury)"
    
    # VETO CONDITION 3: Star mismatch is extreme (one team loses superstar)
    star_mismatch = features.get('star_mismatch', 0)
    shock_diff = features.get('injury_shock_diff', 0)
    
    # If home has huge star advantage but model betting on away
    if star_mismatch < -1 and shock_diff < -3.0 and model_prob < 0.5:
        return True, f"AWAY star advantage (mismatch={star_mismatch}, shock={shock_diff:.1f}) but model betting HOME"
    
    # If away has huge star advantage but model betting on home
    if star_mismatch > 1 and shock_diff > 3.0 and model_prob > 0.5:
        return True, f"HOME star advantage (mismatch={star_mismatch}, shock={shock_diff:.1f}) but model betting AWAY"
    
    return False, ""


def kelly_backtest_with_veto(model_path: str, data_path: str, 
                              starting_bankroll: float = 1000.0,
                              kelly_fraction: float = 0.25,
                              edge_threshold: float = 0.03,
                              veto_enabled: bool = True):
    """
    Run Kelly criterion backtest WITH star player veto.
    
    Args:
        model_path: Path to trained XGBoost model
        data_path: Path to test data CSV
        starting_bankroll: Initial bankroll ($)
        kelly_fraction: Kelly multiplier (0.25 = quarter Kelly)
        edge_threshold: Minimum edge to bet (3%)
        veto_enabled: Enable star player veto (default True)
    """
    # Load model and data
    print(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    
    # Initialize tracking
    bankroll = starting_bankroll
    bankroll_history = [starting_bankroll]
    
    bets_made = 0
    bets_won = 0
    bets_vetoed = 0
    veto_reasons = []
    
    total_staked = 0.0
    total_profit = 0.0
    
    print(f"\nStarting Kelly Backtest (Veto {'ENABLED' if veto_enabled else 'DISABLED'})")
    print(f"Initial Bankroll: ${starting_bankroll:,.2f}")
    print(f"Kelly Fraction: {kelly_fraction:.2f}x")
    print(f"Edge Threshold: {edge_threshold:.1%}")
    print("="*70)
    
    for idx, row in df.iterrows():
        # Get features (assuming CSV has feature columns)
        feature_cols = [c for c in df.columns if c not in ['home_win', 'game_date', 'home_team', 'away_team']]
        X = row[feature_cols].values.reshape(1, -1)
        
        # Predict probability
        prob = model.predict_proba(X)[0, 1]  # P(home win)
        
        # Calculate edge (assuming -110 odds for simplicity)
        breakeven = 0.524  # -110 odds breakeven
        edge = prob - breakeven
        
        # Skip if edge too small
        if edge < edge_threshold:
            bankroll_history.append(bankroll)
            continue
        
        # Create features dict for veto check
        features = row[feature_cols].to_dict()
        
        # CHECK VETO
        if veto_enabled:
            should_veto, reason = should_veto_bet(features, prob)
            if should_veto:
                bets_vetoed += 1
                veto_reasons.append({
                    'game_date': row.get('game_date', 'unknown'),
                    'matchup': f"{row.get('away_team', 'AWAY')} @ {row.get('home_team', 'HOME')}",
                    'model_prob': prob,
                    'reason': reason
                })
                bankroll_history.append(bankroll)
                continue  # SKIP BET
        
        # Calculate Kelly stake
        kelly_frac = calculate_kelly(prob, odds=-110)
        bet_size = bankroll * kelly_frac * kelly_fraction
        bet_size = min(bet_size, bankroll * 0.05)  # Cap at 5% of bankroll
        
        # Place bet
        outcome = row['home_win']
        if outcome == 1:
            profit = bet_size * 0.909  # -110 odds net win
            bets_won += 1
        else:
            profit = -bet_size
        
        bankroll += profit
        total_staked += bet_size
        total_profit += profit
        bets_made += 1
        
        bankroll_history.append(bankroll)
    
    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Final Bankroll:    ${bankroll:,.2f}")
    print(f"Total Profit:      ${bankroll - starting_bankroll:,.2f}")
    print(f"ROI:               {((bankroll / starting_bankroll - 1) * 100):.2f}%")
    print(f"\nBets Made:         {bets_made}")
    print(f"Bets Won:          {bets_won}")
    print(f"Win Rate:          {(bets_won / bets_made * 100) if bets_made > 0 else 0:.1f}%")
    print(f"Bets Vetoed:       {bets_vetoed}")
    print(f"Veto Rate:         {(bets_vetoed / (bets_made + bets_vetoed) * 100) if (bets_made + bets_vetoed) > 0 else 0:.1f}%")
    
    if veto_reasons:
        print(f"\n{'='*70}")
        print(f"VETO EXAMPLES (First 5):")
        print(f"{'='*70}")
        for i, veto in enumerate(veto_reasons[:5], 1):
            print(f"{i}. {veto['game_date']} - {veto['matchup']}")
            print(f"   Model Prob: {veto['model_prob']:.1%}")
            print(f"   Reason: {veto['reason']}")
    
    return {
        'final_bankroll': bankroll,
        'roi': (bankroll / starting_bankroll - 1) * 100,
        'bets_made': bets_made,
        'bets_won': bets_won,
        'win_rate': (bets_won / bets_made) if bets_made > 0 else 0,
        'bets_vetoed': bets_vetoed,
        'bankroll_history': bankroll_history
    }


if __name__ == "__main__":
    # Run backtest WITH veto
    print("🔒 KELLY BACKTEST WITH STAR PLAYER VETO")
    print("="*70)
    
    results_with_veto = kelly_backtest_with_veto(
        model_path='models/xgboost_optuna_uncalibrated.pkl',
        data_path='data/test_set_2024.csv',  # Update with your test data
        veto_enabled=True
    )
    
    print("\n\n🔓 KELLY BACKTEST WITHOUT VETO (Comparison)")
    print("="*70)
    
    results_without_veto = kelly_backtest_with_veto(
        model_path='models/xgboost_optuna_uncalibrated.pkl',
        data_path='data/test_set_2024.csv',
        veto_enabled=False
    )
    
    # Compare
    print("\n\n" + "="*70)
    print("VETO IMPACT ANALYSIS")
    print("="*70)
    roi_diff = results_with_veto['roi'] - results_without_veto['roi']
    print(f"ROI Improvement: {roi_diff:+.2f}%")
    print(f"Bets Avoided: {results_with_veto['bets_vetoed']}")
    
    if roi_diff > 0:
        print("\n✅ VETO IMPROVED ROI - Star player safety valve is working!")
    else:
        print("\n⚠️ VETO REDUCED ROI - May be too aggressive (consider tuning thresholds)")
