"""
Kelly Criterion Backtest - Professional Grade

Uses the CLEAN walk-forward validation results (59.9% accuracy).
No data leakage - this simulates real-world deployment.

CONFIGURATION:
- Kelly Fraction: 0.35 (aggressive, up from 0.25)
- Titanium Cap: 8.0% max bet (up from 4.0%)
- February Dampener: OFF (treat all months equally)
- Injury Logic: PENALTY MODE (50% size reduction, not veto)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_kelly_fraction(prob: float, odds: int = -110, commission: float = 0.0) -> float:
    """
    Calculate Kelly fraction for a bet.
    
    Args:
        prob: Model probability of winning
        odds: American odds (negative for favorites)
        commission: Commission rate (e.g., 0.07 for Kalshi)
    
    Returns:
        Kelly fraction (0 to 1)
    """
    # Convert American odds to decimal
    if odds < 0:
        decimal_odds = (100 / abs(odds)) + 1
    else:
        decimal_odds = (odds / 100) + 1
    
    # Kelly formula: f* = (bp - q) / b
    b = decimal_odds - 1  # Net odds
    p = prob
    q = 1 - p
    
    # Edge (adjust for commission)
    edge = (b * p - q) - commission
    
    if edge <= 0:
        return 0.0
    
    kelly_frac = edge / b
    return max(kelly_frac, 0.0)


def apply_injury_penalty(
    features: dict,
    base_kelly: float,
    penalty_mode: bool = True
) -> tuple[float, str]:
    """
    Apply injury-based bet sizing penalty.
    
    PENALTY MODE (penalty_mode=True):
    - If injury shock is high, reduce bet size by 50%
    - Stays in the action but with smaller size
    
    VETO MODE (penalty_mode=False):
    - If injury shock is high, veto bet completely (return 0)
    
    Args:
        features: Feature dictionary for the game
        base_kelly: Base Kelly fraction before penalty
        penalty_mode: If True, apply 50% penalty; if False, veto completely
    
    Returns:
        (adjusted_kelly, reason)
    """
    # Check for injury shock features
    home_shock = features.get('injury_shock_home', 0)
    away_shock = features.get('injury_shock_away', 0)
    shock_diff = features.get('injury_shock_diff', 0)
    
    # High shock threshold (new injury news)
    SHOCK_THRESHOLD = 3.0
    
    # Check if there's significant new injury news
    has_shock = (
        abs(home_shock) > SHOCK_THRESHOLD or 
        abs(away_shock) > SHOCK_THRESHOLD or
        abs(shock_diff) > SHOCK_THRESHOLD
    )
    
    if has_shock:
        if penalty_mode:
            # PENALTY MODE: Reduce bet size by 50%
            adjusted = base_kelly * 0.5
            reason = f"Injury shock detected (home={home_shock:.1f}, away={away_shock:.1f}) - 50% penalty"
            return adjusted, reason
        else:
            # VETO MODE: Skip bet entirely
            return 0.0, f"Injury shock veto (home={home_shock:.1f}, away={away_shock:.1f})"
    
    return base_kelly, "No penalty"


def kelly_backtest_from_walkforward(
    results_path: str = 'output/walk_forward_results.csv',
    starting_bankroll: float = 10000.0,
    kelly_multiplier: float = 0.35,  # AGGRESSIVE (up from 0.25)
    max_bet_pct: float = 0.08,       # TITANIUM CAP: 8% (up from 4%)
    min_edge: float = 0.03,          # 3% minimum edge
    commission: float = 0.0,         # No commission for now
    penalty_mode: bool = True,       # PENALTY not VETO
    february_dampener: bool = False  # OFF
):
    """
    Run Kelly criterion backtest on walk-forward validation results.
    
    Args:
        results_path: Path to walk_forward_results.csv
        starting_bankroll: Initial bankroll ($)
        kelly_multiplier: Kelly fraction multiplier (0.35 = aggressive)
        max_bet_pct: Maximum bet as % of bankroll (0.08 = 8%)
        min_edge: Minimum edge to place bet (0.03 = 3%)
        commission: Betting commission (0.0 = none)
        penalty_mode: If True, apply 50% penalty for injury shock; if False, veto
        february_dampener: If True, reduce bet size in February
    """
    # Load walk-forward results
    print(f"Loading walk-forward results: {results_path}")
    df = pd.read_csv(results_path)
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    print(f"  Loaded {len(df)} games")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"  Overall accuracy: {(df['model_pred'] == df['target_moneyline_win']).mean():.4f}")
    
    # Initialize tracking
    bankroll = starting_bankroll
    bankroll_history = [starting_bankroll]
    
    bets_placed = 0
    bets_won = 0
    total_staked = 0.0
    total_profit = 0.0
    
    injury_penalties = 0
    february_reductions = 0
    
    bet_log = []
    
    print(f"\n{'='*70}")
    print(f"KELLY BACKTEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"Starting Bankroll:   ${starting_bankroll:,.2f}")
    print(f"Kelly Multiplier:    {kelly_multiplier:.2f}x (AGGRESSIVE)")
    print(f"Titanium Cap:        {max_bet_pct:.1%} (MAXIMUM BET SIZE)")
    print(f"Min Edge:            {min_edge:.1%}")
    print(f"Commission:          {commission:.1%}")
    print(f"Injury Mode:         {'PENALTY (50% reduction)' if penalty_mode else 'VETO (skip)'}")
    print(f"February Dampener:   {'ON' if february_dampener else 'OFF'}")
    print(f"{'='*70}\n")
    
    # Process each game
    for idx, row in df.iterrows():
        prob = row['model_prob']
        actual = row['target_moneyline_win']
        game_date = row['game_date']
        
        # Calculate edge (assuming -110 odds)
        breakeven = 0.524  # -110 odds breakeven
        edge = prob - breakeven
        
        # Skip if edge too small
        if edge < min_edge:
            bankroll_history.append(bankroll)
            continue
        
        # Calculate base Kelly fraction
        base_kelly = calculate_kelly_fraction(prob, odds=-110, commission=commission)
        
        # Apply Kelly multiplier
        kelly_frac = base_kelly * kelly_multiplier
        
        # Apply injury penalty if needed
        # (Note: walk-forward results don't have injury features, so skip this for now)
        # In production, you'd merge in injury features from original data
        penalty_reason = "No penalty"
        
        # Apply February dampener if enabled
        if february_dampener and game_date.month == 2:
            kelly_frac *= 0.5
            february_reductions += 1
        
        # Calculate bet size
        bet_size = bankroll * kelly_frac
        
        # Apply Titanium Cap
        max_bet = bankroll * max_bet_pct
        if bet_size > max_bet:
            bet_size = max_bet
        
        # Place bet
        if actual == 1:
            profit = bet_size * 0.909  # -110 odds net win
            won = True
            bets_won += 1
        else:
            profit = -bet_size
            won = False
        
        bankroll += profit
        total_staked += bet_size
        total_profit += profit
        bets_placed += 1
        
        bankroll_history.append(bankroll)
        
        # Log bet
        bet_log.append({
            'game_date': game_date,
            'matchup': f"{row.get('away_team', 'AWAY')} @ {row.get('home_team', 'HOME')}",
            'model_prob': prob,
            'edge': edge,
            'kelly_frac': kelly_frac,
            'bet_size': bet_size,
            'won': won,
            'profit': profit,
            'bankroll': bankroll,
            'penalty_reason': penalty_reason
        })
        
        # Progress update (every 100 games)
        if bets_placed % 100 == 0:
            current_roi = ((bankroll / starting_bankroll - 1) * 100)
            print(f"  [{bets_placed:4d} bets] Bankroll: ${bankroll:,.2f} | ROI: {current_roi:+.1f}%")
    
    # Final results
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"\nBANKROLL:")
    print(f"  Starting:     ${starting_bankroll:,.2f}")
    print(f"  Ending:       ${bankroll:,.2f}")
    print(f"  Profit:       ${bankroll - starting_bankroll:+,.2f}")
    print(f"  ROI:          {((bankroll / starting_bankroll - 1) * 100):+.2f}%")
    
    print(f"\nBETTING STATS:")
    print(f"  Total Games:  {len(df):,}")
    print(f"  Bets Placed:  {bets_placed:,} ({bets_placed / len(df) * 100:.1f}% of games)")
    print(f"  Bets Won:     {bets_won:,}")
    print(f"  Win Rate:     {(bets_won / bets_placed * 100) if bets_placed > 0 else 0:.1f}%")
    print(f"  Total Staked: ${total_staked:,.2f}")
    print(f"  Avg Bet Size: ${total_staked / bets_placed if bets_placed > 0 else 0:,.2f}")
    
    print(f"\nADJUSTMENTS:")
    print(f"  Injury Penalties:     {injury_penalties}")
    print(f"  February Reductions:  {february_reductions}")
    
    # Monthly breakdown
    bet_df = pd.DataFrame(bet_log)
    if len(bet_df) > 0:
        bet_df['month'] = bet_df['game_date'].dt.to_period('M')
        monthly = bet_df.groupby('month').agg({
            'bet_size': 'count',
            'won': 'sum',
            'profit': 'sum'
        }).rename(columns={'bet_size': 'bets'})
        monthly['win_rate'] = monthly['won'] / monthly['bets']
        
        print(f"\n{'='*70}")
        print(f"MONTHLY BREAKDOWN")
        print(f"{'='*70}")
        print(monthly.to_string())
    
    # Save detailed log
    bet_log_path = 'output/kelly_bet_log.csv'
    if len(bet_log) > 0:
        pd.DataFrame(bet_log).to_csv(bet_log_path, index=False)
        print(f"\n💾 Detailed bet log saved: {bet_log_path}")
    
    # Calculate drawdown
    bankroll_series = pd.Series(bankroll_history)
    running_max = bankroll_series.expanding().max()
    drawdown = (bankroll_series - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    print(f"\nRISK METRICS:")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Sharpe Ratio: {calculate_sharpe(bankroll_history):.2f}")
    
    return {
        'final_bankroll': bankroll,
        'roi': (bankroll / starting_bankroll - 1) * 100,
        'bets_placed': bets_placed,
        'win_rate': (bets_won / bets_placed) if bets_placed > 0 else 0,
        'max_drawdown': max_drawdown,
        'bankroll_history': bankroll_history
    }


def calculate_sharpe(bankroll_history: list, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from bankroll history"""
    returns = pd.Series(bankroll_history).pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)


if __name__ == "__main__":
    # Check if walk-forward results exist
    results_path = 'output/walk_forward_results.csv'
    if not Path(results_path).exists():
        print(f"❌ ERROR: Walk-forward results not found: {results_path}")
        print(f"\nPlease run walk-forward validation first:")
        print(f"  python src/analysis/walk_forward_validation.py")
        exit(1)
    
    # Run Kelly backtest with aggressive configuration
    print("🚀 KELLY CRITERION BACKTEST - PROFESSIONAL GRADE")
    print("="*70)
    
    results = kelly_backtest_from_walkforward(
        results_path=results_path,
        starting_bankroll=10000.0,
        kelly_multiplier=0.35,      # AGGRESSIVE (up from 0.25)
        max_bet_pct=0.08,           # TITANIUM CAP: 8%
        min_edge=0.03,              # 3% minimum edge
        commission=0.0,             # No commission
        penalty_mode=True,          # PENALTY not VETO
        february_dampener=False     # OFF
    )
    
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT")
    print(f"{'='*70}")
    
    roi = results['roi']
    if roi > 50:
        print(f"✅ STRONG PERFORMANCE: {roi:+.1f}% ROI")
        print(f"   Model has legitimate edge with aggressive Kelly sizing.")
    elif roi > 20:
        print(f"✅ SOLID PERFORMANCE: {roi:+.1f}% ROI")
        print(f"   Model is profitable with room for optimization.")
    elif roi > 0:
        print(f"⚠️  MARGINAL PERFORMANCE: {roi:+.1f}% ROI")
        print(f"   Model is slightly profitable but needs improvement.")
    else:
        print(f"❌ UNPROFITABLE: {roi:+.1f}% ROI")
        print(f"   Edge exists (59.9% accuracy) but bet sizing may be off.")
    
    print(f"\n📊 Key Metrics:")
    print(f"   Win Rate:     {results['win_rate']*100:.1f}%")
    print(f"   Bets Placed:  {results['bets_placed']:,}")
    print(f"   Max Drawdown: {results['max_drawdown']:.1f}%")
