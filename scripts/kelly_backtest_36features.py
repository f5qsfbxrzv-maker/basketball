"""
Kelly criterion backtest for 36-feature model with comprehensive ROI analysis.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INITIAL_BANKROLL = 10000
KELLY_MULTIPLIER = 0.25  # Quarter Kelly
MIN_EDGE = 0.03  # 3% minimum edge
MAX_BET_PCT = 0.08  # 8% max bet
KALSHI_COMMISSION = 0.03  # 3% commission

def calculate_kelly_stake(edge, win_prob, bankroll):
    """Calculate Kelly stake with constraints."""
    if edge <= MIN_EDGE:
        return 0
    
    # Kelly formula: f = edge / odds
    # For binary outcome: f = p - (1-p)/odds
    # Simplified for 50-50 market: f = 2*edge
    kelly_fraction = edge
    
    # Apply multiplier and cap
    stake_pct = min(kelly_fraction * KELLY_MULTIPLIER, MAX_BET_PCT)
    return bankroll * stake_pct

def main():
    logger.info("Loading 36-feature model...")
    model = joblib.load("models/xgboost_36features_tuned.pkl")
    
    logger.info("Loading training data...")
    df = pd.read_csv("data/training_data_with_features.csv")
    
    # Sort by date for time-series backtest
    df = df.sort_values('date').reset_index(drop=True)
    
    # Features
    feature_cols = [c for c in df.columns if c not in [
        'game_id', 'date', 'home_team', 'away_team', 'season',
        'target_spread', 'target_spread_cover', 'target_moneyline_win',
        'target_game_total', 'target_over_under'
    ]]
    
    # Use last 20% as test set (walk-forward simulation)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()
    
    logger.info(f"Backtesting on {len(df_test)} games from {df_test['date'].min()} to {df_test['date'].max()}")
    
    # Get predictions
    X_test = df_test[feature_cols]
    df_test['pred_prob'] = model.predict_proba(X_test)[:, 1]
    
    # Calculate edge (assuming fair market at 50%)
    df_test['edge'] = abs(df_test['pred_prob'] - 0.5) - KALSHI_COMMISSION
    
    # Kelly backtest
    bankroll = INITIAL_BANKROLL
    bet_log = []
    
    for idx, row in df_test.iterrows():
        if row['edge'] <= MIN_EDGE:
            continue
        
        # Determine bet direction
        bet_home = row['pred_prob'] > 0.5
        win_prob = row['pred_prob'] if bet_home else (1 - row['pred_prob'])
        
        # Calculate stake
        stake = calculate_kelly_stake(row['edge'], win_prob, bankroll)
        
        if stake < 10:  # Min $10 bet
            continue
        
        # Determine outcome
        actual_home_win = row['target_moneyline_win'] == 1
        won = (bet_home and actual_home_win) or (not bet_home and not actual_home_win)
        
        # Calculate profit
        if won:
            profit = stake * 0.97  # After commission
        else:
            profit = -stake
        
        bankroll += profit
        
        bet_log.append({
            'date': row['date'],
            'game_id': row['game_id'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'bet_on': row['home_team'] if bet_home else row['away_team'],
            'stake': stake,
            'pred_prob': row['pred_prob'],
            'edge': row['edge'],
            'won': won,
            'profit': profit,
            'bankroll': bankroll
        })
    
    # Create results dataframe
    results = pd.DataFrame(bet_log)
    
    # Calculate metrics
    total_bets = len(results)
    wins = results['won'].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets
    
    total_staked = results['stake'].sum()
    total_profit = results['profit'].sum()
    roi = total_profit / total_staked
    
    final_bankroll = results['bankroll'].iloc[-1]
    total_return = (final_bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL
    
    # Sharpe ratio (annualized)
    daily_returns = results.groupby('date')['profit'].sum()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Print results
    logger.info(f"\n{'='*60}")
    logger.info("KELLY BACKTEST RESULTS (36 Features)")
    logger.info(f"{'='*60}")
    logger.info(f"Total Bets:        {total_bets:,}")
    logger.info(f"Wins:              {wins:,} ({win_rate*100:.1f}%)")
    logger.info(f"Losses:            {losses:,}")
    logger.info(f"")
    logger.info(f"Total Staked:      ${total_staked:,.2f}")
    logger.info(f"Total Profit:      ${total_profit:,.2f}")
    logger.info(f"ROI:               {roi*100:+.2f}%")
    logger.info(f"")
    logger.info(f"Initial Bankroll:  ${INITIAL_BANKROLL:,.2f}")
    logger.info(f"Final Bankroll:    ${final_bankroll:,.2f}")
    logger.info(f"Total Return:      {total_return*100:+.2f}%")
    logger.info(f"")
    logger.info(f"Sharpe Ratio:      {sharpe:.2f}")
    logger.info(f"Avg Bet Size:      ${results['stake'].mean():,.2f}")
    logger.info(f"Max Bet Size:      ${results['stake'].max():,.2f}")
    logger.info(f"{'='*60}")
    
    # Save results
    results.to_csv("output/kelly_backtest_36features.csv", index=False)
    logger.info("\n✅ Bet log saved to output/kelly_backtest_36features.csv")
    
    # Save summary
    summary = {
        'total_bets': total_bets,
        'win_rate': win_rate,
        'roi': roi,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'final_bankroll': final_bankroll
    }
    pd.Series(summary).to_csv("output/kelly_backtest_summary_36features.csv")
    logger.info("✅ Summary saved to output/kelly_backtest_summary_36features.csv")

if __name__ == "__main__":
    main()
