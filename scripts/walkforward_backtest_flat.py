"""
Walk-forward backtest with flat unit wagers.
Expanding window: retrain every 7 days, predict next day only.
$100 flat bet when edge > 3%.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

FLAT_BET_SIZE = 100  # $100 per bet
MIN_EDGE = 0.03      # 3% minimum edge
KALSHI_COMMISSION = 0.07  # 7% on wins

def remove_vig(yes_price, no_price):
    """Convert market prices to fair probabilities"""
    yes_implied = yes_price / 100
    no_implied = no_price / 100
    total_prob = yes_implied + no_implied
    
    fair_yes = yes_implied / total_prob
    fair_no = no_implied / total_prob
    
    return fair_yes, fair_no

def main():
    logger.info("="*60)
    logger.info("WALK-FORWARD BACKTEST (FLAT UNITS)")
    logger.info("="*60)
    logger.info(f"Bet size: ${FLAT_BET_SIZE}")
    logger.info(f"Min edge: {MIN_EDGE*100}%")
    logger.info(f"Commission: {KALSHI_COMMISSION*100}%")
    logger.info("")
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv("data/training_data_with_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"  Total games: {len(df):,}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Get features
    exclude_cols = ['date', 'game_id', 'home_team', 'away_team', 'season', 
                   'target_spread', 'target_spread_cover', 'target_moneyline_win', 
                   'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Load best params
    import json
    with open("output/best_params.json", "r") as f:
        best_params = json.load(f)
    best_params['random_state'] = 42
    best_params['tree_method'] = 'hist'
    best_params['eval_metric'] = 'logloss'
    
    # Split data: train on 2015-2023, test on 2023+
    train_cutoff = pd.to_datetime('2023-10-01')
    train_df = df[df['date'] < train_cutoff].copy()
    test_df = df[df['date'] >= train_cutoff].copy()
    
    logger.info(f"\nInitial training: {len(train_df):,} games (up to {train_cutoff.date()})")
    logger.info(f"Testing: {len(test_df):,} games ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    # Get unique test dates
    test_dates = sorted(test_df['date'].unique())
    logger.info(f"Test dates: {len(test_dates)}")
    
    # Walk-forward backtest
    all_bets = []
    retrain_counter = 0
    model = None
    
    logger.info("\nStarting walk-forward backtest...")
    logger.info("Retraining every 7 days\n")
    
    for idx, test_date in enumerate(test_dates):
        # Retrain every 7 days
        if idx % 7 == 0 or model is None:
            retrain_counter += 1
            # Get all data up to (but not including) test date
            train_data = df[df['date'] < test_date].copy()
            
            X_train = train_data[feature_cols]
            y_train = train_data['target_moneyline_win']
            
            logger.info(f"[{idx+1}/{len(test_dates)}] 🔄 RETRAIN #{retrain_counter} on {len(train_data):,} games (up to {test_date.date()})")
            
            model = xgb.XGBClassifier(**best_params)
            model.fit(X_train, y_train, verbose=False)
        
        # Predict on test date only
        day_df = test_df[test_df['date'] == test_date].copy()
        
        if len(day_df) == 0:
            continue
        
        X_test = day_df[feature_cols]
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Make bets
        for i, (_, row) in enumerate(day_df.iterrows()):
            model_prob = y_proba[i]
            
            # Simulate market odds (assuming 50/50 with vig)
            # In production, use real Kalshi prices
            market_yes_price = 52  # cents
            market_no_price = 52   # cents
            
            fair_yes, fair_no = remove_vig(market_yes_price, market_no_price)
            
            # Calculate edge
            edge = model_prob - fair_yes - KALSHI_COMMISSION
            
            if edge > MIN_EDGE:
                # Place bet
                actual_outcome = row['target_moneyline_win']
                payout = FLAT_BET_SIZE * (100 / market_yes_price - 1) if actual_outcome == 1 else -FLAT_BET_SIZE
                
                # Apply commission on wins
                if payout > 0:
                    payout *= (1 - KALSHI_COMMISSION)
                
                all_bets.append({
                    'date': test_date,
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'model_prob': model_prob,
                    'market_prob': fair_yes,
                    'edge': edge,
                    'bet_size': FLAT_BET_SIZE,
                    'actual_outcome': actual_outcome,
                    'pnl': payout
                })
        
        if idx % 50 == 0 and idx > 0:
            logger.info(f"[{idx+1}/{len(test_dates)}] {test_date.date()}: {len(day_df)} games | Total bets: {len(all_bets)}")
    
    # Results
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    
    if len(all_bets) == 0:
        logger.warning("No bets placed!")
        return
    
    bets_df = pd.DataFrame(all_bets)
    
    total_bets = len(bets_df)
    wins = (bets_df['actual_outcome'] == 1).sum()
    losses = total_bets - wins
    win_rate = wins / total_bets
    
    total_wagered = bets_df['bet_size'].sum()
    total_pnl = bets_df['pnl'].sum()
    roi = (total_pnl / total_wagered) * 100
    
    avg_edge = bets_df['edge'].mean()
    avg_model_prob = bets_df['model_prob'].mean()
    
    logger.info(f"Total bets:      {total_bets:,}")
    logger.info(f"Wins:            {wins:,}")
    logger.info(f"Losses:          {losses:,}")
    logger.info(f"Win rate:        {win_rate*100:.2f}%")
    logger.info(f"")
    logger.info(f"Total wagered:   ${total_wagered:,.2f}")
    logger.info(f"Total P&L:       ${total_pnl:,.2f}")
    logger.info(f"ROI:             {roi:.2f}%")
    logger.info(f"")
    logger.info(f"Avg edge:        {avg_edge*100:.2f}%")
    logger.info(f"Avg model prob:  {avg_model_prob*100:.2f}%")
    logger.info(f"")
    logger.info(f"Retrains:        {retrain_counter}")
    
    # Save results
    bets_df.to_csv("output/walkforward_backtest_flat.csv", index=False)
    logger.info("\nBet log saved to: output/walkforward_backtest_flat.csv")
    
    # Save summary
    with open("output/walkforward_performance.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("WALK-FORWARD BACKTEST RESULTS (FLAT UNITS)\n")
        f.write("="*60 + "\n")
        f.write(f"Bet size: ${FLAT_BET_SIZE}\n")
        f.write(f"Min edge: {MIN_EDGE*100}%\n")
        f.write(f"Commission: {KALSHI_COMMISSION*100}%\n")
        f.write(f"\n")
        f.write(f"Total bets:      {total_bets:,}\n")
        f.write(f"Wins:            {wins:,}\n")
        f.write(f"Losses:          {losses:,}\n")
        f.write(f"Win rate:        {win_rate*100:.2f}%\n")
        f.write(f"\n")
        f.write(f"Total wagered:   ${total_wagered:,.2f}\n")
        f.write(f"Total P&L:       ${total_pnl:,.2f}\n")
        f.write(f"ROI:             {roi:.2f}%\n")
        f.write(f"\n")
        f.write(f"Avg edge:        {avg_edge*100:.2f}%\n")
        f.write(f"Avg model prob:  {avg_model_prob*100:.2f}%\n")
        f.write(f"\n")
        f.write(f"Retrains:        {retrain_counter}\n")
    
    logger.info("Summary saved to: output/walkforward_performance.txt")

if __name__ == "__main__":
    main()
