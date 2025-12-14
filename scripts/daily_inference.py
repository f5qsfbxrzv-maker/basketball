"""
PRODUCTION DAILY INFERENCE SYSTEM
Generates betting recommendations for today's NBA games

Usage:
    python scripts/daily_inference.py --date 2024-12-13 --bankroll 10000
    
Features:
- Fetches today's NBA schedule
- Fetches current odds from Kalshi/DraftKings
- Computes all 43 features (ELO, injuries, rest, temporal)
- Applies calibrated model predictions
- Calculates edge vs market odds
- Sizes bets using Kelly criterion with safety caps
- Logs all predictions for future calibration
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MIN_EDGE_FOR_BET = 0.03  # 3% minimum edge
MAX_BET_PCT_OF_BANKROLL = 0.05  # 5% max bet size
KELLY_FRACTION_MULTIPLIER = 0.25  # 25% Kelly (quarter Kelly)
KALSHI_BUY_COMMISSION = 0.01  # 1% commission

class DailyInferenceEngine:
    """Production inference engine for daily betting recommendations"""
    
    def __init__(self, model_path: str = 'models/xgboost_final_trial98.json',
                 calibrator_path: str = 'models/isotonic_calibrator_final.pkl'):
        """Load production model and calibrator"""
        
        logger.info("Loading production model...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        with open(calibrator_path, 'rb') as f:
            self.calibrator = pickle.load(f)
        
        # Load feature names
        self.features = self.model.get_booster().feature_names
        logger.info(f"Model loaded: {len(self.features)} features")
        
    def remove_vig(self, home_odds: int, away_odds: int) -> tuple[float, float]:
        """Remove vig to get fair probabilities from American odds"""
        
        def american_to_implied(odds):
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return -odds / (-odds + 100)
        
        home_implied = american_to_implied(home_odds)
        away_implied = american_to_implied(away_odds)
        total = home_implied + away_implied
        
        return home_implied / total, away_implied / total
    
    def calculate_kelly_stake(self, edge: float, odds: int, bankroll: float) -> float:
        """Calculate Kelly stake with safety caps"""
        
        # Adjust edge for commission
        net_edge = edge - KALSHI_BUY_COMMISSION
        
        if net_edge <= MIN_EDGE_FOR_BET:
            return 0.0
        
        # Convert American odds to decimal payout
        if odds > 0:
            payout = odds / 100
        else:
            payout = 100 / -odds
        
        # Kelly fraction: edge / payout
        kelly_fraction = net_edge / payout
        
        # Apply conservative multiplier (25% Kelly)
        kelly_stake = kelly_fraction * KELLY_FRACTION_MULTIPLIER * bankroll
        
        # Safety caps
        max_bet = bankroll * MAX_BET_PCT_OF_BANKROLL
        kelly_stake = min(kelly_stake, max_bet)
        kelly_stake = max(kelly_stake, 0.0)
        
        return kelly_stake
    
    def fetch_todays_games(self, target_date: str = None) -> pd.DataFrame:
        """Fetch today's NBA schedule
        
        TODO: Integrate with nba_stats_collector_v2.py to fetch live schedule
        For now, returns mock data for testing
        """
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching games for {target_date}...")
        
        # TODO: Replace with actual NBA API call
        # from V2.v2.services.nba_stats_collector_v2 import NBAStatsCollector
        # collector = NBAStatsCollector()
        # games = collector.get_games_for_date(target_date)
        
        # Mock data for testing
        mock_games = pd.DataFrame([
            {
                'date': target_date,
                'home_team': 'BOS',
                'away_team': 'MIA',
                'game_time': '19:30',
                'game_id': 'TEST_001'
            },
            {
                'date': target_date,
                'home_team': 'LAL',
                'away_team': 'GSW',
                'game_time': '22:00',
                'game_id': 'TEST_002'
            }
        ])
        
        logger.info(f"Found {len(mock_games)} games")
        return mock_games
    
    def fetch_odds(self, games: pd.DataFrame) -> pd.DataFrame:
        """Fetch current odds for games
        
        TODO: Integrate with odds_service.py for live odds
        """
        
        logger.info("Fetching current odds...")
        
        # TODO: Replace with actual odds API
        # from V2.v2.services.odds_service import OddsService
        # odds_service = OddsService()
        # odds = odds_service.get_moneyline_odds(games)
        
        # Mock odds for testing
        games['home_ml_odds'] = [-200, -150]
        games['away_ml_odds'] = [+170, +130]
        
        return games
    
    def compute_features(self, games: pd.DataFrame) -> pd.DataFrame:
        """Compute all 43 features for today's games
        
        TODO: Integrate with feature_calculator_v5.py
        """
        
        logger.info("Computing features...")
        
        # TODO: Replace with actual feature computation
        # from V2.v2.features.feature_calculator_v5 import FeatureCalculatorV5
        # calculator = FeatureCalculatorV5()
        # features = calculator.calculate_game_features(games)
        
        # Mock features for testing (zeros for now)
        for feat in self.features:
            games[feat] = 0.0
        
        # Add some realistic values for key features
        games['season_year'] = 2024
        games['season_year_normalized'] = 0.9
        games['games_into_season'] = 250
        games['season_progress'] = 0.3
        
        logger.info("Features computed")
        return games
    
    def generate_predictions(self, games: pd.DataFrame) -> pd.DataFrame:
        """Generate calibrated predictions for games"""
        
        logger.info("Generating predictions...")
        
        # Extract features
        X = games[self.features]
        
        # Raw predictions
        raw_probs = self.model.predict_proba(X)[:, 1]
        
        # Calibrated predictions
        cal_probs = self.calibrator.transform(raw_probs)
        
        games['raw_prob'] = raw_probs
        games['calibrated_prob'] = cal_probs
        
        logger.info(f"Predictions generated for {len(games)} games")
        return games
    
    def calculate_edges(self, games: pd.DataFrame) -> pd.DataFrame:
        """Calculate edge vs market odds"""
        
        logger.info("Calculating edges...")
        
        # Remove vig from market odds
        fair_probs = games.apply(
            lambda row: self.remove_vig(row['home_ml_odds'], row['away_ml_odds']),
            axis=1
        )
        
        games['home_fair_prob'] = [fp[0] for fp in fair_probs]
        games['away_fair_prob'] = [fp[1] for fp in fair_probs]
        
        # Edge = model prob - fair market prob
        games['edge'] = games['calibrated_prob'] - games['home_fair_prob']
        games['abs_edge'] = games['edge'].abs()
        
        return games
    
    def generate_recommendations(self, games: pd.DataFrame, bankroll: float) -> pd.DataFrame:
        """Generate betting recommendations with Kelly sizing"""
        
        logger.info(f"Generating recommendations for ${bankroll:,.0f} bankroll...")
        
        recommendations = []
        
        for idx, row in games.iterrows():
            # Determine bet side
            if row['edge'] > 0:
                bet_side = 'HOME'
                bet_team = row['home_team']
                odds = row['home_ml_odds']
                edge = row['edge']
            else:
                bet_side = 'AWAY'
                bet_team = row['away_team']
                odds = row['away_ml_odds']
                edge = -row['edge']
            
            # Calculate stake
            stake = self.calculate_kelly_stake(edge, odds, bankroll)
            
            # Only recommend if stake > 0
            if stake > 0:
                # Calculate potential profit
                if odds > 0:
                    potential_profit = stake * (odds / 100)
                else:
                    potential_profit = stake * (100 / -odds)
                
                recommendations.append({
                    'game_time': row['game_time'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'bet_team': bet_team,
                    'bet_side': bet_side,
                    'odds': odds,
                    'model_prob': row['calibrated_prob'],
                    'market_prob': row['home_fair_prob'] if bet_side == 'HOME' else row['away_fair_prob'],
                    'edge': edge,
                    'stake': stake,
                    'stake_pct': (stake / bankroll) * 100,
                    'potential_profit': potential_profit,
                    'roi': (potential_profit / stake) * 100,
                    'game_id': row['game_id']
                })
        
        rec_df = pd.DataFrame(recommendations)
        
        if len(rec_df) > 0:
            rec_df = rec_df.sort_values('edge', ascending=False)
            logger.info(f"Generated {len(rec_df)} betting recommendations")
        else:
            logger.info("No bets meet minimum edge threshold")
        
        return rec_df
    
    def log_predictions(self, games: pd.DataFrame, output_path: str = 'data/live/predictions_log.csv'):
        """Log predictions for future calibration updates"""
        
        logger.info("Logging predictions...")
        
        log_df = games[['date', 'home_team', 'away_team', 'game_id', 
                        'calibrated_prob', 'home_ml_odds', 'away_ml_odds']].copy()
        
        log_df['prediction_timestamp'] = datetime.now().isoformat()
        log_df['model_version'] = 'trial98_isotonic'
        
        # Append to log file
        if Path(output_path).exists():
            existing = pd.read_csv(output_path)
            log_df = pd.concat([existing, log_df], ignore_index=True)
        
        log_df.to_csv(output_path, index=False)
        logger.info(f"Predictions logged to {output_path}")
    
    def print_recommendations(self, recommendations: pd.DataFrame):
        """Print betting recommendations to console"""
        
        print("\n" + "="*100)
        print("BETTING RECOMMENDATIONS")
        print("="*100)
        
        if len(recommendations) == 0:
            print("\n❌ No bets meet minimum edge threshold")
            print(f"   Minimum edge required: {MIN_EDGE_FOR_BET*100:.1f}%")
            return
        
        total_stake = recommendations['stake'].sum()
        total_potential = recommendations['potential_profit'].sum()
        
        print(f"\nTotal recommendations: {len(recommendations)}")
        print(f"Total stake: ${total_stake:,.2f}")
        print(f"Total potential profit: ${total_potential:,.2f}")
        print(f"Average edge: {recommendations['edge'].mean()*100:.2f}%")
        
        print("\n" + "-"*100)
        print(f"{'Time':<8} {'Matchup':<20} {'Bet':<15} {'Odds':<8} {'Edge':<8} {'Stake':<12} {'Profit':<12}")
        print("-"*100)
        
        for idx, row in recommendations.iterrows():
            matchup = f"{row['away_team']}@{row['home_team']}"
            bet = f"{row['bet_team']} ({row['bet_side']})"
            odds_str = f"{row['odds']:+d}"
            edge_str = f"{row['edge']*100:.2f}%"
            stake_str = f"${row['stake']:.0f} ({row['stake_pct']:.1f}%)"
            profit_str = f"${row['potential_profit']:.0f}"
            
            print(f"{row['game_time']:<8} {matchup:<20} {bet:<15} {odds_str:<8} {edge_str:<8} {stake_str:<12} {profit_str:<12}")
        
        print("-"*100)
        print("\n💡 BETTING INSTRUCTIONS:")
        print("   1. Verify odds are still available (they move quickly)")
        print("   2. Log actual odds obtained in bet tracking system")
        print("   3. Place bets 8-12 hours before game time (opening lines)")
        print("   4. Update predictions_log.csv with outcomes after games settle")
        print("\n⚠️  RISK WARNING:")
        print("   - Start with 25% of recommended stakes for first 2 weeks")
        print("   - Track performance vs backtest expectations")
        print("   - Stop betting if cumulative loss exceeds 10% of bankroll")
        
        print("\n" + "="*100)


def main():
    """Main entry point for daily inference"""
    
    parser = argparse.ArgumentParser(description='Generate daily NBA betting recommendations')
    parser.add_argument('--date', type=str, default=None, 
                       help='Target date (YYYY-MM-DD). Default: today')
    parser.add_argument('--bankroll', type=float, default=10000,
                       help='Current bankroll in dollars. Default: 10000')
    parser.add_argument('--output', type=str, default='data/live/daily_recommendations.csv',
                       help='Output path for recommendations CSV')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = DailyInferenceEngine()
    
    # Fetch today's games
    games = engine.fetch_todays_games(args.date)
    
    if len(games) == 0:
        logger.info("No games scheduled for today")
        return
    
    # Fetch odds
    games = engine.fetch_odds(games)
    
    # Compute features
    games = engine.compute_features(games)
    
    # Generate predictions
    games = engine.generate_predictions(games)
    
    # Calculate edges
    games = engine.calculate_edges(games)
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(games, args.bankroll)
    
    # Log predictions for calibration
    engine.log_predictions(games)
    
    # Save recommendations
    if len(recommendations) > 0:
        recommendations.to_csv(args.output, index=False)
        logger.info(f"Recommendations saved to {args.output}")
    
    # Print to console
    engine.print_recommendations(recommendations)
    
    logger.info("Daily inference complete")


if __name__ == '__main__':
    main()
