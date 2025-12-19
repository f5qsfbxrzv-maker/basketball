"""
Enhanced Bet Tracker for Trial 1306
Tracks all predictions, outcomes, and performance metrics
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class BetTracker:
    """Comprehensive bet tracking with ROI, win%, and all metrics"""
    
    def __init__(self, db_path: str = 'data/live/nba_betting_data.db'):
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create all tracking tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trial1306_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_date TIMESTAMP NOT NULL,
                game_date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                
                -- Prediction details
                bet_type TEXT NOT NULL,
                predicted_winner TEXT NOT NULL,
                model_version TEXT DEFAULT 'Trial1306',
                
                -- Probabilities
                model_probability REAL NOT NULL,
                fair_probability REAL NOT NULL,
                calibrated_probability REAL,
                
                -- Odds & Market
                market_odds REAL NOT NULL,
                market_source TEXT DEFAULT 'Kalshi',
                yes_price REAL,
                no_price REAL,
                
                -- Betting Strategy
                edge REAL NOT NULL,
                kelly_fraction REAL,
                stake_amount REAL NOT NULL,
                bankroll_at_bet REAL NOT NULL,
                threshold_type TEXT,
                
                -- Features (for analysis)
                home_composite_elo REAL,
                away_composite_elo REAL,
                injury_matchup_advantage REAL,
                
                -- Outcome
                outcome TEXT,
                actual_winner TEXT,
                home_score INTEGER,
                away_score INTEGER,
                profit_loss REAL,
                graded_at TIMESTAMP,
                
                -- Metadata
                created_at TIMESTAMP NOT NULL,
                UNIQUE(game_date, home_team, away_team, bet_type)
            )
        """)
        
        # Performance metrics table (daily snapshots)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                
                -- Overall Performance
                total_bets INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                pending INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                
                -- Financial
                total_staked REAL NOT NULL,
                total_profit_loss REAL NOT NULL,
                roi REAL NOT NULL,
                average_stake REAL NOT NULL,
                
                -- Strategy Metrics
                average_edge REAL NOT NULL,
                average_kelly REAL NOT NULL,
                average_model_prob REAL NOT NULL,
                
                -- Bankroll
                starting_bankroll REAL NOT NULL,
                current_bankroll REAL NOT NULL,
                peak_bankroll REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                
                -- Calibration
                brier_score REAL,
                log_loss REAL,
                
                created_at TIMESTAMP NOT NULL
            )
        """)
        
        # Bet types breakdown
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bet_type_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                
                total_bets INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                
                total_staked REAL NOT NULL,
                total_profit_loss REAL NOT NULL,
                roi REAL NOT NULL,
                
                created_at TIMESTAMP NOT NULL,
                UNIQUE(date, bet_type)
            )
        """)
        
        conn.commit()
        conn.close()
        print("[BET TRACKER] Database tables initialized")
    
    def log_bet(self, 
                game_date: str,
                home_team: str,
                away_team: str,
                bet_type: str,
                predicted_winner: str,
                model_probability: float,
                fair_probability: float,
                market_odds: float,
                edge: float,
                stake_amount: float,
                bankroll: float,
                kelly_fraction: Optional[float] = None,
                calibrated_prob: Optional[float] = None,
                threshold_type: Optional[str] = None,
                market_source: str = 'Kalshi',
                yes_price: Optional[float] = None,
                no_price: Optional[float] = None,
                home_elo: Optional[float] = None,
                away_elo: Optional[float] = None,
                injury_advantage: Optional[float] = None) -> bool:
        """
        Log a bet to the database
        
        Returns:
            True if logged successfully, False if duplicate
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO trial1306_bets (
                    bet_date, game_date, home_team, away_team, bet_type,
                    predicted_winner, model_probability, fair_probability,
                    calibrated_probability, market_odds, market_source,
                    yes_price, no_price, edge, kelly_fraction, stake_amount,
                    bankroll_at_bet, threshold_type, home_composite_elo,
                    away_composite_elo, injury_matchup_advantage, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                game_date, home_team, away_team, bet_type,
                predicted_winner, model_probability, fair_probability,
                calibrated_prob, market_odds, market_source,
                yes_price, no_price, edge, kelly_fraction, stake_amount,
                bankroll, threshold_type, home_elo, away_elo, injury_advantage,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            print(f"[BET LOGGED] {away_team} @ {home_team} | {predicted_winner} | Edge: {edge:.2%} | Stake: ${stake_amount:.2f}")
            return True
            
        except sqlite3.IntegrityError:
            print(f"[BET DUPLICATE] {away_team} @ {home_team} already logged")
            return False
        finally:
            conn.close()
    
    def grade_bets(self, game_date: str) -> int:
        """
        Grade bets for completed games
        
        Returns:
            Number of bets graded
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get ungraded bets for this date
        cursor.execute("""
            SELECT id, home_team, away_team, predicted_winner, market_odds, stake_amount
            FROM trial1306_bets
            WHERE game_date = ? AND outcome IS NULL
        """, (game_date,))
        
        pending_bets = cursor.fetchall()
        
        if not pending_bets:
            conn.close()
            return 0
        
        graded_count = 0
        
        for bet_id, home, away, predicted, odds, stake in pending_bets:
            # Look up actual result from game_logs
            cursor.execute("""
                SELECT wl, PTS
                FROM game_logs
                WHERE GAME_DATE = ? AND TEAM_ABBREVIATION = ?
            """, (game_date, home))
            
            home_result = cursor.fetchone()
            
            cursor.execute("""
                SELECT wl, PTS
                FROM game_logs
                WHERE GAME_DATE = ? AND TEAM_ABBREVIATION = ?
            """, (game_date, away))
            
            away_result = cursor.fetchone()
            
            if home_result and away_result:
                home_wl, home_pts = home_result
                away_wl, away_pts = away_result
                
                # Determine actual winner
                if home_wl == 'W':
                    actual_winner = home
                else:
                    actual_winner = away
                
                # Calculate profit/loss
                if predicted == actual_winner:
                    outcome = 'WIN'
                    # American odds calculation
                    if odds > 0:
                        profit = stake * (odds / 100)
                    else:
                        profit = stake * (100 / abs(odds))
                else:
                    outcome = 'LOSS'
                    profit = -stake
                
                # Update bet record
                cursor.execute("""
                    UPDATE trial1306_bets
                    SET outcome = ?, actual_winner = ?, 
                        home_score = ?, away_score = ?,
                        profit_loss = ?, graded_at = ?
                    WHERE id = ?
                """, (outcome, actual_winner, home_pts, away_pts, profit, 
                      datetime.now().isoformat(), bet_id))
                
                graded_count += 1
                print(f"[GRADED] {away} @ {home}: {outcome} (P/L: ${profit:+.2f})")
        
        conn.commit()
        conn.close()
        
        if graded_count > 0:
            # Update metrics after grading
            self.update_metrics()
        
        return graded_count
    
    def update_metrics(self):
        """Calculate and store current performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all graded bets
        cursor.execute("""
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as pending,
                AVG(CASE WHEN outcome IS NOT NULL THEN edge END) as avg_edge,
                AVG(CASE WHEN outcome IS NOT NULL THEN kelly_fraction END) as avg_kelly,
                AVG(CASE WHEN outcome IS NOT NULL THEN model_probability END) as avg_model_prob,
                SUM(stake_amount) as total_staked,
                SUM(CASE WHEN outcome IS NOT NULL THEN profit_loss ELSE 0 END) as total_pl,
                AVG(stake_amount) as avg_stake,
                MAX(bankroll_at_bet) as peak_bankroll
            FROM trial1306_bets
        """)
        
        result = cursor.fetchone()
        
        if result and result[0] > 0:
            (total_bets, wins, losses, pending, avg_edge, avg_kelly, avg_model_prob,
             total_staked, total_pl, avg_stake, peak_bankroll) = result
            
            wins = wins or 0
            losses = losses or 0
            graded = wins + losses
            
            if graded > 0:
                win_rate = wins / graded
                roi = (total_pl / total_staked) if total_staked > 0 else 0.0
            else:
                win_rate = 0.0
                roi = 0.0
            
            # Calculate current bankroll (assuming starting bankroll from config)
            starting_bankroll = 2200.0  # From dashboard
            current_bankroll = starting_bankroll + (total_pl or 0)
            
            # Calculate max drawdown
            cursor.execute("""
                SELECT bankroll_at_bet + SUM(profit_loss) OVER (ORDER BY bet_date)
                FROM trial1306_bets
                WHERE outcome IS NOT NULL
            """)
            bankroll_history = [r[0] for r in cursor.fetchall()]
            
            if bankroll_history:
                peak = starting_bankroll
                max_dd = 0.0
                for balance in bankroll_history:
                    if balance > peak:
                        peak = balance
                    drawdown = (peak - balance) / peak if peak > 0 else 0.0
                    if drawdown > max_dd:
                        max_dd = drawdown
            else:
                max_dd = 0.0
            
            # Calculate calibration metrics (Brier score)
            cursor.execute("""
                SELECT model_probability, 
                       CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END as actual
                FROM trial1306_bets
                WHERE outcome IS NOT NULL
            """)
            predictions = cursor.fetchall()
            
            if predictions:
                brier = np.mean([(p[0] - p[1])**2 for p in predictions])
                log_loss_val = -np.mean([
                    p[1] * np.log(max(p[0], 1e-15)) + (1-p[1]) * np.log(max(1-p[0], 1e-15))
                    for p in predictions
                ])
            else:
                brier = None
                log_loss_val = None
            
            # Insert/update metrics
            today = datetime.now().date().isoformat()
            
            cursor.execute("""
                INSERT OR REPLACE INTO performance_metrics (
                    date, total_bets, wins, losses, pending, win_rate,
                    total_staked, total_profit_loss, roi, average_stake,
                    average_edge, average_kelly, average_model_prob,
                    starting_bankroll, current_bankroll, peak_bankroll,
                    max_drawdown, brier_score, log_loss, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                today, total_bets, wins, losses, pending, win_rate,
                total_staked, total_pl, roi, avg_stake,
                avg_edge or 0, avg_kelly or 0, avg_model_prob or 0,
                starting_bankroll, current_bankroll, peak_bankroll or starting_bankroll,
                max_dd, brier, log_loss_val, datetime.now().isoformat()
            ))
            
            conn.commit()
            print(f"[METRICS UPDATED] ROI: {roi:.2%} | Win Rate: {win_rate:.2%} | Bets: {total_bets}")
        
        conn.close()
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM performance_metrics 
            ORDER BY date DESC LIMIT 1
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            return df.iloc[0].to_dict()
        else:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'roi': 0.0,
                'total_profit_loss': 0.0,
                'current_bankroll': 2200.0
            }
    
    def get_recent_bets(self, limit: int = 20) -> pd.DataFrame:
        """Get recent bets with all details"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                game_date, home_team, away_team, bet_type,
                predicted_winner, model_probability, edge,
                market_odds, stake_amount, outcome, profit_loss,
                created_at
            FROM trial1306_bets
            ORDER BY bet_date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def get_performance_by_type(self) -> pd.DataFrame:
        """Get performance breakdown by bet type"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                bet_type,
                COUNT(*) as total_bets,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END) as win_rate,
                SUM(stake_amount) as total_staked,
                SUM(COALESCE(profit_loss, 0)) as total_profit,
                SUM(COALESCE(profit_loss, 0)) / SUM(stake_amount) as roi,
                AVG(edge) as avg_edge
            FROM trial1306_bets
            WHERE outcome IS NOT NULL
            GROUP BY bet_type
            ORDER BY roi DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df


if __name__ == '__main__':
    # Test the tracker
    tracker = BetTracker()
    
    print("\n" + "="*60)
    print("BET TRACKER INITIALIZED")
    print("="*60)
    
    metrics = tracker.get_metrics()
    print(f"\nCurrent Metrics:")
    print(f"  Total Bets: {metrics['total_bets']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  ROI: {metrics['roi']:.2%}")
    print(f"  Bankroll: ${metrics['current_bankroll']:.2f}")
    
    recent = tracker.get_recent_bets(limit=10)
    if len(recent) > 0:
        print(f"\nRecent Bets ({len(recent)}):")
        print(recent[['game_date', 'home_team', 'away_team', 'predicted_winner', 'outcome', 'profit_loss']].to_string())
    else:
        print("\nNo bets logged yet")
