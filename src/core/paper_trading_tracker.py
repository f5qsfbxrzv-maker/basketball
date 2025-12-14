"""
Paper Trading Tracker - Track predictions and calculate performance metrics
"""
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd


class PaperTradingTracker:
    """Track predictions and outcomes for paper trading"""
    
    def __init__(self, db_path: str = 'data/live/nba_betting_data.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create predictions table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_winner TEXT,
                model_probability REAL,
                fair_probability REAL,
                odds REAL,
                edge REAL,
                stake REAL,
                outcome TEXT,
                actual_winner TEXT,
                profit_loss REAL,
                timestamp TEXT NOT NULL,
                UNIQUE(game_date, home_team, away_team, prediction_type)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, 
                      game_date: str,
                      home_team: str,
                      away_team: str,
                      prediction_type: str,
                      predicted_winner: str,
                      model_probability: float,
                      fair_probability: float,
                      odds: float,
                      edge: float,
                      stake: float):
        """Log a prediction to the database (only if not already logged)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if prediction already exists
            cursor.execute("""
                SELECT id FROM paper_predictions
                WHERE game_date = ? AND home_team = ? AND away_team = ? AND prediction_type = ?
            """, (game_date, home_team, away_team, prediction_type))
            
            existing = cursor.fetchone()
            
            if existing:
                print(f"[PAPER TRADING] Skipping duplicate: {away_team} @ {home_team} - {prediction_type}")
                conn.close()
                return
            
            # Insert new prediction
            cursor.execute("""
                INSERT INTO paper_predictions
                (game_date, home_team, away_team, prediction_type, predicted_winner,
                 model_probability, fair_probability, odds, edge, stake, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_date, home_team, away_team, prediction_type, predicted_winner,
                model_probability, fair_probability, odds, edge, stake,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            print(f"[PAPER TRADING] Logged: {away_team} @ {home_team} - {predicted_winner} (edge: {edge:.1%})")
            
        except Exception as e:
            print(f"[ERROR] Failed to log prediction: {e}")
        finally:
            conn.close()
    
    def update_outcomes_from_api(self, game_date: str):
        """Update outcomes for a specific date using game results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get predictions for this date that don't have outcomes
        cursor.execute("""
            SELECT id, game_date, home_team, away_team, prediction_type, predicted_winner, odds, stake
            FROM paper_predictions
            WHERE game_date = ? AND outcome IS NULL
        """, (game_date,))
        
        predictions = cursor.fetchall()
        
        if not predictions:
            print(f"[PAPER TRADING] No pending predictions for {game_date}")
            conn.close()
            return
        
        updated_count = 0
        
        for pred in predictions:
            pred_id, gdate, home, away, pred_type, predicted, odds, stake = pred
            
            # Look up actual result
            cursor.execute("""
                SELECT home_score, away_score, home_team, away_team
                FROM game_results
                WHERE game_date = ? AND home_team = ? AND away_team = ?
            """, (gdate, home, away))
            
            result = cursor.fetchone()
            
            if result:
                home_score, away_score, _, _ = result
                
                # Determine winner
                if home_score > away_score:
                    actual_winner = home
                elif away_score > home_score:
                    actual_winner = away
                else:
                    actual_winner = 'TIE'
                
                # Calculate outcome
                if predicted == actual_winner:
                    outcome = 'WIN'
                    # American odds profit calculation
                    if odds > 0:
                        profit = stake * (odds / 100)
                    else:
                        profit = stake * (100 / abs(odds))
                else:
                    outcome = 'LOSS'
                    profit = -stake
                
                # Update prediction
                cursor.execute("""
                    UPDATE paper_predictions
                    SET outcome = ?, actual_winner = ?, profit_loss = ?
                    WHERE id = ?
                """, (outcome, actual_winner, profit, pred_id))
                
                updated_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"[PAPER TRADING] Updated {updated_count} predictions for {game_date}")
    
    def generate_performance_report(self) -> Dict:
        """Generate performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all settled predictions
        cursor.execute("""
            SELECT outcome, profit_loss, edge, stake, model_probability, fair_probability
            FROM paper_predictions
            WHERE outcome IS NOT NULL
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                'total_bets': 0,
                'win_rate': 0.0,
                'roi': 0.0,
                'total_profit': 0.0,
                'avg_stake': 0.0,
                'brier_score': 0.0,
                'edge_buckets': []
            }
        
        # Calculate metrics
        outcomes = [r[0] for r in rows]
        profits = [r[1] for r in rows]
        edges = [r[2] for r in rows]
        stakes = [r[3] for r in rows]
        
        total_bets = len(rows)
        wins = outcomes.count('WIN')
        win_rate = wins / total_bets if total_bets > 0 else 0.0
        
        total_profit = sum(profits)
        total_staked = sum(stakes)
        roi = total_profit / total_staked if total_staked > 0 else 0.0
        
        avg_stake = total_staked / total_bets if total_bets > 0 else 0.0
        
        # Brier score calculation
        brier_scores = []
        for r in rows:
            outcome, _, _, _, model_prob, _ = r
            actual = 1.0 if outcome == 'WIN' else 0.0
            brier_scores.append((model_prob - actual) ** 2)
        
        brier_score = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0
        
        # Edge bucket analysis
        edge_buckets = self._calculate_edge_buckets(rows)
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'total_profit': total_profit,
            'avg_stake': avg_stake,
            'brier_score': brier_score,
            'edge_buckets': edge_buckets
        }
    
    def _calculate_edge_buckets(self, rows: List) -> List[Dict]:
        """Group predictions by edge ranges"""
        buckets = {
            '0-5%': {'count': 0, 'wins': 0, 'total_profit': 0, 'total_stake': 0},
            '5-8%': {'count': 0, 'wins': 0, 'total_profit': 0, 'total_stake': 0},
            '8-10%': {'count': 0, 'wins': 0, 'total_profit': 0, 'total_stake': 0},
            '10-15%': {'count': 0, 'wins': 0, 'total_profit': 0, 'total_stake': 0},
            '15%+': {'count': 0, 'wins': 0, 'total_profit': 0, 'total_stake': 0}
        }
        
        for row in rows:
            outcome, profit, edge, stake = row[0], row[1], row[2], row[3]
            
            # Determine bucket
            if edge < 0.05:
                bucket_key = '0-5%'
            elif edge < 0.08:
                bucket_key = '5-8%'
            elif edge < 0.10:
                bucket_key = '8-10%'
            elif edge < 0.15:
                bucket_key = '10-15%'
            else:
                bucket_key = '15%+'
            
            bucket = buckets[bucket_key]
            bucket['count'] += 1
            if outcome == 'WIN':
                bucket['wins'] += 1
            bucket['total_profit'] += profit
            bucket['total_stake'] += stake
        
        # Convert to list format
        result = []
        for range_str, data in buckets.items():
            if data['count'] > 0:
                roi = data['total_profit'] / data['total_stake'] if data['total_stake'] > 0 else 0.0
                result.append({
                    'range': range_str,
                    'count': data['count'],
                    'wins': data['wins'],
                    'roi': roi
                })
        
        return result
    
    def get_recent_predictions(self, limit: int = 50) -> pd.DataFrame:
        """Get recent predictions as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT game_date, home_team, away_team, prediction_type, predicted_winner,
                   model_probability, fair_probability, odds, edge, stake, outcome, profit_loss
            FROM paper_predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df


if __name__ == '__main__':
    # Test the tracker
    tracker = PaperTradingTracker()
    
    print("\n=== TESTING PAPER TRADING TRACKER ===\n")
    
    # Generate report
    report = tracker.generate_performance_report()
    
    print(f"Total Bets: {report['total_bets']}")
    print(f"Win Rate: {report['win_rate']:.1%}")
    print(f"ROI: {report['roi']:.1%}")
    print(f"Total Profit: ${report['total_profit']:.2f}")
    print(f"Brier Score: {report['brier_score']:.4f}")
    
    if report['edge_buckets']:
        print("\nEdge Buckets:")
        for bucket in report['edge_buckets']:
            print(f"  {bucket['range']}: {bucket['count']} bets, {bucket['wins']} wins, ROI: {bucket['roi']:.1%}")
