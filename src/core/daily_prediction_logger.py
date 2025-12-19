"""
Daily Prediction Logger for Trial 1306
Logs ALL predictions (bet or not) for model performance tracking
Tracks opening vs closing lines, only logs once per game
"""

import sqlite3
from datetime import datetime
from typing import Dict, Optional
import pandas as pd


class DailyPredictionLogger:
    """Log all predictions for performance tracking"""
    
    def __init__(self, db_path: str = 'data/live/nba_betting_data.db'):
        self.db_path = db_path
        self._ensure_table()
    
    def _ensure_table(self):
        """Create prediction logging table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TIMESTAMP NOT NULL,
                game_date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                
                -- Model predictions
                model_home_prob REAL NOT NULL,
                model_away_prob REAL NOT NULL,
                model_version TEXT DEFAULT 'Trial1306',
                
                -- Opening odds (first seen)
                opening_home_ml INTEGER,
                opening_away_ml INTEGER,
                opening_timestamp TIMESTAMP,
                opening_source TEXT,
                
                -- Closing odds (last seen before game)
                closing_home_ml INTEGER,
                closing_away_ml INTEGER,
                closing_timestamp TIMESTAMP,
                closing_source TEXT,
                
                -- Market comparison
                opening_home_prob REAL,
                opening_away_prob REAL,
                closing_home_prob REAL,
                closing_away_prob REAL,
                
                -- Edges
                opening_home_edge REAL,
                opening_away_edge REAL,
                closing_home_edge REAL,
                closing_away_edge REAL,
                
                -- Best bet at time of prediction
                best_pick TEXT,
                best_edge REAL,
                qualified_bet BOOLEAN DEFAULT 0,
                stake_if_bet REAL,
                
                -- Key features for analysis
                home_elo REAL,
                away_elo REAL,
                injury_advantage REAL,
                rest_advantage REAL,
                
                -- Actual outcome
                actual_winner TEXT,
                home_score INTEGER,
                away_score INTEGER,
                graded_at TIMESTAMP,
                
                -- Model performance metrics
                log_loss REAL,
                brier_score REAL,
                beat_opening BOOLEAN,
                beat_closing BOOLEAN,
                
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP,
                
                UNIQUE(game_date, home_team, away_team)
            )
        """)
        
        conn.commit()
        conn.close()
        print("[DAILY LOGGER] Prediction tracking table initialized")
    
    def log_prediction(self,
                      game_date: str,
                      home_team: str,
                      away_team: str,
                      model_home_prob: float,
                      model_away_prob: float,
                      home_odds: int,
                      away_odds: int,
                      odds_source: str,
                      best_bet: Optional[Dict] = None,
                      features: Optional[Dict] = None) -> bool:
        """
        Log a prediction (only once per game)
        
        Returns:
            True if logged, False if already exists (then update closing odds)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if prediction already exists
            cursor.execute("""
                SELECT id, opening_home_ml, opening_away_ml
                FROM daily_predictions
                WHERE game_date = ? AND home_team = ? AND away_team = ?
            """, (game_date, home_team, away_team))
            
            existing = cursor.fetchone()
            
            # Calculate market probabilities
            opening_home_prob = self._odds_to_prob(home_odds)
            opening_away_prob = self._odds_to_prob(away_odds)
            
            # Calculate edges
            opening_home_edge = model_home_prob - opening_home_prob
            opening_away_edge = model_away_prob - opening_away_prob
            
            if existing:
                # Update closing odds (this is line movement tracking)
                pred_id = existing[0]
                
                cursor.execute("""
                    UPDATE daily_predictions
                    SET closing_home_ml = ?,
                        closing_away_ml = ?,
                        closing_timestamp = ?,
                        closing_source = ?,
                        closing_home_prob = ?,
                        closing_away_prob = ?,
                        closing_home_edge = ?,
                        closing_away_edge = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    home_odds, away_odds, datetime.now().isoformat(), odds_source,
                    opening_home_prob, opening_away_prob,
                    opening_home_edge, opening_away_edge,
                    datetime.now().isoformat(), pred_id
                ))
                
                conn.commit()
                print(f"[DAILY LOGGER] Updated closing line: {away_team} @ {home_team}")
                return False
            
            else:
                # Insert new prediction (first time seeing this game)
                best_pick = best_bet.get('pick') if best_bet else None
                best_edge = best_bet.get('edge') if best_bet else None
                qualified = best_bet.get('qualifies', False) if best_bet else False
                stake = best_bet.get('stake', 0.0) if best_bet else 0.0
                
                home_elo = features.get('home_composite_elo') if features else None
                away_elo = features.get('away_composite_elo') if features else None
                injury_adv = features.get('injury_matchup_advantage') if features else None
                rest_adv = features.get('rest_advantage', 0.0) if features else None
                
                cursor.execute("""
                    INSERT INTO daily_predictions (
                        prediction_date, game_date, home_team, away_team,
                        model_home_prob, model_away_prob,
                        opening_home_ml, opening_away_ml, opening_timestamp, opening_source,
                        opening_home_prob, opening_away_prob,
                        opening_home_edge, opening_away_edge,
                        best_pick, best_edge, qualified_bet, stake_if_bet,
                        home_elo, away_elo, injury_advantage, rest_advantage,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(), game_date, home_team, away_team,
                    model_home_prob, model_away_prob,
                    home_odds, away_odds, datetime.now().isoformat(), odds_source,
                    opening_home_prob, opening_away_prob,
                    opening_home_edge, opening_away_edge,
                    best_pick, best_edge, qualified, stake,
                    home_elo, away_elo, injury_adv, rest_adv,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                print(f"[DAILY LOGGER] ✅ Logged: {away_team} @ {home_team} | Model: {model_home_prob:.1%} vs Market: {opening_home_prob:.1%}")
                return True
        
        except sqlite3.IntegrityError as e:
            print(f"[DAILY LOGGER] Duplicate prevented: {away_team} @ {home_team}")
            return False
        finally:
            conn.close()
    
    def grade_predictions(self, game_date: str) -> int:
        """
        Grade predictions after games finish
        
        Returns:
            Number of predictions graded
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get ungraded predictions
        cursor.execute("""
            SELECT id, home_team, away_team, model_home_prob, model_away_prob
            FROM daily_predictions
            WHERE game_date = ? AND actual_winner IS NULL
        """, (game_date,))
        
        pending = cursor.fetchall()
        
        if not pending:
            conn.close()
            return 0
        
        graded_count = 0
        
        for pred_id, home, away, model_home_prob, model_away_prob in pending:
            # Look up actual result
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
                
                actual_winner = home if home_wl == 'W' else away
                
                # Calculate Brier score
                actual_home = 1.0 if home_wl == 'W' else 0.0
                brier = (model_home_prob - actual_home) ** 2
                
                # Calculate log loss
                prob_actual = model_home_prob if home_wl == 'W' else model_away_prob
                log_loss = -1 * (actual_home * pd.np.log(max(model_home_prob, 1e-15)) + 
                                 (1-actual_home) * pd.np.log(max(model_away_prob, 1e-15)))
                
                # Get edges to check if beat market
                cursor.execute("""
                    SELECT opening_home_edge, opening_away_edge, 
                           closing_home_edge, closing_away_edge
                    FROM daily_predictions
                    WHERE id = ?
                """, (pred_id,))
                
                edges = cursor.fetchone()
                if edges:
                    op_h_edge, op_a_edge, cl_h_edge, cl_a_edge = edges
                    
                    # Did model's prediction beat opening market?
                    beat_opening = (home_wl == 'W' and op_h_edge > 0) or (away_wl == 'W' and op_a_edge > 0)
                    
                    # Did model's prediction beat closing market?
                    beat_closing = False
                    if cl_h_edge is not None and cl_a_edge is not None:
                        beat_closing = (home_wl == 'W' and cl_h_edge > 0) or (away_wl == 'W' and cl_a_edge > 0)
                else:
                    beat_opening = False
                    beat_closing = False
                
                # Update prediction
                cursor.execute("""
                    UPDATE daily_predictions
                    SET actual_winner = ?,
                        home_score = ?,
                        away_score = ?,
                        graded_at = ?,
                        log_loss = ?,
                        brier_score = ?,
                        beat_opening = ?,
                        beat_closing = ?
                    WHERE id = ?
                """, (actual_winner, home_pts, away_pts, datetime.now().isoformat(),
                      log_loss, brier, beat_opening, beat_closing, pred_id))
                
                graded_count += 1
                print(f"[GRADED] {away} @ {home}: {actual_winner} won | Brier: {brier:.4f}")
        
        conn.commit()
        conn.close()
        
        return graded_count
    
    def get_model_performance(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get model performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                COUNT(*) as total_predictions,
                AVG(brier_score) as avg_brier,
                AVG(log_loss) as avg_log_loss,
                SUM(CASE WHEN beat_opening = 1 THEN 1 ELSE 0 END) as beat_opening_count,
                SUM(CASE WHEN beat_closing = 1 THEN 1 ELSE 0 END) as beat_closing_count,
                AVG(ABS(opening_home_edge)) as avg_edge_vs_opening,
                AVG(ABS(closing_home_edge)) as avg_edge_vs_closing
            FROM daily_predictions
            WHERE actual_winner IS NOT NULL
        """
        
        if start_date:
            query += f" AND game_date >= '{start_date}'"
        if end_date:
            query += f" AND game_date <= '{end_date}'"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_line_movement_analysis(self) -> pd.DataFrame:
        """Analyze how lines moved from opening to closing"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                game_date, home_team, away_team,
                opening_home_ml, closing_home_ml,
                opening_away_ml, closing_away_ml,
                model_home_prob, opening_home_prob, closing_home_prob,
                actual_winner,
                (closing_home_ml - opening_home_ml) as home_line_movement,
                (closing_away_ml - opening_away_ml) as away_line_movement
            FROM daily_predictions
            WHERE closing_home_ml IS NOT NULL
            ORDER BY game_date DESC
            LIMIT 100
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def _odds_to_prob(self, american_odds: int) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)


if __name__ == '__main__':
    # Test the logger
    logger = DailyPredictionLogger()
    
    print("\n" + "="*60)
    print("DAILY PREDICTION LOGGER INITIALIZED")
    print("="*60)
    
    # Check if there are any predictions
    conn = sqlite3.connect('data/live/nba_betting_data.db')
    count = conn.execute("SELECT COUNT(*) FROM daily_predictions").fetchone()[0]
    conn.close()
    
    print(f"\nTotal predictions logged: {count}")
    
    if count > 0:
        perf = logger.get_model_performance()
        print("\nModel Performance:")
        print(perf.to_string())
