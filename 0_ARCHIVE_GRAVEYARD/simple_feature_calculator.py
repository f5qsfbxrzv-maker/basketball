"""
Simple Feature Calculator - Returns exact 36 features the model needs
Uses data/live/nba_betting_data.db directly
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

class SimpleFeatureCalculator:
    """Calculate the exact 36 features needed by moneyline_model_enhanced.pkl"""
    
    # Team name to abbreviation mapping
    TEAM_MAP = {
        'Hawks': 'ATL', 'Celtics': 'BOS', 'Nets': 'BKN', 'Hornets': 'CHA',
        'Bulls': 'CHI', 'Cavaliers': 'CLE', 'Mavericks': 'DAL', 'Nuggets': 'DEN',
        'Pistons': 'DET', 'Warriors': 'GSW', 'Rockets': 'HOU', 'Pacers': 'IND',
        'Clippers': 'LAC', 'Lakers': 'LAL', 'Grizzlies': 'MEM', 'Heat': 'MIA',
        'Bucks': 'MIL', 'Timberwolves': 'MIN', 'Pelicans': 'NOP', 'Knicks': 'NYK',
        'Thunder': 'OKC', 'Magic': 'ORL', '76ers': 'PHI', 'Suns': 'PHX',
        'Trail Blazers': 'POR', 'Kings': 'SAC', 'Spurs': 'SAS', 'Raptors': 'TOR',
        'Jazz': 'UTA', 'Wizards': 'WAS',
        # Also accept abbreviations directly
        'ATL': 'ATL', 'BOS': 'BOS', 'BKN': 'BKN', 'CHA': 'CHA', 'CHI': 'CHI',
        'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GSW': 'GSW',
        'HOU': 'HOU', 'IND': 'IND', 'LAC': 'LAC', 'LAL': 'LAL', 'MEM': 'MEM',
        'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN', 'NOP': 'NOP', 'NYK': 'NYK',
        'OKC': 'OKC', 'ORL': 'ORL', 'PHI': 'PHI', 'PHX': 'PHX', 'POR': 'POR',
        'SAC': 'SAC', 'SAS': 'SAS', 'TOR': 'TOR', 'UTA': 'UTA', 'WAS': 'WAS'
    }
    
    def __init__(self, db_path: str = 'data/live/nba_betting_data.db'):
        self.db_path = db_path
    
    def calculate_game_features(self, home_team: str, away_team: str, game_date: str = None) -> Dict[str, float]:
        """
        Calculate 36 features for a game
        
        Returns dict with keys:
        composite_elo_diff, off_elo_diff, def_elo_diff,
        home_composite_elo, away_composite_elo,
        off_rating_diff, def_rating_diff,
        home_off_rating, away_off_rating,
        home_def_rating, away_def_rating,
        pace_diff, avg_pace, home_pace, away_pace,
        home_efg, away_efg, efg_diff,
        home_tov, away_tov,
        home_orb, away_orb,
        home_ftr, away_ftr,
        home_rest_days, away_rest_days, rest_advantage,
        home_back_to_back, away_back_to_back, both_rested,
        predicted_pace, pace_up_game, pace_down_game,
        altitude_game,
        off_def_matchup_home, off_def_matchup_away
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert team names to abbreviations
        home_abb = self.TEAM_MAP.get(home_team, home_team)
        away_abb = self.TEAM_MAP.get(away_team, away_team)
        
        conn = sqlite3.connect(self.db_path)
        
        # Get ELO ratings
        home_elo = self._get_elo(conn, home_abb, game_date)
        away_elo = self._get_elo(conn, away_abb, game_date)
        
        # Get team stats (ratings, pace, four factors)
        home_stats = self._get_team_stats(conn, home_abb, game_date)
        away_stats = self._get_team_stats(conn, away_abb, game_date)
        
        # Get rest info
        home_rest = self._get_rest_days(conn, home_abb, game_date)
        away_rest = self._get_rest_days(conn, away_abb, game_date)
        
        conn.close()
        
        # Calculate features
        features = {}
        
        # ELO features
        features['composite_elo_diff'] = home_elo['composite'] - away_elo['composite']
        features['off_elo_diff'] = home_elo['offensive'] - away_elo['offensive']
        features['def_elo_diff'] = home_elo['defensive'] - away_elo['defensive']
        features['home_composite_elo'] = home_elo['composite']
        features['away_composite_elo'] = away_elo['composite']
        
        # Rating features
        features['off_rating_diff'] = home_stats['off_rating'] - away_stats['off_rating']
        features['def_rating_diff'] = home_stats['def_rating'] - away_stats['def_rating']
        features['home_off_rating'] = home_stats['off_rating']
        features['away_off_rating'] = away_stats['off_rating']
        features['home_def_rating'] = home_stats['def_rating']
        features['away_def_rating'] = away_stats['def_rating']
        
        # Pace features
        features['pace_diff'] = home_stats['pace'] - away_stats['pace']
        features['avg_pace'] = (home_stats['pace'] + away_stats['pace']) / 2
        features['home_pace'] = home_stats['pace']
        features['away_pace'] = away_stats['pace']
        
        # Four Factors
        features['home_efg'] = home_stats['efg']
        features['away_efg'] = away_stats['efg']
        features['efg_diff'] = home_stats['efg'] - away_stats['efg']
        features['home_tov'] = home_stats['tov_pct']
        features['away_tov'] = away_stats['tov_pct']
        features['home_orb'] = home_stats['orb_pct']
        features['away_orb'] = away_stats['orb_pct']
        features['home_ftr'] = home_stats['ftr']
        features['away_ftr'] = away_stats['ftr']
        
        # Rest features
        features['home_rest_days'] = home_rest['rest_days']
        features['away_rest_days'] = away_rest['rest_days']
        features['rest_advantage'] = home_rest['rest_days'] - away_rest['rest_days']
        features['home_back_to_back'] = 1 if home_rest['rest_days'] == 1 else 0
        features['away_back_to_back'] = 1 if away_rest['rest_days'] == 1 else 0
        features['both_rested'] = 1 if (home_rest['rest_days'] >= 2 and away_rest['rest_days'] >= 2) else 0
        
        # Pace predictions
        features['predicted_pace'] = features['avg_pace']
        features['pace_up_game'] = 1 if features['avg_pace'] > 101 else 0
        features['pace_down_game'] = 1 if features['avg_pace'] < 97 else 0
        
        # Altitude (Denver only)
        features['altitude_game'] = 1 if home_abb == 'DEN' else 0
        
        # Matchup features (offense vs defense)
        features['off_def_matchup_home'] = home_stats['off_rating'] * away_stats['def_rating'] / 110
        features['off_def_matchup_away'] = away_stats['off_rating'] * home_stats['def_rating'] / 110
        
        return features
    
    def _get_elo(self, conn, team: str, as_of_date: str) -> Dict[str, float]:
        """Get ELO ratings from database"""
        query = """
        SELECT composite_elo, off_elo, def_elo
        FROM elo_ratings
        WHERE team = ? AND game_date <= ?
        ORDER BY game_date DESC
        LIMIT 1
        """
        result = pd.read_sql_query(query, conn, params=(team, as_of_date))
        
        if len(result) == 0:
            # Return default ELO
            return {'composite': 1500, 'offensive': 1500, 'defensive': 1500}
        
        return {
            'composite': float(result['composite_elo'].iloc[0]),
            'offensive': float(result['off_elo'].iloc[0]),
            'defensive': float(result['def_elo'].iloc[0])
        }
    
    def _get_team_stats(self, conn, team: str, as_of_date: str) -> Dict[str, float]:
        """Get team stats from game_advanced_stats (last 10 games avg)"""
        query = """
        SELECT 
            off_rating, def_rating, pace,
            efg_pct as efg, tov_pct, orb_pct, fta_rate as ftr
        FROM game_advanced_stats
        WHERE team_abb = ? AND game_date < ?
        ORDER BY game_date DESC
        LIMIT 10
        """
        result = pd.read_sql_query(query, conn, params=(team, as_of_date))
        
        if len(result) == 0:
            # Return league averages
            return {
                'off_rating': 112.0, 'def_rating': 112.0, 'pace': 99.0,
                'efg': 0.54, 'tov_pct': 13.0, 'orb_pct': 25.0, 'ftr': 0.22
            }
        
        return {
            'off_rating': float(result['off_rating'].mean()),
            'def_rating': float(result['def_rating'].mean()),
            'pace': float(result['pace'].mean()),
            'efg': float(result['efg'].mean()),
            'tov_pct': float(result['tov_pct'].mean()),
            'orb_pct': float(result['orb_pct'].mean()),
            'ftr': float(result['ftr'].mean())
        }
    
    def _get_rest_days(self, conn, team: str, game_date: str) -> Dict[str, int]:
        """Calculate rest days from game_logs"""
        query = """
        SELECT GAME_DATE
        FROM game_logs
        WHERE TEAM_ABBREVIATION = ? AND GAME_DATE < ?
        ORDER BY GAME_DATE DESC
        LIMIT 1
        """
        result = pd.read_sql_query(query, conn, params=(team, game_date))
        
        if len(result) == 0:
            return {'rest_days': 2}  # Default to rested
        
        last_game = pd.to_datetime(result['GAME_DATE'].iloc[0])
        current_game = pd.to_datetime(game_date)
        days_diff = (current_game - last_game).days
        
        return {'rest_days': max(1, days_diff)}
