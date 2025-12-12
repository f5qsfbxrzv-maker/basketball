"""
Advanced Feature Calculator for NBA Betting Models
Implements professional-level interaction features:
1. Foul Synergy (totals predictor)
2. Chaos Factor (ATS predictor via transition opportunities)
3. 3PT Variance (cover metric)

Uses ROLLING 10-game averages (not season) for recency bias
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta


from v2.database_paths import NBA_BETTING_DB

class AdvancedFeatureCalculator:
    """
    Calculates advanced interaction features using rolling game logs
    
    Philosophy:
    - Teams change style throughout the year (use L10 not season average)
    - Matchup matters: Offense draws fouls vs Defense commits them
    - Chaos = Steals × Opponent Turnovers (transition points)
    - 3PT variance drives unpredictable outcomes (cover metric)
    """
    
    def __init__(self, db_path: str = str(NBA_BETTING_DB), window: int = 10):
        """
        Args:
            db_path: Path to nba_betting_data.db
            window: Rolling window size (default 10 games)
        """
        self.db_path = db_path
        self.window = window
        self._game_logs_cache: Optional[pd.DataFrame] = None
        self._indexed_logs: Optional[Dict] = None  # Team-indexed for fast lookups
    
    def _load_game_logs(self) -> pd.DataFrame:
        """Load all game logs with necessary columns"""
        if self._game_logs_cache is not None:
            return self._game_logs_cache
        
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT 
                TEAM_ABBREVIATION as team,
                GAME_DATE as game_date,
                FTA, POSS_EST,
                STL, TOV,
                FG3A,
                PF as fouls_committed
            FROM game_logs
            WHERE POSS_EST > 0
            ORDER BY GAME_DATE
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert game_date to datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Calculate rates per 100 possessions
        df['fta_rate'] = (df['FTA'] / df['POSS_EST']) * 100
        df['tov_pct'] = (df['TOV'] / df['POSS_EST']) * 100
        df['stl_pct'] = (df['STL'] / df['POSS_EST']) * 100
        df['foul_rate'] = (df['fouls_committed'] / df['POSS_EST']) * 100
        df['3pa_per_100'] = (df['FG3A'] / df['POSS_EST']) * 100
        
        self._game_logs_cache = df
        
        # Build index for fast lookups
        self._build_team_index()
        
        return df
    
    def _build_team_index(self):
        """Build team-indexed dictionary for fast rolling stat lookups"""
        if self._game_logs_cache is None:
            return
        
        self._indexed_logs = {}
        for team in self._game_logs_cache['team'].unique():
            team_logs = self._game_logs_cache[
                self._game_logs_cache['team'] == team
            ].copy()
            team_logs = team_logs.sort_values('game_date')
            self._indexed_logs[team] = team_logs
    
    def get_rolling_stats(
        self, 
        team: str, 
        as_of_date: str,
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get rolling averages for a team as of a specific date
        
        Args:
            team: Team abbreviation (e.g., 'LAL')
            as_of_date: Date to compute rolling stats before
            window: Rolling window (defaults to self.window)
        
        Returns:
            Dict with keys: fta_rate, tov_pct, stl_pct, foul_rate, 3pa_per_100
        """
        if window is None:
            window = self.window
        
        # Ensure data is loaded and indexed
        if self._indexed_logs is None:
            self._load_game_logs()
        
        # Use indexed lookup instead of filtering full dataframe
        team_logs = self._indexed_logs.get(team)
        if team_logs is None:
            # Return league averages as fallback
            return {
                'fta_rate': 22.0,      # League avg ~22 FTA per 100 poss
                'tov_pct': 14.0,       # League avg ~14 TOV per 100 poss
                'stl_pct': 8.0,        # League avg ~8 STL per 100 poss
                'foul_rate': 20.0,     # League avg ~20 fouls per 100 poss
                '3pa_per_100': 35.0,   # League avg ~35 3PA per 100 poss
            }
        
        as_of = pd.to_datetime(as_of_date)
        
        # Get games before this date (already sorted by game_date)
        # Use boolean indexing + numpy slicing for speed
        mask = team_logs['game_date'] < as_of
        recent_indices = np.where(mask)[0]
        
        if len(recent_indices) == 0:
            # Return league averages as fallback
            return {
                'fta_rate': 22.0,
                'tov_pct': 14.0,
                'stl_pct': 8.0,
                'foul_rate': 20.0,
                '3pa_per_100': 35.0,
            }
        
        # Get last N games (window size)
        start_idx = max(0, recent_indices[-1] - window + 1)
        end_idx = recent_indices[-1] + 1
        team_games = team_logs.iloc[start_idx:end_idx]
        
        return {
            'fta_rate': team_games['fta_rate'].mean(),
            'tov_pct': team_games['tov_pct'].mean(),
            'stl_pct': team_games['stl_pct'].mean(),
            'foul_rate': team_games['foul_rate'].mean(),
            '3pa_per_100': team_games['3pa_per_100'].mean(),
        }
    
    def calculate_advanced_features(
        self,
        home_team: str,
        away_team: str,
        game_date: str
    ) -> Dict[str, float]:
        """
        Calculate all advanced interaction features for a matchup
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Date of game (YYYY-MM-DD)
        
        Returns:
            Dict with 7 advanced features:
            - foul_synergy_home: Home draws fouls × Away commits fouls
            - foul_synergy_away: Away draws fouls × Home commits fouls
            - total_foul_environment: Sum of both synergies (totals predictor)
            - chaos_home: Home steals × Away turnovers (transition advantage)
            - chaos_away: Away steals × Home turnovers
            - net_chaos: Chaos differential (ATS predictor)
            - vol_3p_diff: 3PA volume differential (variance/cover metric)
        """
        # Get rolling stats for both teams
        home_stats = self.get_rolling_stats(home_team, game_date)
        away_stats = self.get_rolling_stats(away_team, game_date)
        
        # ===================================================================
        # TOTALS FEATURES: Foul Synergy
        # ===================================================================
        # Logic: Offense draws fouls (FTA rate) × Defense commits fouls
        # High foul rate = more free throws = higher total
        
        foul_synergy_home = (
            home_stats['fta_rate'] * away_stats['foul_rate']
        ) / 100  # Normalize interaction
        
        foul_synergy_away = (
            away_stats['fta_rate'] * home_stats['foul_rate']
        ) / 100
        
        total_foul_environment = foul_synergy_home + foul_synergy_away
        
        # ===================================================================
        # ATS FEATURES: Chaos Factor (Transition Opportunities)
        # ===================================================================
        # Logic: Steals create fast break points (high-efficiency)
        # Team with positive Net_Chaos gets more easy buckets = covers spread
        
        chaos_home = home_stats['stl_pct'] * away_stats['tov_pct']
        chaos_away = away_stats['stl_pct'] * home_stats['tov_pct']
        net_chaos = chaos_home - chaos_away
        
        # Positive Net_Chaos = Home team gets more transition opportunities
        
        # ===================================================================
        # COVER FEATURES: 3PT Volume Variance
        # ===================================================================
        # Logic: High 3PA volume creates variance (makes/misses swing)
        # Large differential = one team lives/dies by 3PT shooting
        # This increases unpredictability → harder to predict cover
        
        vol_3p_diff = home_stats['3pa_per_100'] - away_stats['3pa_per_100']
        
        # Positive = Home shoots more 3s (more variance in home performance)
        
        return {
            # Foul Synergy (Totals)
            'foul_synergy_home': foul_synergy_home,
            'foul_synergy_away': foul_synergy_away,
            'total_foul_environment': total_foul_environment,
            
            # Chaos Factor (ATS)
            'chaos_home': chaos_home,
            'chaos_away': chaos_away,
            'net_chaos': net_chaos,
            
            # 3PT Variance (Cover)
            'vol_3p_diff': vol_3p_diff,
        }
    
    def get_feature_names(self) -> list[str]:
        """Return list of all feature names for schema documentation"""
        return [
            'foul_synergy_home',
            'foul_synergy_away',
            'total_foul_environment',
            'chaos_home',
            'chaos_away',
            'net_chaos',
            'vol_3p_diff',
        ]
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Return feature descriptions for documentation"""
        return {
            'foul_synergy_home': 'Home FTA rate × Away foul rate (draws fouls vs commits)',
            'foul_synergy_away': 'Away FTA rate × Home foul rate',
            'total_foul_environment': 'Sum of both synergies (totals predictor)',
            'chaos_home': 'Home steals × Away turnovers (transition advantage)',
            'chaos_away': 'Away steals × Home turnovers',
            'net_chaos': 'Chaos differential (positive = Home transition edge)',
            'vol_3p_diff': '3PA volume differential (variance/cover metric)',
        }


def example_usage():
    """Demonstration of advanced feature calculator"""
    calc = AdvancedFeatureCalculator(window=10)  # Last 10 games
    
    # Example: Lakers vs Warriors on 2025-01-15
    features = calc.calculate_advanced_features(
        home_team='LAL',
        away_team='GSW',
        game_date='2025-01-15'
    )
    
    print("Advanced Features for LAL vs GSW:")
    print("=" * 60)
    
    for name, value in features.items():
        desc = calc.get_feature_descriptions()[name]
        print(f"{name:25s}: {value:8.3f}  ({desc})")
    
    print("\n" + "=" * 60)
    print("Interpretation:")
    print("=" * 60)
    
    if features['total_foul_environment'] > 5.0:
        print("✓ HIGH foul environment → Expect higher total")
    
    if features['net_chaos'] > 0.5:
        print("✓ Home team has transition advantage → Favors home ATS")
    elif features['net_chaos'] < -0.5:
        print("✓ Away team has transition advantage → Favors away ATS")
    
    if abs(features['vol_3p_diff']) > 10:
        print("✓ Large 3PT volume differential → High variance (harder to predict)")


if __name__ == "__main__":
    example_usage()
