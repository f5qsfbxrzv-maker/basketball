"""
EWMA (Exponentially Weighted Moving Average) Feature Calculator
Implements "World Class" recency bias with metric-specific decay rates

Philosophy:
- Recent games matter MORE than old games (not equal weight)
- Pace changes fast (Span=7) vs Defense changes slow (Span=15)
- Bayesian priors prevent early-season overreaction
- Never hits zero (smooth fade into history)
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime


from v2.database_paths import NBA_BETTING_DB

class EWMAFeatureCalculator:
    """
    Calculate exponentially weighted stats with metric-specific decay rates
    
    Decay Rate Philosophy:
    - Pace: Span=7 (teams can decide to run tomorrow)
    - Turnovers: Span=10 (form-based funk lasts 2-3 weeks)
    - Defense: Span=15 (system-based, chemistry takes time)
    - 3PT%: Span=30 (high variance, mostly luck short-term)
    - General: Span=10 (balanced default)
    
    Bayesian Prior:
    - Seed each team with 10 "ghost games" of league average
    - Game 1: (1 real + 10 ghost) = muted impact
    - Game 50: Ghost games decayed to near zero
    """
    
    # League average priors (2024-25 season)
    LEAGUE_PRIORS = {
        'pace': 99.0,
        'efg': 0.550,
        'tov_pct': 14.0,
        'fta_rate': 22.0,
        'stl_pct': 8.0,
        'foul_rate': 20.0,
        '3pa_per_100': 35.0,
        '3p_pct': 0.360,
    }
    
    # Metric-specific span settings (World Class methodology)
    # Philosophy: Different stats stabilize at different rates
    DECAY_SPANS = {
        # VOLATILE (Span 5-8): Changes happen overnight
        'pace': 7,           # Teams decide to run fast → immediate change
        
        # FORM-BASED (Span 10-15): Effort/focus slumps last 2-3 weeks
        'tov_pct': 10,       # Turnover funk is temporary
        'fta_rate': 10,      # Free throw aggression
        'stl_pct': 10,       # Steal rate (defensive effort)
        'foul_rate': 10,     # Foul commitment
        'efg': 12,           # Effective field goal (form-based)
        '3pa_per_100': 15,   # 3PT volume decision
        
        # IDENTITY (Span 20-30): System stats don't change easily
        # Defense is chemistry-based, takes time to build/break
        
        # NOISE (Span 30-40): High variance, trust season average
        '3p_pct': 40,        # 3PT% is mostly luck short-term (DON'T CHASE HOT STREAKS!)
    }
    
    GHOST_GAMES = 10  # Bayesian prior strength
    
    def __init__(self, db_path: str = str(NBA_BETTING_DB)):
        """
        Args:
            db_path: Path to nba_betting_data.db
        """
        self.db_path = db_path
        self._game_logs_cache: Optional[pd.DataFrame] = None
        self._ewma_cache: Optional[Dict[str, pd.DataFrame]] = None
        self._team_logs_cache: Dict[str, pd.DataFrame] = {}  # Pre-grouped team logs for fast lookups
    
    def _load_and_prepare_game_logs(self) -> pd.DataFrame:
        """Load game logs and calculate per-100 stats"""
        if self._game_logs_cache is not None:
            return self._game_logs_cache
        
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT 
                TEAM_ABBREVIATION as team,
                GAME_DATE as game_date,
                GAME_ID as game_id,
                FTA, POSS_EST,
                STL, TOV,
                FG3A, FG3M,
                PF as fouls_committed,
                PTS,
                FGM, FGA
            FROM game_logs
            WHERE POSS_EST > 0
            ORDER BY GAME_DATE
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert date
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Calculate per-100 possession stats
        df['fta_rate'] = (df['FTA'] / df['POSS_EST']) * 100
        df['tov_pct'] = (df['TOV'] / df['POSS_EST']) * 100
        df['stl_pct'] = (df['STL'] / df['POSS_EST']) * 100
        df['foul_rate'] = (df['fouls_committed'] / df['POSS_EST']) * 100
        df['3pa_per_100'] = (df['FG3A'] / df['POSS_EST']) * 100
        df['3p_pct'] = df['FG3M'] / df['FG3A'].replace(0, np.nan)
        df['efg'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, np.nan)
        
        # Estimate pace (possessions per 48 minutes, assumes 240 mins)
        df['pace'] = (df['POSS_EST'] / 240) * 48
        
        # Sort by team and date for EWMA
        df = df.sort_values(['team', 'game_date'])
        
        self._game_logs_cache = df
        
        # Pre-group by team for fast lookups
        print("   Pre-grouping by team for fast EWMA lookups...")
        for team in df['team'].unique():
            self._team_logs_cache[team] = df[df['team'] == team].copy()
        print(f"   Cached {len(self._team_logs_cache)} teams")
        
        return df
    
    def _add_bayesian_priors(self, df: pd.DataFrame, metric: str) -> pd.Series:
        """
        Add ghost games of league average to prevent early-season overreaction
        
        Args:
            df: Team-specific game log sorted by date
            metric: Metric name (e.g., 'pace', 'def_rating')
        
        Returns:
            Series with priors prepended
        """
        prior_value = self.LEAGUE_PRIORS.get(metric, df[metric].mean())
        
        # Create ghost games DataFrame
        ghost_games = pd.Series([prior_value] * self.GHOST_GAMES)
        
        # Prepend to actual data
        return pd.concat([ghost_games, df[metric]], ignore_index=True)
    
    def _calculate_ewma_by_team(self, metric: str) -> Dict[str, pd.DataFrame]:
        """
        Calculate EWMA for a specific metric across all teams
        
        Args:
            metric: Metric name (must exist in DECAY_SPANS and game logs)
        
        Returns:
            Dict mapping team -> DataFrame with [game_date, ewma_value]
        """
        df = self._load_and_prepare_game_logs()
        span = self.DECAY_SPANS.get(metric, 10)  # Default span=10
        
        ewma_by_team = {}
        
        for team in df['team'].unique():
            team_df = df[df['team'] == team].copy()
            
            # Add Bayesian priors
            metric_with_priors = self._add_bayesian_priors(team_df, metric)
            
            # Calculate EWMA
            ewma_values = metric_with_priors.ewm(span=span, adjust=False).mean()
            
            # Remove ghost games, keep only real games
            ewma_values = ewma_values.iloc[self.GHOST_GAMES:].reset_index(drop=True)
            
            # Align with original dates
            team_df['ewma_' + metric] = ewma_values.values
            
            ewma_by_team[team] = team_df[['game_date', 'ewma_' + metric]].copy()
        
        # Initialize cache dict if needed and store this metric's results
        if self._ewma_cache is None:
            self._ewma_cache = {}
        self._ewma_cache[metric] = ewma_by_team

        return ewma_by_team
    
    def get_ewma_stats(
        self,
        team: str,
        as_of_date: str
    ) -> Dict[str, float]:
        """
        Get EWMA stats for a team as of a specific date
        
        Args:
            team: Team abbreviation (e.g., 'LAL')
            as_of_date: Date to get stats before (YYYY-MM-DD)
        
        Returns:
            Dict with EWMA values for all metrics
        """
        # Load data if needed
        if self._game_logs_cache is None:
            self._load_and_prepare_game_logs()
        
        as_of = pd.to_datetime(as_of_date)
        
        # Get team's games before this date (use pre-grouped cache)
        if team not in self._team_logs_cache:
            # Return league priors if team not found
            return self.LEAGUE_PRIORS.copy()
        
        team_logs = self._team_logs_cache[team]
        team_logs = team_logs[team_logs['game_date'] < as_of].copy()
        
        if len(team_logs) == 0:
            # Return league priors if no data
            return self.LEAGUE_PRIORS.copy()
        
        # Calculate EWMA for each metric
        result = {}
        
        for metric, span in self.DECAY_SPANS.items():
            if metric not in team_logs.columns:
                result[f'ewma_{metric}'] = self.LEAGUE_PRIORS.get(metric, 0)
                continue
            # Use cached EWMA values when available to avoid recomputation
            if self._ewma_cache is None or metric not in self._ewma_cache:
                # Compute and cache for all teams for this metric
                self._calculate_ewma_by_team(metric)

            metric_cache = self._ewma_cache.get(metric, {})
            team_metric_df = metric_cache.get(team)
            if team_metric_df is None or len(team_metric_df) == 0:
                result[f'ewma_{metric}'] = self.LEAGUE_PRIORS.get(metric, 0)
                continue

            # Find last EWMA value before as_of using searchsorted (faster than boolean masking)
            # Ensure game_date is datetime64
            dates = team_metric_df['game_date'].values
            try:
                # numpy searchsorted expects comparable dtypes; convert as_of to numpy datetime64
                if not isinstance(as_of, (str,)):
                    as_of_np = np.datetime64(as_of)
                else:
                    as_of_np = np.datetime64(pd.to_datetime(as_of))
                # numpy searchsorted expects sorted values
                idx = dates.searchsorted(as_of_np)
            except Exception:
                # Fallback to slower method if searchsorted fails
                vals_before = team_metric_df[team_metric_df['game_date'] < as_of]
                if len(vals_before) == 0:
                    result[f'ewma_{metric}'] = self.LEAGUE_PRIORS.get(metric, 0)
                else:
                    result[f'ewma_{metric}'] = float(vals_before['ewma_' + metric].iloc[-1])
                continue

            if idx == 0:
                result[f'ewma_{metric}'] = self.LEAGUE_PRIORS.get(metric, 0)
            else:
                # idx points to first element >= as_of, so take idx-1
                val = team_metric_df['ewma_' + metric].iat[idx-1]
                result[f'ewma_{metric}'] = float(val)
        
        return result
    
    def calculate_ewma_features(
        self,
        home_team: str,
        away_team: str,
        game_date: str
    ) -> Dict[str, float]:
        """
        Calculate EWMA features for a matchup
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Date of game (YYYY-MM-DD)
        
        Returns:
            Dict with EWMA features for both teams
        """
        home_stats = self.get_ewma_stats(home_team, game_date)
        away_stats = self.get_ewma_stats(away_team, game_date)
        
        # Build feature dict
        features = {}
        
        # Individual team stats
        for metric in self.DECAY_SPANS.keys():
            features[f'home_ewma_{metric}'] = home_stats.get(f'ewma_{metric}', 0)
            features[f'away_ewma_{metric}'] = away_stats.get(f'ewma_{metric}', 0)
        
        # Differentials
        features['ewma_pace_diff'] = (
            home_stats.get('ewma_pace', 0) - away_stats.get('ewma_pace', 0)
        )
        features['ewma_efg_diff'] = (
            home_stats.get('ewma_efg', 0) - away_stats.get('ewma_efg', 0)
        )
        features['ewma_tov_diff'] = (
            home_stats.get('ewma_tov_pct', 0) - away_stats.get('ewma_tov_pct', 0)
        )
        
        # Interaction terms (using EWMA instead of simple averages)
        features['ewma_foul_synergy_home'] = (
            home_stats.get('ewma_fta_rate', 0) * away_stats.get('ewma_foul_rate', 0)
        ) / 100
        
        features['ewma_foul_synergy_away'] = (
            away_stats.get('ewma_fta_rate', 0) * home_stats.get('ewma_foul_rate', 0)
        ) / 100
        
        features['ewma_chaos_home'] = (
            home_stats.get('ewma_stl_pct', 0) * away_stats.get('ewma_tov_pct', 0)
        )
        
        features['ewma_chaos_away'] = (
            away_stats.get('ewma_stl_pct', 0) * home_stats.get('ewma_tov_pct', 0)
        )
        
        features['ewma_net_chaos'] = (
            features['ewma_chaos_home'] - features['ewma_chaos_away']
        )
        
        features['ewma_vol_3p_diff'] = (
            home_stats.get('ewma_3pa_per_100', 0) - away_stats.get('ewma_3pa_per_100', 0)
        )
        
        return features
    
    def get_feature_names(self) -> list[str]:
        """Return list of all EWMA feature names"""
        features = []
        
        # Individual team stats
        for metric in self.DECAY_SPANS.keys():
            features.append(f'home_ewma_{metric}')
            features.append(f'away_ewma_{metric}')
        
        # Differentials
        features.extend([
            'ewma_pace_diff',
            'ewma_efg_diff',
            'ewma_tov_diff',
        ])
        
        # Interactions
        features.extend([
            'ewma_foul_synergy_home',
            'ewma_foul_synergy_away',
            'ewma_chaos_home',
            'ewma_chaos_away',
            'ewma_net_chaos',
            'ewma_vol_3p_diff',
        ])
        
        return features


def example_usage():
    """Demonstration of EWMA feature calculator"""
    calc = EWMAFeatureCalculator(db_path='../../nba_betting_data.db')
    
    # Example: Lakers vs Warriors on 2025-01-15
    features = calc.calculate_ewma_features(
        home_team='LAL',
        away_team='GSW',
        game_date='2025-01-15'
    )
    
    print("EWMA Features for LAL vs GSW:")
    print("=" * 70)
    
    print("\nHome Team (LAL) EWMA Stats:")
    for key, value in features.items():
        if key.startswith('home_ewma_'):
            metric = key.replace('home_ewma_', '')
            span = EWMAFeatureCalculator.DECAY_SPANS.get(metric, 10)
            print(f"  {metric:20s}: {value:8.3f} (Span={span})")
    
    print("\nAway Team (GSW) EWMA Stats:")
    for key, value in features.items():
        if key.startswith('away_ewma_'):
            metric = key.replace('away_ewma_', '')
            span = EWMAFeatureCalculator.DECAY_SPANS.get(metric, 10)
            print(f"  {metric:20s}: {value:8.3f} (Span={span})")
    
    print("\nInteraction Features:")
    for key in ['ewma_foul_synergy_home', 'ewma_chaos_home', 'ewma_net_chaos', 'ewma_vol_3p_diff']:
        print(f"  {key:30s}: {features[key]:8.3f}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    
    # Pace trend
    pace_diff = features['ewma_pace_diff']
    if pace_diff > 2:
        print(f"✓ Home team plays {pace_diff:.1f} possessions FASTER → Higher total expected")
    elif pace_diff < -2:
        print(f"✓ Away team plays {abs(pace_diff):.1f} possessions FASTER → Higher total expected")
    
    # Defense trend
    def_diff = features['ewma_def_rating_diff']
    if def_diff > 3:
        print(f"✓ Home team defense WORSE by {def_diff:.1f} points → Favors away")
    elif def_diff < -3:
        print(f"✓ Home team defense BETTER by {abs(def_diff):.1f} points → Favors home")
    
    # Chaos
    if features['ewma_net_chaos'] > 0.5:
        print("✓ Home team has transition advantage (steals × opp turnovers)")
    elif features['ewma_net_chaos'] < -0.5:
        print("✓ Away team has transition advantage (steals × opp turnovers)")


if __name__ == "__main__":
    example_usage()
