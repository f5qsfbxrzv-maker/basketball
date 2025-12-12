"""
LINEUP CONTINUITY TRACKER - THE HIDDEN ALPHA
Uses play-by-play data to track actual lineup combinations and chemistry

CRITICAL INSIGHT:
Box scores show who played, NOT who played TOGETHER.
- Lineup A: Steph + Klay + Draymond (30 min, +15 net rating)
- Lineup B: Steph + Klay + Bench (18 min, -5 net rating)
Box scores show 48 min for Steph, but miss the CHEMISTRY difference.

METHODOLOGY:
1. pbpstats extracts possession-level lineup data from NBA API
2. Track games played together for starting 5
3. New lineups (<5 games) get -1.0 point penalty
4. Established lineups (>20 games) get +0.5 boost
5. Garbage time automatically filtered

DATA SOURCE:
- pbpstats library (wraps NBA stats API)
- Handles substitution tracking and ordering
- Provides clean lineup IDs per possession

EXPECTED IMPACT: -0.5 RMSE (lineup chemistry matters for sets/rotations)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional pbpstats import (graceful degradation)
try:
    from pbpstats.client import Client
    PBPSTATS_AVAILABLE = True
except ImportError:
    PBPSTATS_AVAILABLE = False
    logger.warning("pbpstats not installed - lineup tracking disabled")


class LineupTracker:
    """
    Track lineup continuity and chemistry from play-by-play data
    """
    
    def __init__(self, data_dir: str = "./nba_pbp_data"):
        """
        Initialize lineup tracker with pbpstats client
        
        Args:
            data_dir: Directory to cache play-by-play data
        """
        self.data_dir = data_dir
        
        if not PBPSTATS_AVAILABLE:
            logger.warning("LineupTracker initialized but pbpstats unavailable")
            self.client = None
            return
        
        # Initialize pbpstats client
        try:
            settings = {
                "dir": data_dir,
                "Boxscore": {"source": "file", "data_provider": "stats_nba"},
                "Possessions": {"source": "file", "data_provider": "stats_nba"},
            }
            self.client = Client(settings)
            logger.info(f"LineupTracker initialized with data_dir={data_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize pbpstats client: {e}")
            self.client = None
        
        # Cache of lineup continuity by team
        # Format: {team_abbrev: {frozenset(player_ids): games_together}}
        self.lineup_history: Dict[str, Dict[frozenset, int]] = defaultdict(lambda: defaultdict(int))
    
    def extract_lineups_from_game(self, game_id: str) -> Dict[str, List[Set[str]]]:
        """
        Extract all lineup combinations from a single game
        
        Args:
            game_id: NBA game ID (e.g., '0022300001')
        
        Returns:
            Dict mapping team -> list of lineup sets (excluding garbage time)
        """
        if not self.client:
            logger.warning("pbpstats client not available")
            return {}
        
        try:
            game = self.client.Game(game_id)
            
            lineups_by_team = defaultdict(list)
            
            for possession in game.possessions.items:
                # Skip garbage time (pbpstats has built-in filter)
                if possession.is_garbage_time:
                    continue
                
                # Extract offense and defense lineups
                offense_team = possession.offense_team_id
                defense_team = possession.defense_team_id
                
                offense_lineup = set(possession.offense_lineup_ids)
                defense_lineup = set(possession.defense_lineup_ids)
                
                lineups_by_team[offense_team].append(offense_lineup)
                lineups_by_team[defense_team].append(defense_lineup)
            
            logger.debug(f"Extracted lineups from game {game_id}")
            return dict(lineups_by_team)
        
        except Exception as e:
            logger.error(f"Failed to extract lineups from game {game_id}: {e}")
            return {}
    
    def update_lineup_history(self, game_id: str, team_abbrev: str):
        """
        Update lineup history after a game
        
        Args:
            game_id: NBA game ID
            team_abbrev: Team abbreviation (e.g., 'GSW')
        """
        lineups = self.extract_lineups_from_game(game_id)
        
        if not lineups:
            return
        
        # Get lineups for this team
        team_lineups = lineups.get(team_abbrev, [])
        
        # Track unique lineups used
        unique_lineups = set()
        for lineup in team_lineups:
            if len(lineup) == 5:  # Only track full 5-man lineups
                unique_lineups.add(frozenset(lineup))
        
        # Increment games together for each lineup
        for lineup in unique_lineups:
            self.lineup_history[team_abbrev][lineup] += 1
        
        logger.debug(f"Updated lineup history for {team_abbrev}: {len(unique_lineups)} lineups")
    
    def get_starting_lineup_continuity(
        self,
        team_abbrev: str,
        starting_five: List[str],
    ) -> int:
        """
        Get number of games this starting 5 has played together
        
        Args:
            team_abbrev: Team abbreviation
            starting_five: List of 5 player IDs
        
        Returns:
            Number of games this lineup has played together
        """
        if len(starting_five) != 5:
            logger.warning(f"Invalid starting five: {len(starting_five)} players")
            return 0
        
        lineup_set = frozenset(starting_five)
        games_together = self.lineup_history[team_abbrev].get(lineup_set, 0)
        
        return games_together
    
    def calculate_continuity_penalty(self, games_together: int) -> float:
        """
        Calculate lineup continuity penalty/bonus
        
        Args:
            games_together: Number of games lineup has played together
        
        Returns:
            Penalty/bonus in points (negative = penalty, positive = bonus)
        """
        if games_together < 5:
            # New lineup - chemistry issues (sets, rotations, switches)
            return -1.0
        elif games_together > 20:
            # Established lineup - chemistry bonus
            return 0.5
        else:
            # Middle ground - neutral
            return 0.0
    
    def get_lineup_features(
        self,
        team_abbrev: str,
        starting_five: List[str],
    ) -> Dict[str, float]:
        """
        Get lineup continuity features for ML model
        
        Args:
            team_abbrev: Team abbreviation
            starting_five: List of 5 player IDs
        
        Returns:
            Dict with games_together and continuity_penalty features
        """
        games_together = self.get_starting_lineup_continuity(team_abbrev, starting_five)
        penalty = self.calculate_continuity_penalty(games_together)
        
        return {
            'games_together': float(games_together),
            'continuity_penalty': penalty,
            'is_new_lineup': 1.0 if games_together < 5 else 0.0,
            'is_established': 1.0 if games_together > 20 else 0.0,
        }


# --- INSTALLATION INSTRUCTIONS ---
"""
To enable lineup tracking:

1. Install pbpstats:
   pip install pbpstats

2. Update requirements.txt:
   Add: pbpstats>=1.0.0

3. First-time setup (downloads data):
   from v2.features.lineup_tracker import LineupTracker
   tracker = LineupTracker(data_dir="./nba_pbp_data")
   tracker.extract_lineups_from_game("0022300001")  # Sample game

4. Integration with feature_calculator_v5.py:
   - Add lineup_tracker as class attribute
   - Call get_lineup_features() during feature extraction
   - Requires starting lineup data (from rotowire or nba.com)
"""


if __name__ == "__main__":
    # Example usage
    if PBPSTATS_AVAILABLE:
        tracker = LineupTracker(data_dir="./nba_pbp_data")
        
        # Example: Track Warriors lineup continuity
        warriors_starting_five = [
            "201939",  # Steph Curry
            "202691",  # Klay Thompson
            "203110",  # Draymond Green
            "1629638", # Andrew Wiggins
            "1628369", # Kevon Looney
        ]
        
        features = tracker.get_lineup_features("GSW", warriors_starting_five)
        print(f"\nWarriors Starting 5 Continuity:")
        print(f"  Games Together: {features['games_together']:.0f}")
        print(f"  Continuity Penalty: {features['continuity_penalty']:+.1f} pts")
        print(f"  New Lineup: {features['is_new_lineup'] == 1.0}")
    else:
        print("\nLineup tracking requires 'pbpstats' library:")
        print("  pip install pbpstats")
