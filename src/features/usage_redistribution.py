"""
Usage Rate Redistribution Model
When star players are OUT, their usage doesn't disappear - it redistributes
Sharp insight: Don't under-project totals when stars sit
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ===================================================================
# USAGE RATE REDISTRIBUTION RULES
# ===================================================================
# When a player with X% usage is OUT, where does it go?
#
# Research shows:
# 1. Primary ball-handler OUT → Secondary ball-handler gets ~40% of usage
# 2. Wing scorer OUT → Usage spreads more evenly (30% to backup, 70% distributed)
# 3. Big man OUT → Minimal usage change (mostly defensive impact)
#
# Example: Luka Doncic (35% usage) is OUT
#   → Kyrie Irving gets +14% usage (40% of 35%)
#   → Other starters get +3-4% each (60% of 35% / 4 players)
# ===================================================================

POSITION_REDISTRIBUTION_RULES = {
    'PG': {  # Point guard out
        'primary_ball_handler': 0.40,  # 40% to secondary PG/combo guard
        'wings': 0.35,                  # 35% to wings (split)
        'bigs': 0.25,                   # 25% to bigs (split)
    },
    'SG': {  # Shooting guard out
        'primary_ball_handler': 0.30,
        'wings': 0.45,
        'bigs': 0.25,
    },
    'SF': {  # Small forward out
        'primary_ball_handler': 0.25,
        'wings': 0.50,
        'bigs': 0.25,
    },
    'PF': {  # Power forward out
        'primary_ball_handler': 0.20,
        'wings': 0.30,
        'bigs': 0.50,
    },
    'C': {  # Center out
        'primary_ball_handler': 0.15,
        'wings': 0.25,
        'bigs': 0.60,
    },
}


@dataclass
class PlayerUsageProfile:
    """Player usage and efficiency profile"""
    name: str
    position: str
    usage_rate: float           # Percentage of team possessions used
    true_shooting: float        # True shooting percentage
    points_per_game: float
    assists_per_game: float
    is_primary_ball_handler: bool
    

@dataclass
class UsageRedistribution:
    """Result of usage redistribution calculation"""
    injured_player: str
    usage_removed: float
    redistributions: Dict[str, float]  # Player name → additional usage
    total_points_adjustment: float     # Net expected points change
    

class UsageRedistributionModel:
    """
    Model usage rate redistribution when key players are injured/resting
    """
    
    def __init__(self):
        self.team_rosters: Dict[str, List[PlayerUsageProfile]] = {}
    
    def load_team_roster(
        self, 
        team_name: str,
        roster_data: pd.DataFrame
    ) -> None:
        """
        Load team roster with usage profiles
        
        Expected DataFrame columns:
        - PLAYER_NAME
        - POSITION
        - USG_PCT (usage rate)
        - TS_PCT (true shooting %)
        - PTS (points per game)
        - AST (assists per game)
        """
        profiles = []
        
        for _, row in roster_data.iterrows():
            # Identify primary ball handler
            # Heuristic: High usage + high assists
            is_primary = (
                row['USG_PCT'] > 25.0 and 
                row['AST'] > 5.0
            )
            
            profile = PlayerUsageProfile(
                name=row['PLAYER_NAME'],
                position=row['POSITION'],
                usage_rate=row['USG_PCT'],
                true_shooting=row['TS_PCT'],
                points_per_game=row['PTS'],
                assists_per_game=row['AST'],
                is_primary_ball_handler=is_primary,
            )
            profiles.append(profile)
        
        self.team_rosters[team_name] = profiles
        logger.info("roster_loaded", team=team_name, players=len(profiles))
    
    def calculate_redistribution(
        self,
        team_name: str,
        injured_player: str,
        probability_playing: float = 0.0,
    ) -> Optional[UsageRedistribution]:
        """
        Calculate how usage redistributes when a player is out/questionable
        
        Args:
            team_name: Team name
            injured_player: Player name who is injured
            probability_playing: 0.0 = definitely out, 0.5 = questionable, 1.0 = playing
            
        Returns:
            UsageRedistribution with expected point adjustments
        """
        if team_name not in self.team_rosters:
            logger.warning("team_roster_not_loaded", team=team_name)
            return None
        
        roster = self.team_rosters[team_name]
        
        # Find injured player
        injured_profile = None
        for player in roster:
            if player.name == injured_player:
                injured_profile = player
                break
        
        if not injured_profile:
            logger.warning("player_not_found", player=injured_player)
            return None
        
        # Calculate expected usage removal
        # If 50% chance to play, only remove 50% of their usage
        usage_removed = injured_profile.usage_rate * (1.0 - probability_playing)
        
        if usage_removed < 1.0:  # Negligible impact
            return None
        
        # Get redistribution rules for position
        rules = POSITION_REDISTRIBUTION_RULES.get(
            injured_profile.position, 
            POSITION_REDISTRIBUTION_RULES['SF']  # Default to SF rules
        )
        
        # Categorize remaining players
        primary_handlers = [p for p in roster if p.is_primary_ball_handler and p.name != injured_player]
        wings = [p for p in roster if p.position in ['SG', 'SF'] and p.name != injured_player]
        bigs = [p for p in roster if p.position in ['PF', 'C'] and p.name != injured_player]
        
        redistributions = {}
        total_points_adjustment = 0.0
        
        # Redistribute to primary ball handlers
        if primary_handlers:
            usage_to_primary = usage_removed * rules['primary_ball_handler']
            per_player = usage_to_primary / len(primary_handlers)
            
            for player in primary_handlers:
                redistributions[player.name] = per_player
                # Estimate points: New usage × player efficiency
                # Simplified: Additional possessions × TS% × 2 (average shot value)
                new_points = per_player * player.true_shooting * 2.0
                total_points_adjustment += new_points
        
        # Redistribute to wings
        if wings:
            usage_to_wings = usage_removed * rules['wings']
            per_player = usage_to_wings / len(wings)
            
            for player in wings:
                redistributions[player.name] = redistributions.get(player.name, 0) + per_player
                new_points = per_player * player.true_shooting * 1.8  # Wings slightly less efficient
                total_points_adjustment += new_points
        
        # Redistribute to bigs
        if bigs:
            usage_to_bigs = usage_removed * rules['bigs']
            per_player = usage_to_bigs / len(bigs)
            
            for player in bigs:
                redistributions[player.name] = redistributions.get(player.name, 0) + per_player
                new_points = per_player * player.true_shooting * 2.2  # Bigs more efficient
                total_points_adjustment += new_points
        
        # Net adjustment: Replacement players less efficient than star
        # Subtract injured player's expected points
        injured_points = injured_profile.points_per_game * (1.0 - probability_playing)
        net_adjustment = total_points_adjustment - injured_points
        
        logger.info("usage_redistributed",
                   injured_player=injured_player,
                   usage_removed=usage_removed,
                   injured_ppg=injured_points,
                   replacement_ppg=total_points_adjustment,
                   net_adjustment=net_adjustment)
        
        return UsageRedistribution(
            injured_player=injured_player,
            usage_removed=usage_removed,
            redistributions=redistributions,
            total_points_adjustment=net_adjustment,
        )
    
    def get_team_total_adjustment(
        self,
        team_name: str,
        injured_players: Dict[str, float],  # player_name → probability_playing
    ) -> float:
        """
        Calculate total expected points adjustment for multiple injuries
        
        Args:
            team_name: Team name
            injured_players: Dictionary of player name → play probability
            
        Returns:
            Net expected points adjustment (negative = fewer points)
        """
        total_adjustment = 0.0
        
        for player_name, prob_playing in injured_players.items():
            redistribution = self.calculate_redistribution(
                team_name, 
                player_name,
                prob_playing
            )
            
            if redistribution:
                total_adjustment += redistribution.total_points_adjustment
        
        return total_adjustment


if __name__ == "__main__":
    # Example usage
    model = UsageRedistributionModel()
    
    # Simulate Dallas roster
    dallas_roster = pd.DataFrame([
        {
            'PLAYER_NAME': 'Luka Doncic',
            'POSITION': 'PG',
            'USG_PCT': 35.0,
            'TS_PCT': 0.580,
            'PTS': 28.5,
            'AST': 8.5,
        },
        {
            'PLAYER_NAME': 'Kyrie Irving',
            'POSITION': 'PG',
            'USG_PCT': 28.0,
            'TS_PCT': 0.590,
            'PTS': 25.0,
            'AST': 5.0,
        },
        {
            'PLAYER_NAME': 'Daniel Gafford',
            'POSITION': 'C',
            'USG_PCT': 15.0,
            'TS_PCT': 0.680,
            'PTS': 11.0,
            'AST': 1.0,
        },
    ])
    
    model.load_team_roster("Dallas Mavericks", dallas_roster)
    
    # Luka is OUT (0% chance to play)
    result = model.calculate_redistribution("Dallas Mavericks", "Luka Doncic", 0.0)
    
    print(f"Luka OUT - Usage Redistribution:")
    print(f"  Usage removed: {result.usage_removed:.1f}%")
    print(f"  Redistributions:")
    for player, usage in result.redistributions.items():
        print(f"    {player}: +{usage:.1f}% usage")
    print(f"  Net points adjustment: {result.total_points_adjustment:+.1f}")
    
    # Luka is Questionable (50% chance to play)
    result_q = model.calculate_redistribution("Dallas Mavericks", "Luka Doncic", 0.5)
    print(f"\nLuka QUESTIONABLE - Usage Redistribution:")
    print(f"  Net points adjustment: {result_q.total_points_adjustment:+.1f}")
