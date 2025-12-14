"""
Simplified injury impact calculator that works with live database structure
Uses position from active_injuries table and PIE from player_stats
"""
import sqlite3
from typing import Dict

# Team abbreviation to full name mapping
TEAM_ABBREV_TO_FULL = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

# Position-specific replacement PIE baselines
POSITION_REPLACEMENT_PIE = {
    'PG': 0.095,   # Point Guard - hardest to replace
    'SG': 0.088,   # Shooting Guard
    'SF': 0.092,   # Small Forward
    'PF': 0.090,   # Power Forward
    'C': 0.093,    # Center - second hardest
    'G': 0.091, 'F': 0.091, 'F-C': 0.091, 'G-F': 0.090
}

# Positional scarcity multipliers (how hard to replace)
POSITION_SCARCITY = {
    'PG': 1.15,  # Elite PGs hard to replace
    'C': 1.12,   # Two-way centers scarce
    'PF': 1.05,
    'SG': 1.0, 'SF': 1.0, 'G': 1.0, 'F': 1.0, 'F-C': 1.08, 'G-F': 1.0
}

# Status-based play probabilities
STATUS_PLAY_PROBABILITIES = {
    'Out': 0.0,
    'Doubtful': 0.25,
    'Questionable': 0.5,
    'Probable': 0.75,
    'GTD': 0.5,  # Game-Time Decision
    'Day-To-Day': 0.75
}


def calculate_team_injury_impact_simple(team_full_name: str, game_date: str, db_path: str) -> float:
    """
    Calculate total PIE-weighted injury impact for a team using live database
    
    Args:
        team_full_name: Team name (full like "Orlando Magic" OR abbreviation like "ORL")
        game_date: Date string (not currently used, for API compatibility)
        db_path: Path to SQLite database
    
    Returns:
        Total impact score (0-15 typically, higher = more injured)
    """
    # Convert abbreviation to full name if needed
    if len(team_full_name) == 3 and team_full_name.upper() in TEAM_ABBREV_TO_FULL:
        team_full_name = TEAM_ABBREV_TO_FULL[team_full_name.upper()]
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get injured players with position from active_injuries table
    cursor.execute("""
        SELECT player_name, status, position
        FROM active_injuries
        WHERE team_name = ?
        AND status IN ('Out', 'Questionable', 'Doubtful', 'Probable', 'GTD', 'Day-To-Day')
    """, (team_full_name,))
    
    injuries = cursor.fetchall()
    total_impact = 0.0
    
    for player_name, status, position in injuries:
        # Get player PIE from player_stats
        cursor.execute("""
            SELECT pie
            FROM player_stats
            WHERE player_name = ?
            ORDER BY season DESC
            LIMIT 1
        """, (player_name,))
        
        pie_row = cursor.fetchone()
        
        if not pie_row or pie_row[0] is None:
            # No PIE data - use position-based replacement level only
            player_pie = POSITION_REPLACEMENT_PIE.get(position, 0.090) * 1.2  # Assume slightly above replacement
        else:
            player_pie = float(pie_row[0])
        
        # Get position info (if N/A, default to Guard position)
        if position == 'N/A' or not position:
            position = 'G'  # Default to guard (middle-of-road replacement value)
        
        replacement_pie = POSITION_REPLACEMENT_PIE.get(position, 0.090)
        scarcity = POSITION_SCARCITY.get(position, 1.0)
        
        # Get play probability
        play_prob = STATUS_PLAY_PROBABILITIES.get(status, 0.5)
        absence_prob = 1.0 - play_prob
        
        # Calculate impact: (player PIE - replacement PIE) * scarcity * absence probability
        raw_impact = (player_pie - replacement_pie) * scarcity * absence_prob
        
        # Scale to net rating points (multiply by 100 for interpretability)
        impact_points = max(0.0, raw_impact) * 100.0
        
        total_impact += impact_points
        
        print(f"  [{team_full_name[:3]}] {player_name:20s} ({position:3s}) - PIE: {player_pie:.3f}, Impact: {impact_points:5.2f} pts ({status})")
    
    conn.close()
    return total_impact


if __name__ == '__main__':
    from datetime import datetime
    
    # Test with today's teams
    today = datetime.now().strftime('%Y-%m-%d')
    db_path = 'data/live/nba_betting_data.db'
    
    test_teams = [
        'New York Knicks',
        'Orlando Magic',
        'San Antonio Spurs',
        'Oklahoma City Thunder'
    ]
    
    print("\n=== TESTING PIE-BASED INJURY CALCULATIONS (SIMPLIFIED) ===\n")
    print(f"Date: {today}\n")
    
    for team in test_teams:
        try:
            print(f"\n{team}:")
            impact = calculate_team_injury_impact_simple(team, today, db_path)
            print(f"  TOTAL IMPACT: {impact:.2f} pts\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
            import traceback
            traceback.print_exc()
