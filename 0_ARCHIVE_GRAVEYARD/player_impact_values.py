"""
Player Impact Values for Injury Penalty Calculation
Based on typical spread value per player
"""

# Default spread values for common players (points impact when out)
# Format: 'Player Name': impact_points
PLAYER_SPREAD_VALUES = {
    # Top tier stars (5+ point impact)
    'Nikola Jokic': 7.5,
    'Luka Doncic': 7.0,
    'Giannis Antetokounmpo': 7.0,
    'Joel Embiid': 6.5,
    'Shai Gilgeous-Alexander': 6.0,
    'Stephen Curry': 6.0,
    'Kevin Durant': 5.5,
    'LeBron James': 5.5,
    'Anthony Davis': 5.0,
    'Jayson Tatum': 5.0,
    'Damian Lillard': 5.0,
    
    # Elite players (3-5 point impact)
    'Anthony Edwards': 4.5,
    'Ja Morant': 4.5,
    'Donovan Mitchell': 4.0,
    'Devin Booker': 4.0,
    'Tyrese Haliburton': 4.0,
    'Kawhi Leonard': 4.5,
    'Jimmy Butler': 4.0,
    'Jaylen Brown': 3.5,
    'Trae Young': 4.0,
    'De Aaron Fox': 3.5,
    'Paolo Banchero': 3.5,
    'Lauri Markkanen': 3.0,
    
    # Quality starters (2-3 point impact)
    'Jalen Brunson': 3.5,
    'Julius Randle': 2.5,
    'Cade Cunningham': 3.0,
    'Franz Wagner': 2.5,
    'Scottie Barnes': 2.5,
    'Evan Mobley': 2.5,
    'Darius Garland': 2.5,
    'LaMelo Ball': 3.5,
    'Zion Williamson': 4.0,
    'Brandon Ingram': 2.5,
    'CJ McCollum': 2.0,
    'Karl-Anthony Towns': 3.0,
    'Rudy Gobert': 2.0,
}

def calculate_injury_penalty(player_name: str, play_probability: float) -> float:
    """
    Calculate the spread penalty for an injured player
    
    Formula: penalty = base_value * (1 - play_probability)
    
    Args:
        player_name: Player's full name
        play_probability: 0.0 (out) to 1.0 (playing)
        
    Returns:
        Points to adjust spread (positive = penalty for team)
    """
    base_value = PLAYER_SPREAD_VALUES.get(player_name, 0.5)  # Default bench player
    penalty = base_value * (1 - play_probability)
    return penalty

if __name__ == "__main__":
    # Test examples
    print("Injury Penalty Calculator Examples:")
    print("=" * 50)
    
    test_cases = [
        ('Nikola Jokic', 0.0, 'Out'),
        ('Nikola Jokic', 0.5, 'Questionable'),
        ('LeBron James', 0.0, 'Out'),
        ('LeBron James', 0.75, 'Probable'),
        ('Unknown Player', 0.0, 'Out'),
    ]
    
    for player, prob, status in test_cases:
        penalty = calculate_injury_penalty(player, prob)
        print(f"{player} ({status}): -{penalty:.1f} points")
