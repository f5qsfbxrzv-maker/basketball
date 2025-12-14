"""
Add Advanced Features to Break 0.56 AUC:
1. Schematic Matchups (rim pressure, 3PT variance, rebounding)
2. PIE-based Active Roster Share (replace binary injury flags)
3. Travel/Logistics (distance, time zones, rhythm)

Requires NBA API data for team stats and player PIE scores.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguedashteamstats, teamplayerdashboard
from nba_api.stats.static import teams
import time
from datetime import datetime, timedelta

print("="*70)
print("ADDING ADVANCED FEATURES - BREAKING THE 0.56 BARRIER")
print("="*70)

# Load lean dataset
print("\n1. Loading lean dataset...")
df = pd.read_csv("data/training_data_lean.csv")
print(f"   Games: {len(df):,}")
print(f"   Current features: {len([c for c in df.columns if c not in ['date','game_id','home_team','away_team','season','target_spread','target_spread_cover','target_moneyline_win','target_game_total','target_over_under','target_home_cover','target_over']])}")

# Arena altitudes and locations for travel calculation
arena_info = {
    'ATL': {'lat': 33.7573, 'lon': -84.3963, 'altitude': 1050},
    'BOS': {'lat': 42.3662, 'lon': -71.0621, 'altitude': 20},
    'BKN': {'lat': 40.6826, 'lon': -73.9754, 'altitude': 50},
    'CHA': {'lat': 35.2251, 'lon': -80.8392, 'altitude': 750},
    'CHI': {'lat': 41.8807, 'lon': -87.6742, 'altitude': 594},
    'CLE': {'lat': 41.4965, 'lon': -81.6882, 'altitude': 653},
    'DAL': {'lat': 32.7905, 'lon': -96.8103, 'altitude': 430},
    'DEN': {'lat': 39.7487, 'lon': -104.8769, 'altitude': 5280},
    'DET': {'lat': 42.3410, 'lon': -83.0550, 'altitude': 600},
    'GSW': {'lat': 37.7680, 'lon': -122.3877, 'altitude': 10},
    'HOU': {'lat': 29.7508, 'lon': -95.3621, 'altitude': 50},
    'IND': {'lat': 39.7640, 'lon': -86.1555, 'altitude': 715},
    'LAC': {'lat': 34.0430, 'lon': -118.2673, 'altitude': 305},
    'LAL': {'lat': 34.0430, 'lon': -118.2673, 'altitude': 305},
    'MEM': {'lat': 35.1382, 'lon': -90.0505, 'altitude': 337},
    'MIA': {'lat': 25.7814, 'lon': -80.1870, 'altitude': 10},
    'MIL': {'lat': 43.0436, 'lon': -87.9167, 'altitude': 617},
    'MIN': {'lat': 44.9795, 'lon': -93.2760, 'altitude': 830},
    'NOP': {'lat': 29.9490, 'lon': -90.0821, 'altitude': 10},
    'NYK': {'lat': 40.7505, 'lon': -73.9934, 'altitude': 35},
    'OKC': {'lat': 35.4634, 'lon': -97.5151, 'altitude': 1200},
    'ORL': {'lat': 28.5392, 'lon': -81.3839, 'altitude': 82},
    'PHI': {'lat': 39.9012, 'lon': -75.1720, 'altitude': 39},
    'PHX': {'lat': 33.4457, 'lon': -112.0712, 'altitude': 1086},
    'POR': {'lat': 45.5316, 'lon': -122.6668, 'altitude': 50},
    'SAC': {'lat': 38.5802, 'lon': -121.4997, 'altitude': 30},
    'SAS': {'lat': 29.4270, 'lon': -98.4375, 'altitude': 650},
    'TOR': {'lat': 43.6435, 'lon': -79.3791, 'altitude': 250},
    'UTA': {'lat': 40.7683, 'lon': -111.9011, 'altitude': 4226},
    'WAS': {'lat': 38.8981, 'lon': -77.0209, 'altitude': 10}
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in miles between two points"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 3959  # Earth radius in miles
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def get_timezone_offset(team):
    """Get timezone offset from EST"""
    pacific = ['LAL', 'LAC', 'GSW', 'POR', 'SAC', 'PHX']
    mountain = ['DEN', 'UTA', 'MIN', 'OKC', 'SAS', 'DAL']
    
    if team in pacific:
        return -3
    elif team in mountain:
        return -2
    else:
        return 0

print("\n2. Creating travel/logistics features...")

# Sort by date for rolling calculations
df = df.sort_values(['away_team', 'date']).reset_index(drop=True)

# Initialize new features
df['rolling_7_day_distance'] = 0.0
df['body_clock_lag'] = 0
df['first_game_home_after_road_trip'] = 0

# Calculate for each game
for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"   Processing game {idx:,}/{len(df):,}...")
    
    away_team = row['away_team']
    home_team = row['home_team']
    game_date = pd.to_datetime(row['date'])
    
    # Get last 7 days of away team's games
    past_games = df[
        (df['away_team'] == away_team) | (df['home_team'] == away_team)
    ]
    past_games = past_games[
        (pd.to_datetime(past_games['date']) < game_date) &
        (pd.to_datetime(past_games['date']) >= game_date - timedelta(days=7))
    ]
    
    # Calculate total distance traveled
    total_distance = 0
    if len(past_games) > 0:
        prev_location = None
        for _, past_game in past_games.iterrows():
            # Determine where team played
            if past_game['away_team'] == away_team:
                location = past_game['home_team']
            else:
                location = past_game['away_team'] if past_game['away_team'] != away_team else past_game['home_team']
            
            if prev_location and location in arena_info and prev_location in arena_info:
                dist = haversine_distance(
                    arena_info[prev_location]['lat'], arena_info[prev_location]['lon'],
                    arena_info[location]['lat'], arena_info[location]['lon']
                )
                total_distance += dist
            
            prev_location = location
        
        # Add distance to current game
        if prev_location and home_team in arena_info and prev_location in arena_info:
            total_distance += haversine_distance(
                arena_info[prev_location]['lat'], arena_info[prev_location]['lon'],
                arena_info[home_team]['lat'], arena_info[home_team]['lon']
            )
    
    df.at[idx, 'rolling_7_day_distance'] = total_distance
    
    # Time zone lag (EST baseline)
    df.at[idx, 'body_clock_lag'] = abs(get_timezone_offset(away_team) - get_timezone_offset(home_team))
    
    # First game home after road trip (5+ games)
    if len(past_games) >= 5:
        all_away = all(past_games['away_team'] == away_team)
        if all_away and row['home_team'] == away_team:
            df.at[idx, 'first_game_home_after_road_trip'] = 1

print(f"   ✓ rolling_7_day_distance: mean={df['rolling_7_day_distance'].mean():.0f} miles")
print(f"   ✓ body_clock_lag: {(df['body_clock_lag'] > 0).sum():,} games with time zone shift")
print(f"   ✓ first_game_home_after_road_trip: {df['first_game_home_after_road_trip'].sum():,} games")

print("\n3. Creating schematic matchup features (using season averages)...")
print("   NOTE: This requires NBA API - using simplified version for now")

# Placeholder - in production, fetch from NBA API
# For now, create synthetic features based on existing data
df['rim_pressure_mismatch'] = np.random.randn(len(df)) * 0.1  # TODO: Replace with real data
df['three_point_variance'] = np.random.randn(len(df)) * 0.1    # TODO: Replace with real data
df['rebounding_mismatch'] = np.random.randn(len(df)) * 0.1     # TODO: Replace with real data

print("   ⚠ Using placeholder data - requires NBA API integration")
print("   TODO: Fetch paint FG%, 3PA rate, OREB%/DREB% from NBA stats")

print("\n4. Creating PIE-based active roster share...")
print("   NOTE: This requires player injury data and PIE scores")

# Placeholder - in production, calculate from injury reports + player PIE
df['home_active_share'] = 1.0 - (np.random.rand(len(df)) * 0.2)  # 0.8-1.0
df['away_active_share'] = 1.0 - (np.random.rand(len(df)) * 0.2)  # 0.8-1.0

print("   ⚠ Using placeholder data - requires injury reports + player PIE")
print("   TODO: Calculate sum(active_PIE) / sum(full_roster_PIE)")

# Count new features
new_features = [
    'rolling_7_day_distance',
    'body_clock_lag', 
    'first_game_home_after_road_trip',
    'rim_pressure_mismatch',
    'three_point_variance',
    'rebounding_mismatch',
    'home_active_share',
    'away_active_share'
]

print(f"\n5. Added {len(new_features)} advanced features:")
for i, feat in enumerate(new_features, 1):
    print(f"   {i}. {feat}")

# Save enhanced dataset
output_path = "data/training_data_enhanced.csv"
df.to_csv(output_path, index=False)

total_features = len([c for c in df.columns if c not in ['date','game_id','home_team','away_team','season','target_spread','target_spread_cover','target_moneyline_win','target_game_total','target_over_under','target_home_cover','target_over']])

print(f"\n6. Saved: {output_path}")
print(f"   Total features: {total_features} (21 lean + {len(new_features)} advanced)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("IMPLEMENTED:")
print("  ✓ Travel distance (rolling 7 days)")
print("  ✓ Time zone lag")
print("  ✓ First game home after road trip")
print("\nTODO (Requires Data Integration):")
print("  ⚠ Schematic matchups (rim pressure, 3PT variance, rebounding)")
print("  ⚠ PIE-based active roster share")
print("  ⚠ Market signals (closing line moves)")
print("  ⚠ Referee variance")
print("\nNEXT STEPS:")
print("  1. Integrate NBA API for team matchup stats")
print("  2. Add injury report scraper for PIE calculations")
print("  3. Train model: python scripts/train_enhanced_model.py")
print("="*70)
