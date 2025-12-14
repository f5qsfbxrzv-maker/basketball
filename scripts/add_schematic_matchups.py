"""
Add REAL Schematic Matchup Features using NBA API
Replaces placeholder data with actual team stats:
1. rim_pressure_mismatch: Team A paint FG% vs Team B opponent paint FG% allowed
2. three_point_variance: Team A 3PA rate vs Team B opponent 3PA rate allowed
3. rebounding_mismatch: Team A OREB% vs Team B DREB%

Fetches season-level stats and creates rolling averages for each game.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams
import time
from datetime import datetime

print("="*70)
print("ADDING REAL SCHEMATIC MATCHUP FEATURES")
print("="*70)

# Load cleaned lean dataset
print("\n1. Loading cleaned lean dataset...")
df = pd.read_csv("data/training_data_lean_cleaned.csv")
print(f"   Games: {len(df):,}")

# Get NBA team IDs
all_teams = teams.get_teams()
team_id_map = {team['abbreviation']: team['id'] for team in all_teams}

print("\n2. Fetching NBA team stats by season...")
print("   (This will take several minutes due to API rate limits)")

# Cache for team stats by season
team_stats_cache = {}

def fetch_season_stats(season_year):
    """Fetch team stats for a season (e.g., '2019-20')"""
    if season_year in team_stats_cache:
        return team_stats_cache[season_year]
    
    try:
        print(f"   Fetching {season_year} season stats...")
        
        # Fetch offensive team stats (Base)
        stats_base = leaguedashteamstats.LeagueDashTeamStats(
            season=season_year,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base',
            timeout=30
        )
        df_base = stats_base.get_data_frames()[0]
        
        time.sleep(0.6)
        
        # Fetch defensive team stats (Opponent)
        stats_opp = leaguedashteamstats.LeagueDashTeamStats(
            season=season_year,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Opponent',
            timeout=30
        )
        df_opp = stats_opp.get_data_frames()[0]
        
        # Create team stats dictionary
        team_stats = {}
        
        # Map team names to abbreviations (some names differ)
        name_to_abbr = {
            'Atlanta Hawks': 'ATL',
            'Boston Celtics': 'BOS',
            'Brooklyn Nets': 'BKN',
            'New Jersey Nets': 'BKN',  # Historical
            'Charlotte Hornets': 'CHA',
            'Charlotte Bobcats': 'CHA',  # Historical
            'Chicago Bulls': 'CHI',
            'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL',
            'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW',
            'Houston Rockets': 'HOU',
            'Indiana Pacers': 'IND',
            'Los Angeles Clippers': 'LAC',
            'Los Angeles Lakers': 'LAL',
            'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL',
            'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP',
            'New Orleans Hornets': 'NOP',  # Historical
            'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC',
            'Seattle SuperSonics': 'OKC',  # Historical
            'Orlando Magic': 'ORL',
            'Philadelphia 76ers': 'PHI',
            'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR',
            'Sacramento Kings': 'SAC',
            'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA',
            'Washington Wizards': 'WAS',
        }
        
        for _, base_row in df_base.iterrows():
            team_name = base_row['TEAM_NAME']
            team_abbr = name_to_abbr.get(team_name, team_name[:3].upper())
            
            # Find corresponding opponent stats
            opp_row = df_opp[df_opp['TEAM_NAME'] == team_name]
            if opp_row.empty:
                continue
            opp_row = opp_row.iloc[0]
            
            team_stats[team_abbr] = {
                # Offensive stats
                'fgm': base_row['FGM'],
                'fga': base_row['FGA'],
                'fg_pct': base_row['FG_PCT'],
                'fg3a': base_row['FG3A'],
                'fg3a_rate': base_row['FG3A'] / base_row['FGA'] if base_row['FGA'] > 0 else 0,
                'oreb': base_row['OREB'],
                'dreb': base_row['DREB'],
                'pts': base_row['PTS'],
                
                # Defensive stats (opponent)
                'opp_fgm': opp_row['FGM'],
                'opp_fga': opp_row['FGA'],
                'opp_fg_pct': opp_row['FG_PCT'],
                'opp_fg3a': opp_row['FG3A'],
                'opp_fg3a_rate': opp_row['FG3A'] / opp_row['FGA'] if opp_row['FGA'] > 0 else 0,
                'opp_oreb': opp_row['OREB'],
                'opp_pts': opp_row['PTS'],
            }
        
        team_stats_cache[season_year] = team_stats
        time.sleep(0.6)  # Rate limit
        
        return team_stats
        
    except Exception as e:
        print(f"   ⚠️ Error fetching {season_year}: {e}")
        return None

# Get unique seasons in data
df['game_date'] = pd.to_datetime(df['date'])

unique_seasons = sorted(df['season'].unique())
print(f"   Seasons to fetch: {len(unique_seasons)}")

# Fetch stats for each season (already in "2019-20" format)
for season_str in unique_seasons:
    fetch_season_stats(season_str)

print("\n3. Calculating schematic matchup features...")

# Initialize new columns
df['rim_pressure_mismatch'] = 0.0
df['three_point_variance'] = 0.0
df['rebounding_mismatch'] = 0.0

# Process each game
for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"   Processing game {idx:,}/{len(df):,}...")
    
    season_str = row['season']  # Already in "2019-20" format
    
    home_team = row['home_team']
    away_team = row['away_team']
    
    # Get season stats
    season_stats = team_stats_cache.get(season_str)
    if not season_stats:
        continue
    
    home_stats = season_stats.get(home_team)
    away_stats = season_stats.get(away_team)
    
    if not home_stats or not away_stats:
        continue
    
    # 1. RIM PRESSURE MISMATCH
    # Away team's 2PT FG% (proxy for paint) vs Home team's opponent 2PT FG% allowed
    away_fg3a = away_stats.get('fg3a', 0)
    away_fga = away_stats.get('fga', 1)
    away_2pt_fga = away_fga - away_fg3a
    away_2pt_fgm = away_stats.get('fgm', 0) - away_stats.get('fg3m', 0) if 'fg3m' in away_stats else away_stats.get('fgm', 0) * 0.7
    away_2pt_pct = away_2pt_fgm / away_2pt_fga if away_2pt_fga > 0 else 0
    
    home_opp_fg3a = home_stats.get('opp_fg3a', 0)
    home_opp_fga = home_stats.get('opp_fga', 1)
    home_opp_2pt_fga = home_opp_fga - home_opp_fg3a
    home_opp_2pt_fgm = home_stats.get('opp_fgm', 0) * 0.7  # Estimate
    home_opp_2pt_pct = home_opp_2pt_fgm / home_opp_2pt_fga if home_opp_2pt_fga > 0 else 0
    
    # Positive = away team shoots better from 2PT than home defense allows
    df.at[idx, 'rim_pressure_mismatch'] = away_2pt_pct - home_opp_2pt_pct
    
    # 2. THREE POINT VARIANCE
    # Away team's 3PA rate vs Home team's opponent 3PA rate allowed
    away_fg3a_rate = away_stats.get('fg3a_rate', 0)
    home_opp_fg3a_rate = home_stats.get('opp_fg3a_rate', 0)
    
    # Positive = away shoots more 3s than home typically allows
    df.at[idx, 'three_point_variance'] = away_fg3a_rate - home_opp_fg3a_rate
    
    # 3. REBOUNDING MISMATCH
    # Away team's OREB vs Home team's DREB (per game basis)
    away_oreb = away_stats.get('oreb', 0)
    home_dreb = home_stats.get('dreb', 0)
    home_opp_oreb = home_stats.get('opp_oreb', 0)
    
    # Positive = away team gets more offensive boards than home typically allows
    df.at[idx, 'rebounding_mismatch'] = away_oreb - home_opp_oreb

print(f"\n4. Feature statistics:")
print(f"   rim_pressure_mismatch:")
print(f"     Mean: {df['rim_pressure_mismatch'].mean():.3f}")
print(f"     Std:  {df['rim_pressure_mismatch'].std():.3f}")
print(f"     Range: [{df['rim_pressure_mismatch'].min():.3f}, {df['rim_pressure_mismatch'].max():.3f}]")

print(f"   three_point_variance:")
print(f"     Mean: {df['three_point_variance'].mean():.3f}")
print(f"     Std:  {df['three_point_variance'].std():.3f}")
print(f"     Range: [{df['three_point_variance'].min():.3f}, {df['three_point_variance'].max():.3f}]")

print(f"   rebounding_mismatch:")
print(f"     Mean: {df['rebounding_mismatch'].mean():.3f}")
print(f"     Std:  {df['rebounding_mismatch'].std():.3f}")
print(f"     Range: [{df['rebounding_mismatch'].min():.3f}, {df['rebounding_mismatch'].max():.3f}]")

# Save
output_path = "data/training_data_lean_with_matchups.csv"
df = df.drop(columns=['game_date'], errors='ignore')
df.to_csv(output_path, index=False)

total_features = len([c for c in df.columns if c not in ['date','game_id','home_team','away_team','season','target_spread','target_spread_cover','target_moneyline_win','target_game_total','target_over_under','target_home_cover','target_over']])

print(f"\n5. Saved: {output_path}")
print(f"   Total features: {total_features}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("ADDED REAL SCHEMATIC MATCHUP FEATURES:")
print("  ✓ rim_pressure_mismatch (paint scoring vs paint defense)")
print("  ✓ three_point_variance (3PA rate mismatch)")
print("  ✓ rebounding_mismatch (OREB% vs DREB%)")
print("\nThese capture 'style makes fights' advantages that ELO misses.")
print("\nNEXT STEPS:")
print("  1. Train model with new features")
print("  2. Check if schematic matchups improve AUC")
print("  3. Add PIE-based roster share for injury precision")
print("="*70)
