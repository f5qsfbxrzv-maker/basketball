"""
Update game_advanced_stats table with recent data (Nov 21 - Dec 7, 2025)
Fetches advanced stats from NBA API for the missing date range
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
import time

# Database path
DB_PATH = "c:/Users/d76do/OneDrive/Documents/New Basketball Model/data/live/nba_betting_data.db"

def get_nba_teams():
    """Get all NBA team abbreviations"""
    all_teams = teams.get_teams()
    return [team['abbreviation'] for team in all_teams]

def fetch_advanced_stats(team_abbrev, season='2024-25', season_type='Regular Season'):
    """
    Fetch game logs for a team and calculate advanced stats
    
    Note: NBA API doesn't have a direct "advanced stats" endpoint,
    so we'll fetch box scores and calculate the Four Factors
    """
    try:
        # Fetch team game logs
        gamelog = teamgamelog.TeamGameLog(
            team_id=get_team_id(team_abbrev),
            season=season,
            season_type_all_star=season_type
        )
        df = gamelog.get_data_frames()[0]
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate advanced stats (Four Factors approximations)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df['team_abb'] = team_abbrev
        
        # eFG% = (FGM + 0.5 * FG3M) / FGA
        df['efg_pct'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, 1)
        
        # TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        df['tov_pct'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV']).replace(0, 1)
        
        # ORB% requires opponent stats - use placeholder for now
        df['orb_pct'] = 0.25  # League average placeholder
        
        # FT Rate = FTA / FGA
        df['fta_rate'] = df['FTA'] / df['FGA'].replace(0, 1)
        
        # Pace estimate (possessions per game)
        df['pace'] = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']
        
        # Offensive/Defensive Rating (points per 100 possessions)
        df['off_rating'] = (df['PTS'] / df['pace'].replace(0, 1)) * 100
        
        # Defensive rating (if opponent points available, else use placeholder)
        if 'OPP_PTS' in df.columns:
            df['def_rating'] = (df['OPP_PTS'] / df['pace'].replace(0, 1)) * 100
        else:
            df['def_rating'] = 110  # League average placeholder
        
        df['net_rating'] = df['off_rating'] - df['def_rating']
        
        # 3-point volume (per 100 possessions)
        df['fg3a_per_100'] = (df['FG3A'] / df['pace'].replace(0, 1)) * 100
        df['fg3_pct'] = df['FG3_PCT']
        
        # Select relevant columns
        return df[[
            'Game_ID', 'GAME_DATE', 'team_abb', 'MATCHUP',
            'efg_pct', 'tov_pct', 'orb_pct', 'fta_rate', 'pace',
            'off_rating', 'def_rating', 'net_rating',
            'fg3a_per_100', 'fg3_pct', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'TOV'
        ]].rename(columns={
            'Game_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup'
        })
        
    except Exception as e:
        print(f"Error fetching {team_abbrev}: {e}")
        return pd.DataFrame()

def get_team_id(team_abbrev):
    """Get NBA team ID from abbreviation"""
    all_teams = teams.get_teams()
    for team in all_teams:
        if team['abbreviation'] == team_abbrev:
            return team['id']
    return None

def update_database(start_date='2024-11-21', end_date='2024-12-07'):
    """Update game_advanced_stats with data from start_date to end_date"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check current data range
    cursor.execute("SELECT MIN(game_date), MAX(game_date), COUNT(*) FROM game_advanced_stats")
    current_min, current_max, current_count = cursor.fetchone()
    
    print(f"Current data range: {current_min} to {current_max} ({current_count} rows)")
    print(f"Updating with data from {start_date} to {end_date}...")
    
    # Get all teams
    nba_teams = get_nba_teams()
    
    all_stats = []
    for i, team in enumerate(nba_teams, 1):
        print(f"[{i}/{len(nba_teams)}] Fetching {team}...", end=" ")
        
        stats_df = fetch_advanced_stats(team, season='2024-25')
        
        if not stats_df.empty:
            # Filter to date range
            stats_df['game_date'] = pd.to_datetime(stats_df['game_date'])
            mask = (stats_df['game_date'] >= start_date) & (stats_df['game_date'] <= end_date)
            filtered = stats_df[mask]
            
            if not filtered.empty:
                all_stats.append(filtered)
                print(f"{len(filtered)} games")
            else:
                print("0 games in range")
        else:
            print("No data")
        
        # Rate limiting
        time.sleep(0.6)
    
    if not all_stats:
        print("\nNo new data to insert")
        conn.close()
        return
    
    # Combine all data
    combined = pd.concat(all_stats, ignore_index=True)
    combined['game_date'] = combined['game_date'].dt.strftime('%Y-%m-%d')
    
    print(f"\nTotal new records: {len(combined)}")
    
    # Delete existing records in date range to avoid duplicates
    cursor.execute("""
        DELETE FROM game_advanced_stats 
        WHERE game_date >= ? AND game_date <= ?
    """, (start_date, end_date))
    deleted = cursor.rowcount
    print(f"Deleted {deleted} existing records in date range")
    
    # Insert new data
    combined.to_sql('game_advanced_stats', conn, if_exists='append', index=False)
    conn.commit()
    
    # Verify update
    cursor.execute("SELECT MIN(game_date), MAX(game_date), COUNT(*) FROM game_advanced_stats")
    new_min, new_max, new_count = cursor.fetchone()
    
    print(f"\nUpdated data range: {new_min} to {new_max} ({new_count} rows)")
    print(f"Added {new_count - current_count + deleted} net new rows")
    
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    update_database(start_date='2024-11-21', end_date='2024-12-07')
