"""
Initialize ELO ratings for 2024-25 season using NBA API data
Downloads game results and calculates off/def ELO ratings
"""
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd
import sqlite3
from datetime import datetime
import time
import sys
sys.path.insert(0, '.')

from src.features.off_def_elo_system import OffDefEloSystem

DB_PATH = r"data\live\nba_betting_data.db"

print("=" * 80)
print("INITIALIZING ELO RATINGS FOR 2025-26 SEASON")
print("=" * 80)

# Get all NBA teams
nba_teams = teams.get_teams()
team_abbrevs = [team['abbreviation'] for team in nba_teams]

print(f"\n✅ Found {len(team_abbrevs)} NBA teams")

# Download 2025-26 season games
print("\n📥 Downloading 2025-26 season games from NBA API...")
print("   (This may take 30-60 seconds due to rate limiting...)")

try:
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable='2025-26',
        season_type_nullable='Regular Season',
        league_id_nullable='00'
    )
    games = gamefinder.get_data_frames()[0]
    
    print(f"✅ Downloaded {len(games)} game records")
    
    # Process into game results (each game appears twice - once per team)
    # Group by GAME_ID to get matchups
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values('GAME_DATE')
    
    game_results = []
    processed_ids = set()
    
    for game_id in games['GAME_ID'].unique():
        if game_id in processed_ids:
            continue
        
        game_rows = games[games['GAME_ID'] == game_id]
        if len(game_rows) != 2:
            continue
        
        # Determine home/away (home team is @)
        row1 = game_rows.iloc[0]
        row2 = game_rows.iloc[1]
        
        # MATCHUP format: "TEAM @ OPPONENT" means TEAM is away
        if '@' in row1['MATCHUP']:
            away_row, home_row = row1, row2
        else:
            home_row, away_row = row1, row2
        
        game_results.append({
            'game_id': game_id,
            'game_date': home_row['GAME_DATE'].strftime('%Y-%m-%d'),
            'home_team': home_row['TEAM_ABBREVIATION'],
            'away_team': away_row['TEAM_ABBREVIATION'],
            'home_score': home_row['PTS'],
            'away_score': away_row['PTS'],
            'season': '2025-26'
        })
        
        processed_ids.add(game_id)
    
    print(f"✅ Processed {len(game_results)} unique games")
    
    # Store in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create game_results table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_results (
            game_id TEXT PRIMARY KEY,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_score INTEGER NOT NULL,
            away_score INTEGER NOT NULL,
            season TEXT NOT NULL
        )
    ''')
    
    # Insert games
    for game in game_results:
        cursor.execute('''
            INSERT OR REPLACE INTO game_results 
            (game_id, game_date, home_team, away_team, home_score, away_score, season)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            game['game_id'],
            game['game_date'],
            game['home_team'],
            game['away_team'],
            game['home_score'],
            game['away_score'],
            game['season']
        ))
    
    conn.commit()
    print(f"✅ Stored {len(game_results)} games in database")
    
    # Calculate team records
    records = {}
    for game in game_results:
        home = game['home_team']
        away = game['away_team']
        
        if home not in records:
            records[home] = {'wins': 0, 'losses': 0}
        if away not in records:
            records[away] = {'wins': 0, 'losses': 0}
        
        if game['home_score'] > game['away_score']:
            records[home]['wins'] += 1
            records[away]['losses'] += 1
        else:
            records[away]['wins'] += 1
            records[home]['losses'] += 1
    
    print("\n" + "=" * 80)
    print("TEAM RECORDS (2025-26)")
    print("=" * 80)
    for team in sorted(records.keys()):
        rec = records[team]
        print(f"   {team}: {rec['wins']}-{rec['losses']}")
    
    # Initialize ELO system and calculate ratings
    print("\n" + "=" * 80)
    print("CALCULATING ELO RATINGS")
    print("=" * 80)
    
    elo_system = OffDefEloSystem(DB_PATH)
    
    # Initialize all teams
    elo_system.initialize_season('2025-26', team_abbrevs)
    print(f"✅ Initialized {len(team_abbrevs)} teams with baseline ELO (1500)")
    
    # Process games in chronological order
    sorted_games = sorted(game_results, key=lambda g: g['game_date'])
    
    for i, game in enumerate(sorted_games, 1):
        if i % 50 == 0:
            print(f"   Processed {i}/{len(sorted_games)} games...")
        
        elo_system.update_game(
            season='2025-26',
            game_date=game['game_date'],
            home_team=game['home_team'],
            away_team=game['away_team'],
            home_points=game['home_score'],
            away_points=game['away_score'],
            is_playoffs=False
        )
    
    print(f"✅ Calculated ELO for {len(sorted_games)} games")
    
    # Show latest ELO ratings
    print("\n" + "=" * 80)
    print("LATEST ELO RATINGS")
    print("=" * 80)
    
    cursor.execute("""
        SELECT team, MAX(composite_elo) as elo, MAX(game_date) as last_game
        FROM elo_ratings
        WHERE season = '2025-26'
        GROUP BY team
        ORDER BY elo DESC
    """)
    
    print(f"\n{'Team':<6} {'Record':<10} {'ELO':<8} {'Last Updated'}")
    print("-" * 40)
    for team, elo, last_game in cursor.fetchall():
        rec = records.get(team, {'wins': 0, 'losses': 0})
        record_str = f"{rec['wins']}-{rec['losses']}"
        print(f"{team:<6} {record_str:<10} {elo:>7.0f}  {last_game}")
    
    # Highlight DET and DAL
    print("\n" + "=" * 80)
    cursor.execute("""
        SELECT team, MAX(composite_elo) as elo
        FROM elo_ratings
        WHERE team IN ('DET', 'DAL')
        GROUP BY team
    """)
    
    for team, elo in cursor.fetchall():
        rec = records.get(team, {'wins': 0, 'losses': 0})
        print(f"✅ {team}: {rec['wins']}-{rec['losses']} → ELO {elo:.0f}")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("✅ ELO INITIALIZATION COMPLETE!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
