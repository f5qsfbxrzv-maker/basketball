"""
Download and update all NBA statistics through today (Dec 17, 2025)
Step 1: Download current data
Step 2: Verify stats are correct
Step 3: Only then recalculate ELO
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import sqlite3
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import time

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

print("=" * 80)
print("STEP 1: DOWNLOAD CURRENT NBA DATA (through Dec 17, 2025)")
print("=" * 80)

# Download game logs from NBA API
print("\n📥 Fetching 2024-25 season game logs from NBA API...")
try:
    game_log = leaguegamelog.LeagueGameLog(
        season="2024-25",
        season_type_all_star='Regular Season',
        player_or_team_abbreviation='T'
    )
    time.sleep(1)  # Rate limiting
    
    df = game_log.get_data_frames()[0]
    print(f"✅ Downloaded {len(df)} team-game records")
    
    # Show columns
    print(f"\n📋 Available columns: {list(df.columns)}")
    
except Exception as e:
    print(f"❌ ERROR downloading data: {e}")
    sys.exit(1)

# Convert to game results format (2 rows per game -> 1 row)
print("\n" + "=" * 80)
print("STEP 2: PROCESS DATA INTO GAME RESULTS")
print("=" * 80)

# Filter to games through today only
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
today = pd.to_datetime('2025-12-17')
df = df[df['GAME_DATE'] <= today]

print(f"\n✅ Filtered to games through 12/17/2025: {len(df)} team-games")
print(f"   Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

# Group by GAME_ID to create one row per game
games = []
for game_id, group in df.groupby('GAME_ID'):
    if len(group) != 2:
        print(f"⚠️  Skipping game {game_id} with {len(group)} teams")
        continue
    
    # Sort by whether home/away (MATCHUP contains '@' for away team)
    group = group.sort_values('MATCHUP')
    
    row1 = group.iloc[0]
    row2 = group.iloc[1]
    
    # Determine home/away
    if '@' in row1['MATCHUP']:
        away_row = row1
        home_row = row2
    else:
        home_row = row1
        away_row = row2
    
    games.append({
        'game_id': game_id,
        'game_date': home_row['GAME_DATE'].strftime('%Y-%m-%d'),
        'season': '2024-25',
        'home_team': home_row['TEAM_ABBREVIATION'],
        'away_team': away_row['TEAM_ABBREVIATION'],
        'home_score': int(home_row['PTS']),
        'away_score': int(away_row['PTS']),
        'home_won': 1 if home_row['WL'] == 'W' else 0,
        'total_points': int(home_row['PTS']) + int(away_row['PTS']),
        'point_differential': int(home_row['PTS']) - int(away_row['PTS'])
    })

game_results_df = pd.DataFrame(games)
print(f"\n✅ Processed {len(game_results_df)} games")

# Calculate team records
print("\n" + "=" * 80)
print("STEP 3: VERIFY TEAM RECORDS")
print("=" * 80)

home_records = game_results_df.groupby('home_team').agg({
    'home_won': 'sum',
    'game_id': 'count'
}).rename(columns={'home_won': 'home_wins', 'game_id': 'home_games'})

away_records = game_results_df.groupby('away_team').agg({
    'home_won': lambda x: (x == 0).sum(),  # Away wins = home_won == 0
    'game_id': 'count'
}).rename(columns={'home_won': 'away_wins', 'game_id': 'away_games'})

# Combine
all_teams = set(home_records.index) | set(away_records.index)
team_records = []
for team in all_teams:
    h_wins = home_records.loc[team, 'home_wins'] if team in home_records.index else 0
    h_games = home_records.loc[team, 'home_games'] if team in home_records.index else 0
    a_wins = away_records.loc[team, 'away_wins'] if team in away_records.index else 0
    a_games = away_records.loc[team, 'away_games'] if team in away_records.index else 0
    
    total_wins = int(h_wins + a_wins)
    total_games = int(h_games + a_games)
    total_losses = total_games - total_wins
    
    team_records.append({
        'team': team,
        'wins': total_wins,
        'losses': total_losses,
        'games': total_games
    })

records_df = pd.DataFrame(team_records).sort_values('wins', ascending=False)

print("\n📊 TEAM RECORDS (through 12/17/2025):")
for _, row in records_df.iterrows():
    team, wins, losses, games = row['team'], row['wins'], row['losses'], row['games']
    if team in ['DET', 'DAL']:
        print(f"   {team}: {wins}-{losses} ({games} games) ⭐")
    else:
        print(f"   {team}: {wins}-{losses} ({games} games)")

# Verification check
print("\n" + "=" * 80)
print("STEP 4: VERIFICATION CHECK")
print("=" * 80)

det_record = records_df[records_df['team'] == 'DET']
dal_record = records_df[records_df['team'] == 'DAL']

if not det_record.empty:
    det_wins = int(det_record['wins'].values[0])
    det_losses = int(det_record['losses'].values[0])
    print(f"\n✓ DET record: {det_wins}-{det_losses}")
    if det_wins == 20 and det_losses == 5:
        print("  ✅ MATCHES expected 20-5!")
    else:
        print(f"  ⚠️  Expected 20-5, got {det_wins}-{det_losses}")
else:
    print("  ❌ DET not found!")

if not dal_record.empty:
    dal_wins = int(dal_record['wins'].values[0])
    dal_losses = int(dal_record['losses'].values[0])
    print(f"\n✓ DAL record: {dal_wins}-{dal_losses}")
    if dal_wins == 10 and dal_losses == 16:
        print("  ✅ MATCHES expected 10-16!")
    else:
        print(f"  ⚠️  Expected 10-16, got {dal_wins}-{dal_losses}")
else:
    print("  ❌ DAL not found!")

# Ask for confirmation before writing
print("\n" + "=" * 80)
print("STEP 5: SAVE TO DATABASE")
print("=" * 80)

user_input = input("\n⚠️  Ready to save to database? This will REPLACE existing game_results. (yes/no): ")

if user_input.lower() != 'yes':
    print("❌ Cancelled by user")
    sys.exit(0)

# Save to database
conn = sqlite3.connect(DB_PATH)

# Replace game_results table (keep schedule separate if needed)
game_results_df.to_sql('game_results', conn, if_exists='replace', index=False)

conn.commit()
conn.close()

print(f"\n✅ Saved {len(game_results_df)} games to database")
print("\n" + "=" * 80)
print("DATABASE UPDATED SUCCESSFULLY!")
print("=" * 80)
print("\n📋 Next step: Run ELO recalculation script")
