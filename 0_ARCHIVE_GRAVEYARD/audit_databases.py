"""Comprehensive database audit to verify all data sources for features"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

db_path = 'nba_betting_data.db'
conn = sqlite3.connect(db_path)

print("="*100)
print("DATABASE AUDIT - Feature Data Sources")
print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
print("="*100)

# 1. GAME_LOGS TABLE
print("\n[1] GAME_LOGS TABLE (for rest days, recency stats)")
print("-"*100)
df = pd.read_sql_query("""
    SELECT 
        MIN(game_date) as earliest,
        MAX(game_date) as latest,
        COUNT(*) as total_games,
        COUNT(DISTINCT team_abbreviation) as num_teams
    FROM game_logs
""", conn)
print(df.to_string(index=False))

# Games per team
games_per_team = pd.read_sql_query("""
    SELECT team_abbreviation, COUNT(*) as games_played
    FROM game_logs
    GROUP BY team_abbreviation
    ORDER BY games_played DESC
""", conn)
print(f"\nGames per team (min: {games_per_team['games_played'].min()}, max: {games_per_team['games_played'].max()}, avg: {games_per_team['games_played'].mean():.1f})")

# Check specific teams for today's games
print("\nLast 3 games for TODAY'S TEAMS (ORL, NYK, OKC, SAS):")
for team in ['ORL', 'NYK', 'OKC', 'SAS']:
    last_games = pd.read_sql_query(f"""
        SELECT GAME_DATE, MATCHUP, WL
        FROM game_logs
        WHERE TEAM_ABBREVIATION = '{team}'
        ORDER BY GAME_DATE DESC
        LIMIT 3
    """, conn)
    if not last_games.empty:
        last_date = last_games.iloc[0]['GAME_DATE']
        days_ago = (datetime.strptime('2025-12-13', '%Y-%m-%d') - datetime.strptime(last_date, '%Y-%m-%d')).days
        print(f"  {team}: Last game {last_date} ({days_ago} days ago) - {last_games.iloc[0]['MATCHUP']}")
    else:
        print(f"  {team}: NO GAMES FOUND")

# 2. TEAM_STATS TABLE
print("\n[2] TEAM_STATS TABLE (for baseline stats)")
print("-"*100)
team_stats_info = pd.read_sql_query("""
    SELECT 
        COUNT(DISTINCT team_name) as num_teams,
        COUNT(*) as total_rows
    FROM team_stats
""", conn)
print(team_stats_info.to_string(index=False))
print("NOTE: team_stats is season averages (updated when you run download script)")

# 3. PLAYER_STATS TABLE
print("\n[3] PLAYER_STATS TABLE (for injury PIE values)")
print("-"*100)
player_stats_info = pd.read_sql_query("""
    SELECT 
        COUNT(DISTINCT player_name) as num_players,
        COUNT(*) as total_rows,
        MIN(pie) as min_pie,
        MAX(pie) as max_pie,
        AVG(pie) as avg_pie
    FROM player_stats
""", conn)
print(player_stats_info.to_string(index=False))

# 4. GAME_ADVANCED_STATS TABLE
print("\n[4] GAME_ADVANCED_STATS TABLE (for Four Factors)")
print("-"*100)
adv_stats_info = pd.read_sql_query("""
    SELECT 
        MIN(game_date) as earliest,
        MAX(game_date) as latest,
        COUNT(*) as total_games,
        COUNT(DISTINCT team_abbreviation) as num_teams
    FROM game_advanced_stats
""", conn)
print(adv_stats_info.to_string(index=False))

# Check if game_advanced_stats is up to date with game_logs
game_logs_latest = pd.read_sql_query("SELECT MAX(game_date) as latest FROM game_logs", conn).iloc[0]['latest']
game_adv_latest = pd.read_sql_query("SELECT MAX(game_date) as latest FROM game_advanced_stats", conn).iloc[0]['latest']
if game_logs_latest == game_adv_latest:
    print(f"✓ game_advanced_stats is UP TO DATE with game_logs ({game_adv_latest})")
else:
    print(f"✗ game_advanced_stats ({game_adv_latest}) is BEHIND game_logs ({game_logs_latest})")

# 5. ACTIVE_INJURIES TABLE
print("\n[5] ACTIVE_INJURIES TABLE (for injury impact)")
print("-"*100)
injuries_info = pd.read_sql_query("""
    SELECT 
        COUNT(*) as total_injuries,
        COUNT(DISTINCT team_name) as teams_with_injuries,
        COUNT(DISTINCT player_name) as injured_players,
        MAX(last_updated) as last_updated
    FROM active_injuries
""", conn)
print(injuries_info.to_string(index=False))

# Check injuries for today's teams
print("\nInjuries for TODAY'S TEAMS:")
for team in ['Orlando Magic', 'New York Knicks', 'Oklahoma City Thunder', 'San Antonio Spurs']:
    inj_count = pd.read_sql_query(f"""
        SELECT COUNT(*) as cnt
        FROM active_injuries
        WHERE team_name = '{team}'
    """, conn).iloc[0]['cnt']
    if inj_count > 0:
        injuries = pd.read_sql_query(f"""
            SELECT player_name, status
            FROM active_injuries
            WHERE team_name = '{team}'
        """, conn)
        print(f"  {team}: {inj_count} injuries")
        for _, inj in injuries.iterrows():
            print(f"    - {inj['player_name']} ({inj['status']})")
    else:
        print(f"  {team}: No injuries")

# 6. MANUAL REST DAYS CALCULATION
print("\n[6] MANUAL REST DAYS CALCULATION VERIFICATION")
print("-"*100)
print("Calculating rest days manually for today (2025-12-13):")

for team in ['ORL', 'NYK', 'OKC', 'SAS']:
    last_game = pd.read_sql_query(f"""
        SELECT GAME_DATE
        FROM game_logs
        WHERE TEAM_ABBREVIATION = '{team}'
        ORDER BY GAME_DATE DESC
        LIMIT 1
    """, conn)
    
    if not last_game.empty:
        last_date = last_game.iloc[0]['GAME_DATE']
        today = datetime.strptime('2025-12-13', '%Y-%m-%d')
        last = datetime.strptime(last_date, '%Y-%m-%d')
        rest_days = (today - last).days - 1  # Subtract 1 because day of game doesn't count
        print(f"  {team}: Last game {last_date} → Rest days = {rest_days}")
    else:
        print(f"  {team}: No games found")

print("\n" + "="*100)
print("AUDIT SUMMARY")
print("="*100)

# Check data freshness
today = datetime.now().date()
game_logs_date = datetime.strptime(game_logs_latest, '%Y-%m-%d').date()
days_behind = (today - game_logs_date).days

if days_behind == 0:
    print("✓ game_logs is CURRENT (today)")
elif days_behind == 1:
    print("✓ game_logs is 1 day old (acceptable - games from yesterday)")
else:
    print(f"✗ game_logs is {days_behind} days old - NEEDS UPDATE")

print(f"\nExpected rest days for today's teams:")
print(f"  ORL/NYK: Should be 3 days (last played Dec 9)")
print(f"  OKC/SAS: Should be 2 days (last played Dec 10)")

conn.close()
