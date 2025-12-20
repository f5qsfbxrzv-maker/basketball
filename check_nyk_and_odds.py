"""Check NYK injuries and verify odds generation for all teams"""
import sqlite3

# Check NYK injuries
print("="*80)
print("NEW YORK KNICKS INJURY CHECK")
print("="*80)

conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT player_name, status, position, injury_desc 
    FROM active_injuries 
    WHERE team_name = 'New York Knicks'
''')

injuries = cursor.fetchall()
print(f"\nInjuries found: {len(injuries)}")
for player, status, pos, desc in injuries:
    print(f"  {player:25s} {status:15s} {pos:5s} {desc}")

# Check if we have odds data
print("\n" + "="*80)
print("CHECKING ODDS DATA")
print("="*80)

cursor.execute('''
    SELECT COUNT(DISTINCT team_abb) as teams_with_odds
    FROM game_advanced_stats
    WHERE season = '2024-25'
''')

result = cursor.fetchone()
print(f"\nTeams with game data in 2024-25: {result[0] if result else 0}")

# Check historical odds table
try:
    cursor.execute('''
        SELECT COUNT(*) as total_odds,
               COUNT(DISTINCT home_team) as home_teams,
               COUNT(DISTINCT away_team) as away_teams
        FROM historical_odds
        WHERE season = '2024-25'
    ''')
    
    total, home_teams, away_teams = cursor.fetchone()
    print(f"\nHistorical odds records: {total}")
    print(f"Unique home teams with odds: {home_teams}")
    print(f"Unique away teams with odds: {away_teams}")
    
    # Show recent odds
    cursor.execute('''
        SELECT game_date, home_team, away_team, home_ml, away_ml
        FROM historical_odds
        WHERE season = '2024-25'
        ORDER BY game_date DESC
        LIMIT 5
    ''')
    
    print("\nRecent odds samples:")
    for date, home, away, home_ml, away_ml in cursor.fetchall():
        print(f"  {date} {away:3s} @ {home:3s}: Home {home_ml:>5.0f} / Away {away_ml:>5.0f}")
        
except Exception as e:
    print(f"Error checking historical odds: {e}")

conn.close()

# Check if Kalshi client is configured
print("\n" + "="*80)
print("CHECKING KALSHI ODDS CONFIGURATION")
print("="*80)

try:
    from src.services.kalshi_odds_fetcher import KalshiClient
    import json
    
    with open('config/kalshi_credentials.json', 'r') as f:
        creds = json.load(f)
    
    print(f"✓ Kalshi credentials file exists")
    print(f"  Environment: {creds.get('environment', 'N/A')}")
    print(f"  API Key: {creds.get('api_key', 'N/A')[:20]}...")
    
    # Try to initialize client
    try:
        client = KalshiClient()
        print(f"✓ Kalshi client initialized")
        print(f"  Base URL: {client.base_url if hasattr(client, 'base_url') else 'N/A'}")
    except Exception as e:
        print(f"✗ Kalshi client initialization failed: {e}")
        
except Exception as e:
    print(f"✗ Kalshi configuration issue: {e}")

print("\n" + "="*80)
