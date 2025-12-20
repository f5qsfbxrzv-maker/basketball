"""Check odds tables and Kalshi setup"""
import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

print("="*80)
print("DATABASE TABLES")
print("="*80)

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()
print("\nAll tables:")
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()[0]
    print(f"  {table[0]:30s} {count:>10,} rows")

print("\n" + "="*80)
print("CHECKING KALSHI CLIENT")
print("="*80)

try:
    from src.services.kalshi_client import KalshiClient
    import json
    
    with open('config/kalshi_credentials.json', 'r') as f:
        creds = json.load(f)
    
    print("\n✓ Kalshi credentials loaded")
    print(f"  Environment: {creds.get('environment')}")
    
    client = KalshiClient()
    print("\n✓ Kalshi client initialized")
    print(f"  Base URL: {client.base_url}")
    print(f"  Authenticated: {client.authenticated}")
    
except Exception as e:
    print(f"\n✗ Kalshi client issue: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("CHECKING DASHBOARD ODDS FETCHER")
print("="*80)

try:
    from src.services.live_odds_fetcher import LiveOddsFetcher
    
    fetcher = LiveOddsFetcher(db_path='data/live/nba_betting_data.db')
    print("\n✓ LiveOddsFetcher initialized")
    
    # Check if it has Kalshi client
    if hasattr(fetcher, 'kalshi_client') and fetcher.kalshi_client:
        print("✓ LiveOddsFetcher has Kalshi client")
    else:
        print("✗ LiveOddsFetcher missing Kalshi client")
        
except Exception as e:
    print(f"\n✗ LiveOddsFetcher issue: {e}")
    import traceback
    traceback.print_exc()

conn.close()
