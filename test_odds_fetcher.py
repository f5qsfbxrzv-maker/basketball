"""Test LiveOddsFetcher with Kalshi integration"""
import sys
sys.path.append('c:\\Users\\d76do\\OneDrive\\Documents\\New Basketball Model')

from src.services.live_odds_fetcher import LiveOddsFetcher
from datetime import datetime, timedelta

def test_odds_fetcher():
    print("=" * 60)
    print("LIVE ODDS FETCHER TEST")
    print("=" * 60)
    
    # Initialize fetcher
    print("\n1. Initializing LiveOddsFetcher...")
    fetcher = LiveOddsFetcher()
    
    if fetcher.kalshi_client:
        print("   ✅ Kalshi client connected!")
    else:
        print("   ⚠️ No Kalshi client (will use defaults)")
    
    # Test with tomorrow's games
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n2. Testing odds fetch for tomorrow ({tomorrow})...")
    print("   Teams: Lakers (LAL) @ Cavaliers (CLE)")
    
    odds = fetcher.get_moneyline_odds(
        home_team='CLE',
        away_team='LAL',
        game_date=tomorrow
    )
    
    print("\n3. Results:")
    print(f"   Home ML (CLE): {odds.get('home_ml', 'N/A')}")
    print(f"   Away ML (LAL): {odds.get('away_ml', 'N/A')}")
    print(f"   Source: {odds.get('source', 'unknown')}")
    
    if 'yes_price' in odds:
        print(f"   Kalshi Yes Price: {odds['yes_price']}¢")
        print(f"   Kalshi No Price: {odds['no_price']}¢")
    
    print("\n" + "=" * 60)
    print("✅ ODDS FETCHER TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_odds_fetcher()
