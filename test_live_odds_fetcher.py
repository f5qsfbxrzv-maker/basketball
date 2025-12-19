"""
Test LiveOddsFetcher with Working Credentials
"""

import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'services'))

from live_odds_fetcher import LiveOddsFetcher

def main():
    print("=" * 70)
    print("TESTING LIVE ODDS FETCHER")
    print("=" * 70)
    
    # Initialize fetcher (should auto-load from .kalshi_credentials)
    print("\n[1] Initializing LiveOddsFetcher...")
    fetcher = LiveOddsFetcher()
    
    if fetcher.kalshi_client:
        print("[SUCCESS] Kalshi client loaded!")
        
        # Test authentication
        try:
            account = fetcher.kalshi_client.get_account_info()
            if account:
                balance = account.get('balance', 0)
                print(f"[SUCCESS] Connected! Balance: ${balance / 100:.2f}")
        except:
            pass
    else:
        print("[WARNING] No Kalshi client - will use defaults")
    
    # Test fetching odds for various games
    print("\n[2] Testing moneyline odds fetching...")
    
    test_games = [
        ('MIN', 'DAL', '2025-12-16', 'Timberwolves vs Mavericks'),
        ('GSW', 'LAL', '2025-12-16', 'Warriors vs Lakers'),
        ('BOS', 'MIA', '2025-12-16', 'Celtics vs Heat'),
        ('DEN', 'LAC', '2025-12-16', 'Nuggets vs Clippers'),
    ]
    
    for home, away, date, description in test_games:
        print(f"\n   {description} ({home} vs {away}):")
        odds = fetcher.get_moneyline_odds(home, away, date)
        
        source = odds.get('source', 'unknown')
        home_ml = odds.get('home_ml', 0)
        away_ml = odds.get('away_ml', 0)
        
        print(f"      Source: {source}")
        print(f"      Home ML: {home_ml:+.0f}")
        print(f"      Away ML: {away_ml:+.0f}")
        
        if source == 'kalshi':
            yes_price = odds.get('yes_price', 0)
            no_price = odds.get('no_price', 0)
            print(f"      Kalshi Prices: {yes_price}c (home) / {no_price}c (away)")
            print(f"      [SUCCESS] LIVE KALSHI ODDS!")
        elif source == 'default':
            print(f"      [INFO] Using default odds (no market found)")
    
    # Test vig removal
    print("\n[3] Testing vig removal...")
    test_ml_pairs = [
        (-110, -110, "Even odds"),
        (-200, +170, "Heavy favorite"),
        (+150, -180, "Underdog home"),
    ]
    
    for home_ml, away_ml, description in test_ml_pairs:
        home_fair, away_fair = fetcher.remove_vig(home_ml, away_ml)
        print(f"   {description}: {home_ml:+d} / {away_ml:+d}")
        print(f"      Fair probs: {home_fair:.1%} home / {away_fair:.1%} away")
        print(f"      Sum: {(home_fair + away_fair):.3f} (should be 1.000)")
    
    print("\n" + "=" * 70)
    print("LIVE ODDS FETCHER TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
