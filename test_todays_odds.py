"""
Test Kalshi odds fetching for today's games
"""
from src.services.live_odds_fetcher import LiveOddsFetcher
from datetime import datetime

fetcher = LiveOddsFetcher()

# Today's games
games = [
    ('DAL', 'PHI', '2025-12-20'),
    ('BOS', 'TOR', '2025-12-20'),
    ('IND', 'NOP', '2025-12-20'),
    ('CHA', 'DET', '2025-12-20'),
    ('WAS', 'MEM', '2025-12-20'),
    ('PHX', 'GSW', '2025-12-20'),
    ('ORL', 'UTA', '2025-12-20'),
    ('POR', 'SAC', '2025-12-20'),
    ('LAL', 'LAC', '2025-12-20'),
    ('HOU', 'DEN', '2025-12-20'),
]

print('='*80)
print('KALSHI ODDS CHECK - TODAY\'S GAMES (Dec 20, 2025)')
print('='*80)

proper_odds = []
default_odds = []
no_odds = []

for away, home, date in games:
    print(f"\n{away} @ {home} ({date}):")
    
    try:
        odds_data = fetcher.get_moneyline_odds(home, away, date)
        
        if odds_data is None:
            print(f"  ❌ NO ODDS RETURNED (None)")
            no_odds.append(f"{away} @ {home}")
        else:
            home_ml = odds_data.get('home_ml')
            away_ml = odds_data.get('away_ml')
            source = odds_data.get('source', 'unknown')
            yes_price = odds_data.get('yes_price')
            no_price = odds_data.get('no_price')
            
            # Check if these are default -110 odds
            if home_ml == -110 and away_ml == -110:
                print(f"  ⚠️  DEFAULT ODDS: {home_ml}/{away_ml} (source: {source})")
                default_odds.append(f"{away} @ {home}")
            else:
                print(f"  ✅ REAL ODDS: {home_ml}/{away_ml} (source: {source})")
                print(f"     Prices: {yes_price}c / {no_price}c")
                proper_odds.append(f"{away} @ {home}")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        no_odds.append(f"{away} @ {home}")

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f"✅ Real odds: {len(proper_odds)} games")
for game in proper_odds:
    print(f"   - {game}")

print(f"\n⚠️  Default -110: {len(default_odds)} games")
for game in default_odds:
    print(f"   - {game}")

print(f"\n❌ No odds (None): {len(no_odds)} games")
for game in no_odds:
    print(f"   - {game}")

print('='*80)
