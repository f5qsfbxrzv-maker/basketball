"""Quick test of LiveOddsFetcher"""
import sys
sys.path.insert(0, 'src/services')
from live_odds_fetcher import LiveOddsFetcher

f = LiveOddsFetcher()
odds = f.get_moneyline_odds('CLE', 'CHI', '2025-12-17')

print(f"Source: {odds['source']}")
print(f"CLE: {odds.get('home_ml')} (Price: {odds.get('yes_price')})")
print(f"CHI: {odds.get('away_ml')} (Price: {odds.get('no_price')})")

if odds['source'] == 'kalshi' and odds.get('yes_price', 0) > 50:
    print("\n✅ LiveOddsFetcher working correctly!")
else:
    print("\n❌ LiveOddsFetcher not working")
