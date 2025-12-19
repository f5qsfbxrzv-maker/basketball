"""Test Kalshi with real game"""
from multi_source_odds_service import MultiSourceOddsService
from datetime import datetime
import json

service = MultiSourceOddsService()

print("🏀 Testing MIN @ GSW\n")
odds = service.get_game_odds('GSW', 'MIN', datetime.now())
print(json.dumps(odds, indent=2))

if odds['source'] == 'Kalshi':
    print("\n✅ KALSHI WORKING!")
    print(f"Home (GSW) probability: {odds['kalshi_home_prob']:.1%}")
    print(f"Away (MIN) probability: {odds['kalshi_away_prob']:.1%}")
else:
    print(f"\n❌ Source: {odds['source']}")
