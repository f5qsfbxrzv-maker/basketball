"""Test Kalshi with corrected parsing"""
from multi_source_odds_service import MultiSourceOddsService
from datetime import datetime
import json

service = MultiSourceOddsService()

print("🏀 Testing MIN @ GSW\n")
odds = service.get_game_odds('GSW', 'MIN', datetime.now())
print(json.dumps(odds, indent=2))

if odds['source'] == 'Kalshi' and odds['kalshi_home_prob'] and odds['kalshi_away_prob']:
    print("\n✅ BOTH HOME AND AWAY ODDS WORKING!")
    print(f"Home (GSW) probability: {odds['kalshi_home_prob']:.1%} → {odds['home_ml_odds']}")
    print(f"Away (MIN) probability: {odds['kalshi_away_prob']:.1%} → {odds['away_ml_odds']}")
else:
    print(f"\n⚠️ Issue with odds:")
    print(f"  Source: {odds['source']}")
    print(f"  Home prob: {odds['kalshi_home_prob']}")
    print(f"  Away prob: {odds['kalshi_away_prob']}")
