import json
from pathlib import Path
from datetime import datetime

# Load predictions cache
data = json.loads(Path('predictions_cache.json').read_text())

# Sort by date (most recent first)
games = sorted(data.items(), key=lambda x: x[1]['game_date'], reverse=True)

print('='*80)
print('MOST RECENT PREDICTIONS IN CACHE')
print('='*80)

for i, (key, game) in enumerate(games[:10], 1):
    date = game.get('game_date', 'N/A')
    away = game.get('away_team', 'N/A')
    home = game.get('home_team', 'N/A')
    model_prob = game.get('model_home_prob', 0)
    home_odds = game.get('home_ml_odds', 'N/A')
    away_odds = game.get('away_ml_odds', 'N/A')
    edge = game.get('edge_percentage', 0)
    rec = game.get('recommendation', 'PASS')
    
    print(f"{i}. {date}: {away} @ {home}")
    print(f"   Model: {model_prob:.1%} home win | Odds: {home_odds}/{away_odds}")
    print(f"   Edge: {edge:.2%} | Rec: {rec}")
    print()

print(f"Total cached predictions: {len(data)}")
print('='*80)
