"""
Check what data is in the cached predictions
"""
import json
from pathlib import Path
from pprint import pprint

data = json.loads(Path('predictions_cache.json').read_text())

# Get one prediction
if data:
    key = list(data.keys())[0]
    pred = data[key]
    
    print('='*80)
    print(f'PREDICTION: {key}')
    print('='*80)
    
    print('\n1. BASIC INFO:')
    print(f"   Home: {pred.get('home_team')}")
    print(f"   Away: {pred.get('away_team')}")
    print(f"   Date: {pred.get('game_date')}")
    print(f"   Home Win Prob: {pred.get('home_win_prob')}")
    print(f"   Away Win Prob: {pred.get('away_win_prob')}")
    
    print('\n2. ODDS INFO:')
    print(f"   Odds Source: {pred.get('odds_source')}")
    print(f"   Has Real Odds: {pred.get('has_real_odds')}")
    print(f"   Kalshi Home Prob: {pred.get('kalshi_home_prob')}")
    print(f"   Yes Price: {pred.get('yes_price')}")
    print(f"   No Price: {pred.get('no_price')}")
    
    print('\n3. INJURIES:')
    home_injuries = pred.get('home_injuries', [])
    away_injuries = pred.get('away_injuries', [])
    print(f"   Home Injuries: {len(home_injuries)}")
    print(f"   Away Injuries: {len(away_injuries)}")
    print(f"   Home Injury Impact: {pred.get('home_injury_impact')}")
    print(f"   Away Injury Impact: {pred.get('away_injury_impact')}")
    
    print('\n4. FEATURES (first 20):')
    features = pred.get('features', {})
    if features:
        for i, (k, v) in enumerate(list(features.items())[:20]):
            print(f"   {k}: {v}")
    else:
        print("   NO FEATURES DICT!")
    
    print('\n5. ALL BETS:')
    all_bets = pred.get('all_bets', [])
    print(f"   Number of bets: {len(all_bets)}")
    if all_bets:
        print(f"   First bet: {all_bets[0]}")
    
    print('\n6. BEST BET:')
    best_bet = pred.get('best_bet')
    if best_bet:
        print(f"   {best_bet}")
    else:
        print("   No qualifying bet")
    
    print('\n' + '='*80)
