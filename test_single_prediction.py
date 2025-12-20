"""
Test single game prediction with updated ELO
"""
from nba_gui_dashboard_v2 import NBAPredictionEngine
import json

engine = NBAPredictionEngine()

print('='*80)
print('TESTING LAL @ LAC PREDICTION WITH REAL ELO')
print('='*80)

pred = engine.predict_game(
    home_team='LAC',
    away_team='LAL', 
    game_date='2025-12-20',
    game_time='22:30'
)

if 'error' in pred:
    print(f"\n❌ ERROR: {pred.get('error')}")
    print(f"   Message: {pred.get('message')}")
else:
    print(f"\nMODEL PREDICTION:")
    print(f"  LAC (home): {pred['home_win_prob']:.1%}")
    print(f"  LAL (away): {pred['away_win_prob']:.1%}")
    
    print(f"\nMARKET ODDS:")
    print(f"  LAC prob: {pred.get('kalshi_home_prob', 0):.1%}")
    print(f"  LAL prob: {pred.get('kalshi_away_prob', 0):.1%}")
    
    print(f"\nBEST BET:")
    best = pred.get('best_bet')
    if best:
        print(f"  Pick: {best['pick']}")
        print(f"  Edge: {best['edge']:.1%}")
        print(f"  Class: {best['bet_class']}")
    else:
        print("  No qualifying bet")
    
    print(f"\nKEY FEATURES:")
    features = pred.get('features', {})
    print(f"  off_elo_diff: {features.get('off_elo_diff', 0):.1f}")
    print(f"  def_elo_diff: {features.get('def_elo_diff', 0):.1f}")
    print(f"  home_composite_elo: {features.get('home_composite_elo', 0):.0f}")
    print(f"  away_composite_elo: {features.get('away_composite_elo', 0):.0f}")

print('='*80)
