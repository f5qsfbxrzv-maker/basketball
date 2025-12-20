"""
Analyze specific questionable predictions
"""
import json
from pathlib import Path
from datetime import datetime

# Load predictions
data = json.loads(Path('predictions_cache.json').read_text())

# Find the specific games
lac_lal_key = '2025-12-20_LAL@LAC'
orl_uta_key = '2025-12-20_ORL@UTA'

print('='*80)
print('QUESTIONABLE PREDICTIONS ANALYSIS')
print('='*80)

for key in [lac_lal_key, orl_uta_key]:
    if key in data:
        pred = data[key]
        features = pred.get('features', {})
        
        away = pred['away_team']
        home = pred['home_team']
        
        print(f"\n{away} @ {home}")
        print('-'*80)
        
        print(f"\nMODEL PREDICTION:")
        print(f"  Home ({home}): {pred.get('home_win_prob', 0):.1%}")
        print(f"  Away ({away}): {pred.get('away_win_prob', 0):.1%}")
        
        print(f"\nMARKET ODDS:")
        print(f"  Home prob: {pred.get('kalshi_home_prob', 0):.1%}")
        print(f"  Away prob: {pred.get('kalshi_away_prob', 0):.1%}")
        print(f"  Source: {pred.get('odds_source')}")
        
        print(f"\nKEY FEATURES:")
        print(f"  Home Win %: {features.get('home_win_pct', 0):.3f}")
        print(f"  Away Win %: {features.get('away_win_pct', 0):.3f}")
        print(f"  Home Composite ELO: {features.get('home_composite_elo', 0):.0f}")
        print(f"  Away Composite ELO: {features.get('away_composite_elo', 0):.0f}")
        print(f"  Off ELO Diff: {features.get('off_elo_diff', 0):.1f}")
        print(f"  Def ELO Diff: {features.get('def_elo_diff', 0):.1f}")
        print(f"  Home Off Rating: {features.get('home_off_rating', 0):.1f}")
        print(f"  Away Off Rating: {features.get('away_off_rating', 0):.1f}")
        print(f"  Home Def Rating: {features.get('home_def_rating', 0):.1f}")
        print(f"  Away Def Rating: {features.get('away_def_rating', 0):.1f}")
        
        print(f"\nINJURIES:")
        print(f"  {home} injury impact: {pred.get('home_injury_impact', 0):.1f} pts ({len(pred.get('home_injuries', []))} players)")
        print(f"  {away} injury impact: {pred.get('away_injury_impact', 0):.1f} pts ({len(pred.get('away_injuries', []))} players)")
        
        home_injuries = pred.get('home_injuries', [])
        away_injuries = pred.get('away_injuries', [])
        
        if home_injuries:
            print(f"\n  {home} injured:")
            for inj in home_injuries[:5]:
                print(f"    - {inj.get('player')} ({inj.get('status')}): {inj.get('injury')}")
        
        if away_injuries:
            print(f"\n  {away} injured:")
            for inj in away_injuries[:5]:
                print(f"    - {inj.get('player')} ({inj.get('status')}): {inj.get('injury')}")
        
        print(f"\nBEST BET:")
        best_bet = pred.get('best_bet')
        if best_bet:
            print(f"  Pick: {best_bet.get('pick')}")
            print(f"  Edge: {best_bet.get('edge', 0):.1%}")
            print(f"  Odds: {best_bet.get('odds'):+d}")
            print(f"  Stake: ${best_bet.get('stake', 0):.2f}")
            print(f"  Class: {best_bet.get('bet_class')}")
        else:
            print("  No qualifying bet")

print('\n' + '='*80)
print('DIAGNOSIS')
print('='*80)
print("Check if:")
print("1. Win percentages seem accurate for current season")
print("2. ELO ratings reflect actual team strength")
print("3. Injuries are being weighted correctly")
print("4. Model is overweighting home court advantage")
print('='*80)
