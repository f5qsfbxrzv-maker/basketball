"""
Detailed analysis of DAL vs DET prediction for 12-18-2025
Shows all features, ELO ratings, injury impacts, and model reasoning
"""
import sys
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_prediction_engine_v5 import NBAPredictionEngine
from src.services.espn_schedule_service import ESPNScheduleService
from datetime import datetime

print("="*80)
print("DALLAS @ DETROIT PREDICTION ANALYSIS - 12/18/2025")
print("="*80)

# Initialize services
print("\n[1] Initializing prediction engine...")
predictor = NBAPredictionEngine()

print("\n[2] Fetching schedule for 12/18/2025...")
schedule_service = ESPNScheduleService()
games = schedule_service.fetch_games_for_date('2025-12-18', save_to_db=True)

print(f"\nGames on 12/18/2025:")
for game in games:
    matchup = f"{game['away_team']} @ {game['home_team']}"
    if game['away_team'] == 'DAL' or game['home_team'] == 'DAL':
        print(f"  >>> {matchup} <<<")
    else:
        print(f"  {matchup}")

# Find DAL @ DET
dal_det_game = None
for game in games:
    if (game['away_team'] == 'DAL' and game['home_team'] == 'DET') or \
       (game['away_team'] == 'DET' and game['home_team'] == 'DAL'):
        dal_det_game = game
        break

if not dal_det_game:
    print("\n[ERROR] Could not find DAL vs DET game on 12/18/2025")
    print("Available games:", [(g['away_team'], g['home_team']) for g in games])
    sys.exit(1)

home_team = dal_det_game['home_team']
away_team = dal_det_game['away_team']

print(f"\n[3] Analyzing: {away_team} @ {home_team}")
print("="*80)

# Get prediction with detailed logging
print("\n[4] Running prediction engine...")
prediction = predictor.predict_game(
    home_team=home_team,
    away_team=away_team,
    game_date='2025-12-18',
    game_time=dal_det_game.get('game_time', 'TBD'),
    home_ml_odds=-110,
    away_ml_odds=-110
)

if 'error' in prediction:
    print(f"\n[ERROR] {prediction['error']}")
    sys.exit(1)

print("\n" + "="*80)
print("PREDICTION RESULTS")
print("="*80)

# Win probabilities
print(f"\n📊 WIN PROBABILITIES:")
print(f"  {home_team}: {prediction['home_win_prob']*100:.1f}%")
print(f"  {away_team}: {prediction['away_win_prob']*100:.1f}%")

# Best bet
best_bet = prediction.get('best_bet')
if best_bet:
    print(f"\n💰 BEST BET:")
    print(f"  Pick: {best_bet['pick']} ({best_bet['type']})")
    print(f"  Edge: {best_bet['edge']*100:.2f}%")
    print(f"  Odds: {best_bet['odds']:+d}")
    print(f"  Stake: ${best_bet['stake']:.2f}")
    print(f"  Class: {best_bet['bet_class']}")
    print(f"  Qualifies: {best_bet['qualifies']}")
else:
    print(f"\n❌ NO QUALIFYING BET")

# Feature analysis
print("\n" + "="*80)
print("FEATURE ANALYSIS")
print("="*80)

if 'features' in prediction:
    features = prediction['features']
    
    # Group features by category
    elo_features = {k: v for k, v in features.items() if 'elo' in k.lower()}
    injury_features = {k: v for k, v in features.items() if 'injury' in k.lower()}
    fatigue_features = {k: v for k, v in features.items() if 'fatigue' in k.lower() or 'rest' in k.lower()}
    momentum_features = {k: v for k, v in features.items() if 'streak' in k.lower() or 'momentum' in k.lower()}
    efficiency_features = {k: v for k, v in features.items() if 'efg' in k.lower() or 'ts' in k.lower()}
    
    print("\n🏀 ELO RATINGS:")
    for feat, val in sorted(elo_features.items()):
        print(f"  {feat:40s}: {val:8.2f}")
    
    print("\n🏥 INJURY IMPACT:")
    for feat, val in sorted(injury_features.items()):
        print(f"  {feat:40s}: {val:8.2f}")
    
    print("\n😴 FATIGUE / REST:")
    for feat, val in sorted(fatigue_features.items()):
        print(f"  {feat:40s}: {val:8.2f}")
    
    print("\n📈 MOMENTUM:")
    for feat, val in sorted(momentum_features.items()):
        print(f"  {feat:40s}: {val:8.2f}")
    
    print("\n🎯 EFFICIENCY:")
    for feat, val in sorted(efficiency_features.items()):
        print(f"  {feat:40s}: {val:8.2f}")
    
    # Top 10 features by absolute value (impact)
    print("\n🔝 TOP 10 MOST IMPACTFUL FEATURES (by magnitude):")
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for i, (feat, val) in enumerate(sorted_features, 1):
        direction = "⬆️" if val > 0 else "⬇️"
        print(f"  {i:2d}. {feat:40s}: {val:8.2f} {direction}")

# Injury report
print("\n" + "="*80)
print("INJURY REPORT")
print("="*80)

# Check injuries from predictor's injury data
try:
    from nba_prediction_engine_v5 import get_injury_context
    injury_context = get_injury_context(home_team, away_team, '2025-12-18')
    
    print(f"\n{home_team} (Home) Injuries:")
    home_injuries = injury_context.get('home_injuries', [])
    if home_injuries:
        for inj in home_injuries:
            print(f"  - {inj.get('player', 'Unknown'):25s} ({inj.get('status', 'Unknown'):15s}) Impact: {inj.get('impact', 0):.2f} pts")
    else:
        print(f"  ✅ No significant injuries")
    
    print(f"\n{away_team} (Away) Injuries:")
    away_injuries = injury_context.get('away_injuries', [])
    if away_injuries:
        for inj in away_injuries:
            print(f"  - {inj.get('player', 'Unknown'):25s} ({inj.get('status', 'Unknown'):15s}) Impact: {inj.get('impact', 0):.2f} pts")
    else:
        print(f"  ✅ No significant injuries")
        
except Exception as e:
    print(f"\n[INFO] Could not fetch detailed injury data: {e}")

# All bets analyzed
print("\n" + "="*80)
print("ALL BET OPPORTUNITIES")
print("="*80)

all_bets = prediction.get('all_bets', [])
if all_bets:
    for i, bet in enumerate(all_bets, 1):
        print(f"\n{i}. {bet['type']} - {bet['pick']}")
        print(f"   Edge: {bet['edge']*100:+.2f}% | Model Prob: {bet['model_prob']*100:.1f}% | Market Prob: {bet['market_prob']*100:.1f}%")
        print(f"   Odds: {bet['odds']:+d} | Stake: ${bet['stake']:.2f} | Class: {bet['bet_class']}")
        print(f"   Threshold: {bet['threshold']*100:.1f}% | Qualifies: {'✅' if bet['qualifies'] else '❌'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
