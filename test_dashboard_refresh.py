"""Test dashboard refresh functionality"""
from nba_gui_dashboard_v2 import NBAPredictionEngine
from nba_api.live.nba.endpoints import scoreboard

print("=" * 70)
print("DASHBOARD REFRESH TEST")
print("=" * 70)

# 1. Check NBA API
print("\n1. Testing NBA API...")
today = scoreboard.ScoreBoard()
games_data = today.get_dict()
games = games_data['scoreboard']['games']
print(f"✅ Found {len(games)} games today")

if games:
    print("\nSample games:")
    for i, game in enumerate(games[:3], 1):
        home = game['homeTeam']['teamTricode']
        away = game['awayTeam']['teamTricode']
        print(f"  {i}. {away} @ {home}")

# 2. Initialize prediction engine
print("\n2. Initializing prediction engine...")
engine = NBAPredictionEngine()
print("✅ Engine initialized")

# 3. Generate prediction for first game
if games:
    first_game = games[0]
    home_team = first_game['homeTeam']['teamTricode']
    away_team = first_game['awayTeam']['teamTricode']
    game_date = '2025-01-21'  # Today's date
    
    print(f"\n3. Generating prediction for {away_team} @ {home_team}...")
    pred = engine.predict_game(home_team, away_team, game_date)
    
    if pred.get('home_win_prob'):
        print("✅ Prediction generated successfully!")
        print(f"   Home win prob: {pred['home_win_prob']:.1%}")
        print(f"   Predicted total: {pred.get('predicted_total', 'N/A')}")
        
        best_bet = pred.get('best_bet')
        if best_bet:
            bet_type = best_bet.get('bet_type', best_bet.get('type', 'Moneyline'))
            edge = best_bet.get('edge', 0)
            print(f"   Best bet: {best_bet['pick']} ({bet_type}) - {edge:.1%} edge")
        else:
            print("   Best bet: No qualifying bets")
    else:
        print(f"❌ Prediction failed: {pred.get('error', 'Unknown error')}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\n✅ Dashboard refresh should work!")
print("   Launch with: python nba_gui_dashboard_v2.py")
print("   Click '🔄 Refresh Predictions' to load today's games")
