"""
Test Refresh Functionality
"""
from nba_gui_dashboard_v2 import PREDICTION_ENGINE_AVAILABLE

print("="*80)
print("TESTING REFRESH FUNCTIONALITY")
print("="*80)

print(f"\n✅ Prediction engine available: {PREDICTION_ENGINE_AVAILABLE}")

if PREDICTION_ENGINE_AVAILABLE:
    from nba_api.live.nba.endpoints import scoreboard
    from datetime import datetime
    
    print("\n[1/2] Testing nba_api scoreboard...")
    try:
        today = scoreboard.ScoreBoard()
        games_data = today.get_dict()
        
        game_count = 0
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            game_count = len(games_data['scoreboard']['games'])
            
            print(f"✅ Found {game_count} games today")
            
            # Show first game as example
            if game_count > 0:
                game = games_data['scoreboard']['games'][0]
                home = game['homeTeam']['teamTricode']
                away = game['awayTeam']['teamTricode']
                print(f"   Example: {away} @ {home}")
        else:
            print("⚠️ No games today")
            
    except Exception as e:
        print(f"❌ NBA API error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[2/2] Testing prediction engine...")
    from nba_gui_dashboard_v2 import NBAPredictionEngine
    try:
        engine = NBAPredictionEngine()
        print("✅ Prediction engine initialized")
        print(f"   Model: {len(engine.features)} features")
        print(f"   Bankroll: ${engine.bankroll:,.0f}")
    except Exception as e:
        print(f"❌ Engine error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n❌ Prediction engine not available - check imports")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
if PREDICTION_ENGINE_AVAILABLE:
    print("✅ Dashboard refresh should work")
    print("   Launch: python nba_gui_dashboard_v2.py")
    print("   Click: 🔄 Refresh Predictions")
else:
    print("❌ Dashboard refresh will not work - fix imports first")
