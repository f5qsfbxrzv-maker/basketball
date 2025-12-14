"""
Test the new schedule system and paper trading tracker
"""
from schedule_service import ScheduleService
from paper_trading_tracker import PaperTradingTracker
from datetime import datetime

print("="*70)
print("TESTING SCHEDULE SERVICE")
print("="*70)

# Test schedule service
schedule = ScheduleService()

# Fetch upcoming games
print("\nFetching upcoming games from The Odds API...")
games = schedule.fetch_upcoming_games()

if games:
    print(f"\n✅ Found {len(games)} upcoming games")
    
    # Show games by date
    games_by_date = {}
    for game in games:
        date = game['game_date']
        if date not in games_by_date:
            games_by_date[date] = []
        games_by_date[date].append(game)
    
    for date in sorted(games_by_date.keys()):
        print(f"\n{date}:")
        for game in games_by_date[date]:
            print(f"  {game['away_team']} @ {game['home_team']} - {game['game_time']}")
    
    # Test getting today's games
    today = datetime.now().strftime('%Y-%m-%d')
    today_games = schedule.get_games_for_date(today)
    print(f"\n✅ Games today ({today}): {len(today_games)}")
    
    # Test getting next 3 days
    future_games = schedule.get_games_range(days_ahead=3)
    print(f"✅ Games in next 3 days: {len(future_games)}")

else:
    print("❌ No games found - check API key or NBA season schedule")

print("\n" + "="*70)
print("TESTING PAPER TRADING TRACKER")
print("="*70)

# Test paper trading tracker
tracker = PaperTradingTracker()

# Generate performance report
report = tracker.generate_performance_report()

print(f"\nTotal Bets: {report['total_bets']}")
print(f"Win Rate: {report['win_rate']:.1%}")
print(f"ROI: {report['roi']:.1%}")
print(f"Total Profit: ${report['total_profit']:.2f}")
print(f"Average Stake: ${report['avg_stake']:.2f}")
print(f"Brier Score: {report['brier_score']:.4f}")

if report['edge_buckets']:
    print("\nEdge Buckets:")
    for bucket in report['edge_buckets']:
        print(f"  {bucket['range']}: {bucket['count']} bets, {bucket['wins']} wins, ROI: {bucket['roi']:.1%}")
else:
    print("\nNo historical bets to analyze yet")

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETE")
print("="*70)
