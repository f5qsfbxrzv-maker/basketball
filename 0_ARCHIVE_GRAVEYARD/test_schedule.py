"""Get today's NBA schedule"""
from nba_schedule_downloader import NBAScheduleDownloader

dl = NBAScheduleDownloader()
games = dl.get_games_for_date('2025-12-12')

print(f"📅 Games today ({len(games)}):\n")
for g in games:
    print(f"{g['away_team']} @ {g['home_team']}")
