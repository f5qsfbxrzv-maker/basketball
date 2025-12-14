"""
Download missing data tables for live predictions
Populates: player_stats, game_logs, team_stats, game_advanced_stats
"""
import sys
sys.path.insert(0, 'src')

from src.services._OLD_nba_stats_collector_v2 import NBAStatsCollectorV2
from datetime import datetime

print("=" * 80)
print("DOWNLOADING MISSING NBA DATA")
print("=" * 80)

# Initialize collector
collector = NBAStatsCollectorV2(db_path='nba_betting_data.db')

# Download current season data
current_season = "2024-25"

print(f"\n[1/3] Downloading team stats for {current_season}...")
team_stats = collector.get_team_stats(season=current_season)
print(f"[OK] Downloaded stats for {len(team_stats)} teams")

print(f"\n[2/3] Downloading player impact stats for {current_season}...")
player_stats = collector.get_player_impact_stats(season=current_season)
print(f"[OK] Downloaded stats for {len(player_stats)} players")

print(f"\n[3/3] Downloading game logs for {current_season}...")
games = collector.get_game_logs(season=current_season)
print(f"[OK] Downloaded {len(games)} game logs")

print("\n" + "=" * 80)
print("[SUCCESS] DATA DOWNLOAD COMPLETE")
print("=" * 80)
print("\nTables created:")
print("  - team_stats (for feature calculations)")
print("  - player_stats (for injury impact PIE values)")
print("  - game_logs (for historical stats)")
print("\nYou can now launch the dashboard and make predictions!")
