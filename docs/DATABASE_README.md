# NBA Betting System - Database Location

## Active Database
**Location:** `data/live/nba_betting_data.db`

This is the ONLY active database used by the system.

## What's In It
- **game_logs**: 24,832 games (2015-2025)
- **game_advanced_stats**: 24,832 games with Four Factors
- **active_injuries**: 109 current injuries (updated daily from ESPN)
- **player_stats**: 7,756 player records with PIE values
- **team_stats**: 660 team records (season averages)
- **elo_ratings**: 24,441 ELO history records
- **calibration_outcomes**: 500 tracked predictions
- **And 35+ other tables** for predictions, odds, betting history, etc.

## Date Range
- **Historical**: October 27, 2015 - April 13, 2025
- **Current Season**: October 22, 2025 - December 12, 2025

## Configuration
Set in `config/settings.py`:
```python
DB_PATH = DATA_DIR / "live" / "nba_betting_data.db"
```

## Archived Databases
Old/backup databases are stored in `data/backups/` with timestamps.

## Daily Updates Required
To keep rest days and features accurate:
```bash
python update_game_logs.py  # Downloads latest games
python build_game_advanced_stats.py  # Rebuilds Four Factors
```

Or use the dashboard's "Refresh Data" button.

## IMPORTANT
- ❌ Do NOT create `nba_betting_data.db` in the root directory
- ✅ Always use `data/live/nba_betting_data.db`
- ✅ All scripts should reference the config path
