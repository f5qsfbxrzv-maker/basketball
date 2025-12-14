# DATABASE ORGANIZATION - FINAL STRUCTURE

## PRIMARY DATABASE (IN USE)
✅ **nba_MAIN_database.db** (82 MB) - ROOT FOLDER
   - Main production database with ALL data
   - team_stats: 660 rows
   - game_results: 12,205 rows  
   - game_logs: 24,410 rows
   - elo_ratings: 24,411 rows
   - Plus 38 other populated tables
   - **Used by: NBA_Dashboard_v6_Streamlined.py and all core modules**

## SPECIALIZED DATABASES (KEPT)
📊 **nba_ODDS_history.db** (0.22 MB) - ROOT FOLDER
   - Historical odds tracking: 267 records
   
📊 **nba_PREDICTIONS_history.db** (0.55 MB) - ROOT FOLDER  
   - Historical predictions: 130 records

📊 **Sports_Betting_System/data/database/nba_HISTORICAL_odds_14k.db** (0.77 MB)
   - Legacy historical odds: 14,822 records
   
📊 **V2/data/nba_BANKROLL_tracker.db** (0.02 MB)
   - Bankroll tracking: 1 row

## ARCHIVED (NO LONGER IN USE)
🗑️ Moved to: archive/old_databases_20251204/
   - EMPTY_nba_betting_data.db (was in root - all tables empty)
   - OLD_results_40predictions.db (40 old predictions)
   - DUPLICATE_results.db (duplicate data)
   - OLD_odds_only.db (only had odds, no stats)

## CONFIGURATION CHANGES
Updated default database path in:
- ✅ NBA_Dashboard_v6_Streamlined.py
- ✅ core/kelly_optimizer_v6.py
- ✅ core/calibration_logger_v6.py  
- ✅ core/calibration_fitter_v6.py
- ✅ core/feature_calculator_v6.py

All modules now default to: **nba_MAIN_database.db**
