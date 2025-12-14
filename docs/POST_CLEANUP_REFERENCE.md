# Quick Reference - Post-Cleanup

## System Status
✅ **Health Check:** 45/45 passing (100%)  
✅ **All components verified and functional**  
✅ **Production ready**

## File Locations (Current Versions)

### Production Code (Root)
- **Main System:** `main.py`
- **Dashboard:** `NBA_Dashboard_Enhanced_v5.py`
- **Stats Collector:** `nba_stats_collector_v2.py`
- **Injury Collector:** `injury_data_collector_v2.py`
- **Feature Calculator:** `feature_calculator_v5.py`
- **ELO Calculator:** `dynamic_elo_calculator.py`
- **ML Trainer:** `ml_model_trainer.py`
- **Health Check:** `health_check.py`

### Documentation (docs/)
- **Organization Guide:** `docs/FILE_STRUCTURE.md`
- **Cleanup Summary:** `docs/CLEANUP_RESULTS.md`
- **PACE Documentation:** `docs/GOLD_STANDARD_IMPLEMENTATION.md`
- **Feature Guide:** `docs/FEATURE_CALCULATOR_V5_GUIDE.md`
- **Quick Start:** `docs/QUICK_START.md`

### Tests (tests/)
- **PACE Verification:** `tests/verify_pace_calculation.py`
- **Feature Tests:** `tests/test_feature_calculator.py`
- **Schema Check:** `tests/check_schema.py`
- **DB Status:** `tests/check_db_status.py`

### Old Versions (archived_versions/)
❌ **DO NOT USE** - These are archived for reference only:
- `injury_data_collector.py` (use V2 instead)
- `nba_stats_collector_enhanced.py` (use V2 instead)
- `feature_calculator.py` (use V5 instead)
- `feature_calculator_enhanced.py` (use V5 instead)
- `NBA_Dashboard_Gold_Standard_v4_1.py` (use V5 instead)

## Common Commands

### Launch System
```bash
python main.py
```

### Run Health Check
```bash
python health_check.py
```

### Verify PACE Calculation
```bash
python tests/verify_pace_calculation.py
```

### Check Database Status
```bash
python tests/check_db_status.py
```

### Add New Season Data
```bash
python scripts/add_recent_seasons.py
```

### Download Complete Dataset
```bash
python data_downloads/download_gold_standard_data.py
```

## Import References

### Correct Imports (Use These)
```python
from nba_stats_collector_v2 import NBAStatsCollectorV2
from injury_data_collector_v2 import InjuryDataCollectorV2
from feature_calculator_v5 import FeatureCalculatorV5
from NBA_Dashboard_Enhanced_v5 import NBADashboard
```

### Incorrect Imports (Archived)
```python
# ❌ DON'T USE - These are archived
from injury_data_collector import InjuryDataCollector
from nba_stats_collector_enhanced import NBAStatsCollector
from feature_calculator import NBAFeatureCalculator
from feature_calculator_enhanced import FeatureCalculator
from NBA_Dashboard_Gold_Standard_v4_1 import NbaDashboardGoldStandard
```

## Database Info
- **Primary DB:** `nba_betting_data.db` (31,338 game logs)
- **Backup DB:** `backups/nba_betting_data_backup_old_schema.db`
- **PACE Coverage:** 100% (all 31,338 game logs)
- **Seasons:** 10 (2015-16 through 2024-25)
- **Active Injuries:** 103 tracked

## Troubleshooting

### Health Check Fails
```bash
python health_check.py
# Review errors and warnings
```

### Import Errors
- Ensure using V2/V5 versions (see "Correct Imports" above)
- Check `archived_versions/` folder for old files
- DO NOT import from archived_versions/

### Dashboard Won't Launch
1. Check PyQt6 is installed: `pip install PyQt6`
2. Verify health check passes: `python health_check.py`
3. Check main.py uses UTF-8 encoding fix

### Missing Files
- Check appropriate folder (docs/, tests/, scripts/, etc.)
- See `docs/FILE_STRUCTURE.md` for complete file listing
- Old versions in `archived_versions/` (reference only)

## Issues Fixed in Cleanup

✅ Removed 6 redundant file versions  
✅ Organized 50+ files into logical folders  
✅ Updated all imports to V2/V5  
✅ Fixed health_check.py to reference current files  
✅ Added UTF-8 encoding support for Windows  
✅ Created comprehensive documentation  

## System Ready!
All cleanup complete. System verified and production-ready.

**Launch with:** `python main.py`
