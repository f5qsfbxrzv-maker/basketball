# MIGRATION GUIDE
Generated: 2025-11-30 19:25:46

## Phase 1: Quarantine (DO THIS FIRST)
1. Create `_OLD_CHAOS/` folder OUTSIDE Sports_Betting_System
2. Move EVERYTHING from current project into `_OLD_CHAOS/`
3. Start with clean Sports_Betting_System folder

## Phase 2: Data Migration
### Raw Data (Move to data/raw/)
- `V2/data/raw_odds_ehallmar/moneyline_history_2023_24.csv`
- `V2/data/raw_odds_ehallmar/nba_moneyline_2024_25.csv`
- `V2/data/raw_odds_ehallmar/nba_betting_spread.csv`
- `V2/data/raw_odds_ehallmar/nba_betting_totals.csv`

### Processed Data (Move to data/processed/)
- `V2/training_data/training_data_final.csv` ✅ CANONICAL (85 features, 12,188 rows)

### Database (Move to data/database/)
- All `*.db` files (calibration history, ELO data, bet tracking)

## Phase 3: Golden Master Selection

### Ingestion (src/ingestion/)
**WINNER**: `V2/v2/services/nba_stats_collector_v2.py`
- Handles NBA API rate limiting
- Error recovery
- Data validation

### Processing (src/processing/)
**WINNER**: `V2/v2/features/feature_calculator_v5.py`
- Full 85-feature computation
- Injury-aware ELO
- EWMA, chaos, pace, sharp signals

### Training (src/training/)
**WINNER**: `V2/v2/models/ml_model_trainer.py`
- Ensemble training (XGBoost, LightGBM, RF)
- TimeSeriesSplit validation
- Automated model registry

### Backtesting (src/backtesting/)
**WINNER**: Create new `backtest_pipeline.py` combining:
- Kelly criterion from `V2/v2/core/kelly_optimizer.py`
- Calibration from `V2/v2/core/calibration_fitter.py`

### Prediction (src/prediction/)
**WINNER**: `V2/v2/core/prediction_engine.py`
- Full prediction pipeline
- Calibrated probabilities
- Kelly bet sizing

## Phase 4: Models

### Production (models/production/)
- `V2/models/trained/moneyline_xgb_heavy_final.joblib` (if trained on clean data)
- `V2/models/tuned/best_params_moneyline_heavy.json`

### Archive (models/archive/)
- Any old model files from V2/models/

## Phase 5: Core Code Migration (src/)

### KEEP (Move to src/):
- `V2/v2/core/` → `src/core/`
- `V2/v2/features/` → `src/features/`
- `V2/v2/models/` → `src/models/`
- `V2/v2/services/` → `src/services/`

### DELETE (Leave in _OLD_CHAOS/):
- `run_pipeline_locally.py` (creates LEAKY data)
- `fetch_and_build_modern.py` (creates LEAKY data)
- All `*_v2.py`, `*_v3.py` versioned files
- All `check_*.py`, `debug_*.py` (move to notebooks/archive/)

## Phase 6: Path Updates

After migration, update imports:

```python
# OLD (WRONG)
from V2.v2.core.prediction_engine import PredictionEngine

# NEW (CORRECT)
from src.core.prediction_engine import PredictionEngine
```

Update data paths:
```python
# OLD (WRONG)
df = pd.read_csv('V2/data/training_data_final_modern.csv')  # LEAKY!

# NEW (CORRECT)
df = pd.read_csv('data/processed/training_data_final.csv')
```

## FILES TO NEVER TOUCH AGAIN
- `data/raw/` - Original CSVs (read-only)
- `models/production/` - Only deploy after backtesting
- `config.py` - Single source of truth for paths

## Git Setup (CRITICAL)
```bash
cd Sports_Betting_System
git init
git add .
git commit -m "Initial professional structure"
```

Now you have ONE version of each file. Use Git branches for experiments:
```bash
git checkout -b experiment/new-feature
# Make changes...
git commit -m "Test new feature"
# If it works:
git checkout main
git merge experiment/new-feature
# If it fails:
git checkout main  # Changes are discarded
```

NO MORE `model_v2_final_REAL.py` NONSENSE!
