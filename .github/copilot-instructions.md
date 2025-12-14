# NBA Betting System - Copilot Instructions

This is a comprehensive NBA betting prediction system with machine learning models, advanced statistical analysis, calibration tracking, and risk management.

## Project Architecture
- **Prediction Engine**: Hybrid model combining XGBoost ML with heuristic fallbacks
- **Advanced Models**: Poisson/Negative Binomial, Bayesian Hierarchical, Bivariate Spread-Total correlation
- **ELO System**: Separate offensive/defensive ELO ratings with injury-aware adjustments
- **Calibration**: Isotonic & Platt scaling with automated nightly refits and Brier score tracking
- **Risk Management**: Kelly criterion with drawdown scaling and calibration health factors
- **Feature Engineering**: 120+ features including recency-weighted stats, injury impacts, rest differentials
- **MLOps**: Model registry, versioning, monitoring, and deployment automation
- **Dashboard**: PyQt6 GUI with reliability curves, scenario simulation, and performance tracking

## Core Systems

### 1. Prediction Pipeline
- `prediction_engine.py` - Main prediction orchestration with calibration integration
- `feature_calculator_v5.py` - Feature computation with advanced off/def ELO, injury modeling
- `calibration_fitter.py` - Isotonic/Platt calibration with auto-refit scheduler
- `calibration_logger.py` - Prediction tracking for calibration database

### 2. ELO & Rating Systems
- `off_def_elo_system.py` - Separate offensive/defensive ELO with injury lag integration
- `dynamic_elo_calculator.py` - Legacy composite ELO with rest adjustments
- Injury-aware expected points calculation using replacement-level impact

### 3. Advanced Statistical Models
- `advanced_models.py` - PoissonTotalModel (Negative Binomial), BayesianHierarchicalModel
- `bivariate_model.py` - Correlated spread-total joint distribution modeling
- `scenario_simulator.py` - Monte Carlo simulation for uncertainty quantification

### 4. Risk & Position Sizing
- `kelly_optimizer.py` - Kelly criterion with calibration factor, drawdown scaling, event risk budget
- `calibration_metrics.py` - Brier score, reliability deciles, Wilson confidence intervals
- Drawdown-aware policy: >20% DD → 25% Kelly, >10% DD → 50% Kelly, >5% DD → 75% Kelly

### 5. Data Infrastructure
- `nba_stats_collector_v2.py` - NBA API data ingestion with rate limiting
- `injury_replacement_model.py` - Replacement-level impact using PIE scores
- `feature_cache.py` - Redis-backed caching for computed features
- SQLite databases: `nba_betting_data.db` for calibration, ELO history, bet tracking

### 6. MLOps & Monitoring
- `mlops_infrastructure.py` - ModelRegistry, PerformanceMonitor, ModelDeployment
- `ml_model_trainer.py` - Ensemble training (XGBoost, LightGBM, Random Forest) with time-series CV
- `scripts/retrain_pipeline.py` - Automated retraining with manifest checksum tracking
- `scripts/nightly_tasks.py` - Outcome ingestion, calibration updates, metric reporting

### 7. Dashboard & Visualization
- `NBA_Dashboard_Enhanced_v5.py` - Main PyQt6 interface
- Tabs: Calibration/Reliability, Scenarios, Model Health, Metrics, Risk, Advanced, Logs
- Real-time Brier trend plots, reliability curves with Wilson intervals
- Fair probability display after vig removal

## Development Guidelines

### Theoretical Soundness
- **Calibration is MANDATORY** - Never use raw XGBoost probabilities for Kelly sizing
- All probability predictions must pass through `CalibrationFitter.apply()`
- Track calibration health via Brier score and max decile gap
- Kelly sizing includes calibration factor: `kelly * calibration_factor * drawdown_scale`

### Engineering Standards
- Use `snake_case` for all variables/functions
- Extract magic numbers to `constants.py` with UPPER_SNAKE_CASE names
- Use dataclasses from `data_models.py` for structured data (GameInfo, PredictionResult, etc.)
- Add comprehensive type hints (use `from __future__ import annotations`)
- Use structured logging via `logger_setup.get_structured_adapter()`

### Risk Management Rules
- Minimum edge threshold: 3% (`MIN_EDGE_FOR_BET`)
- Maximum single bet: 5% of bankroll (`MAX_BET_PCT_OF_BANKROLL`)
- Quarter Kelly default (`KELLY_FRACTION_MULTIPLIER = 0.25`)
- Commission-adjusted edge calculation: `edge - KALSHI_BUY_COMMISSION` before Kelly

### Code Organization
- **Core Logic**: `prediction_engine.py`, `kelly_optimizer.py`, `calibration_fitter.py`
- **Data Models**: `data_models.py`, `interfaces.py`
- **Configuration**: `constants.py`, `config.json`
- **Services**: `services/` directory for modular components
- **Scripts**: `scripts/` for automation (retraining, nightly tasks)
- **Tests**: `tests/` for unit tests with pytest

### Key Workflows

**Making Predictions:**
1. Compute features via `FeatureCalculatorV5.calculate_game_features()`
2. Get raw ML probability from `PredictionEngine._model_total_prob()`
3. Apply calibration via `CalibrationFitter.apply(raw_prob)`
4. Calculate edge considering commission
5. Compute Kelly stake with calibration & drawdown factors
6. Log prediction to `CalibrationLogger` for future calibration updates

**Nightly Calibration Update:**
1. Ingest settled game outcomes via `scripts/nightly_tasks.py`
2. Trigger `CalibrationFitter.auto_refit_nightly()` if ≥250 samples
3. Fit both isotonic and Platt models, compute Brier improvement
4. Log Brier trend with model/calibration version metadata
5. Generate reliability plots and export metrics CSV

**Model Retraining:**
1. Run `scripts/retrain_pipeline.py` to extract → feature → train → calibrate → deploy
2. Use TimeSeriesSplit for temporal validation (prevent look-ahead bias)
3. Train ensembles: XGBoost (ATS), LightGBM (Moneyline), Random Forest (Totals)
4. Update `models/manifest.json` with SHA256 checksums, training date, data slice
5. Atomic deployment to avoid race conditions

## Tech Stack
- **Python**: 3.12+ (pinned in `requirements.txt`)
- **ML**: scikit-learn 1.5.1, xgboost 2.1.0, lightgbm 4.3.0
- **Calibration**: Isotonic regression, Platt scaling (logistic)
- **Data**: pandas 2.2.2, numpy 1.26.4, scipy 1.11.4
- **GUI**: PyQt6 6.7.1, matplotlib 3.9.0
- **NBA Data**: nba_api 1.3.0
- **Async/HTTP**: aiohttp 3.9.5, requests 2.32.3
- **MLOps**: mlflow 2.11.3
- **Database**: SQLite (calibration, ELO, bet history)
- **Optional**: PyMC (Bayesian models), arviz (MCMC diagnostics)

## Critical Reminders
- **ALWAYS** calibrate probabilities before Kelly sizing
- Use `constants.py` for all configuration values
- Follow time-series split for backtesting (sort by date, no shuffling)
- Apply vig removal before edge calculation: `fair_prob = remove_vig(yes_price, no_price)`
- Track all predictions in calibration database for continuous improvement
- Separate offensive/defensive ELO provides better signal than composite alone
- Injury impacts flow through ELO expected points calculation (injury-aware ratings)
- Drawdown scaling prevents over-betting during losing streaks

## Avoiding Common Mistakes
1. ❌ Using raw model probabilities → ✅ Always apply calibration
2. ❌ Static Kelly fraction → ✅ Adjust for calibration quality & drawdown
3. ❌ Ignoring commission → ✅ Subtract Kalshi fee from edge before sizing
4. ❌ Random train/test split → ✅ Time-series split only
5. ❌ Magic numbers in code → ✅ Named constants from constants.py
6. ❌ No type hints → ✅ Comprehensive typing with dataclasses
7. ❌ Single composite ELO → ✅ Separate off/def ELO for better prediction
8. ❌ Forgetting injury impact → ✅ Include replacement-level adjustments in ELO