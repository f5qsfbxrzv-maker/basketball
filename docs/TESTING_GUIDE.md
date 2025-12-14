# NBA Betting System - Test Suite Guide

## Quick Start

### Run All Tests
```powershell
python run_all_tests.py
```

### Run Specific Test Suites
```powershell
# Unit tests only
pytest tests/test_unit_features.py -v

# Integration tests
pytest tests/test_integration_pipeline.py -v

# Regression tests
pytest tests/test_regression_snapshots.py -v

# Performance benchmarks
pytest tests/test_performance.py -v -m performance

# Continuous retrain tests
pytest tests/test_continuous_retrain.py -v
```

### Coverage Reports
```powershell
# Generate HTML coverage report
pytest --cov=feature_calculator_v5 --cov=prediction_engine --cov-report=html

# View coverage in browser
start tests/coverage_html/index.html

# Terminal coverage summary
pytest --cov-report=term-missing
```

## Test Categories

### Unit Tests (`test_unit_features.py`)
Tests individual feature calculation functions in isolation:

- **Four Factors Differential**
  - `test_four_factors_balanced_matchup`: Evenly matched teams
  - `test_four_factors_home_advantage`: Home team has clear advantage
  - `test_four_factors_magnitude`: Realistic bounds checking
  - `test_four_factors_identity_comparison_logic`: Validates (H_off - H_def) - (A_off - A_def)
  - `test_four_factors_missing_stats`: Graceful degradation

- **Injury Impact**
  - `test_injury_impact_no_injuries`: Empty injury table
  - `test_injury_impact_differential_signs`: Correct sign (home injuries = negative differential)
  - `test_injury_status_weighting_logic`: OUT=1.0, DOUBTFUL=0.75, etc.
  - `test_injury_pie_calculation_bounds`: PIE-weighted impact within ±50

- **Pace Calculation**
  - `test_pace_features_average`: Average team pace
  - `test_pace_features_fast_vs_slow`: Opposing styles
  - `test_pace_features_extremes`: Very fast vs very slow teams
  - `test_pace_missing_fallback`: Default to ~100 when data missing

- **Rest Differential**
  - `test_rest_differential_keys`: Returns expected keys
  - Tests rest days and back-to-back differentials

- **Feature Consistency**
  - `test_empty_features_structure`: All required keys present
  - `test_feature_types`: All features are numeric

### Integration Tests (`test_integration_pipeline.py`)
End-to-end prediction flow:

- `test_feature_generation_to_prediction`: Features → raw probability
- `test_calibration_pipeline`: Raw prob → calibrated prob
- `test_end_to_end_with_market_line`: Complete flow with market line
- `test_deterministic_features`: Same inputs → same outputs
- `test_symmetry_reversal`: Home/away reversal negates differentials
- `test_no_nan_values`: No NaN in feature outputs
- `test_feature_ranges_realistic`: All features within NBA ranges
- `test_probability_bounds`: Predictions between 0-1

### Regression Tests (`test_regression_snapshots.py`)
Snapshot testing with frozen random seed:

- `test_feature_snapshot_consistency`: Features stable across runs
- `test_prediction_snapshot_consistency`: Predictions stable with same seed
- `test_batch_predictions_stable`: Batch processing deterministic
- `test_four_factors_deterministic`: Four factors calculation stable
- `test_pace_calculation_stable`: Pace calculation doesn't vary

### Continuous Retrain Tests (`test_continuous_retrain.py`)
Dataset update and retraining:

- `test_continuous_retrain_succeeds_with_new_data`: Calibration refit with new data
- Tests auto-refit nightly schedule logic
- Validates min sample requirements

### Performance Tests (`test_performance.py`)
Benchmarking and profiling:

- `test_feature_calculation_latency`: <10ms per game
- `test_batch_feature_generation_throughput`: >100 games/second
- `test_prediction_latency`: <5ms per prediction
- `test_end_to_end_pipeline_performance`: Complete flow <20ms
- Memory profiling for large batch operations

## Coverage Requirements

### Minimum Thresholds (70%)
Core modules must maintain 70% code coverage:
- `feature_calculator_v5.py`
- `prediction_engine.py`
- `calibration_fitter.py`
- `calibration_logger.py`

### Coverage Reports
- **HTML**: `tests/coverage_html/index.html`
- **Terminal**: Shows missing lines
- **JSON**: `tests/coverage.json` for CI/CD

## Test Markers

Use markers to run specific test categories:

```powershell
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Performance benchmarks
pytest -m performance

# Slow tests
pytest -m slow

# Tests requiring database
pytest -m requires_db

# Tests requiring trained models
pytest -m requires_models
```

## Common Issues

### Tests Run Slowly
- Feature calculator loads entire database into memory on initialization
- Use `pytest-xdist` for parallel execution: `pytest -n auto`
- Mock database for faster unit tests

### Import Errors
- Ensure all dependencies installed: `pip install -r test_requirements.txt`
- Activate virtual environment first

### Coverage Below 70%
- Check which lines are missing: `pytest --cov-report=term-missing`
- Add tests for uncovered code paths
- Consider edge cases and error handling

## Best Practices

1. **Run unit tests frequently** during development
2. **Run full suite before commits** to catch regressions
3. **Update snapshots** when intentionally changing model logic
4. **Monitor coverage trends** to maintain code quality
5. **Profile performance** after adding new features

## Continuous Integration

Add to your CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run tests with coverage
  run: |
    pytest --cov --cov-report=xml --cov-fail-under=70
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./tests/coverage.xml
```

## Troubleshooting

### ModuleNotFoundError
```powershell
# Install missing dependencies
pip install -r test_requirements.txt
```

### Database Errors
```powershell
# Ensure database exists
python data_downloads/download_gold_standard_data.py
```

### Model Not Found
```powershell
# Train models first
python ml_model_trainer.py
```

## Contact

For test failures or questions:
1. Check test output for specific failure messages
2. Review relevant module documentation
3. Validate data integrity with `validate_tests.py`
