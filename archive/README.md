# Archive Directory - December 12, 2025

This directory contains superseded files from previous model iterations.

## Archive Structure

### old_models/
Old XGBoost models and calibrators:
- `xgboost_final.pkl` - Previous "final" model (pre-temporal features)
- `xgboost_optuna_*.pkl` - Earlier Optuna tuning attempts
- `xgboost_pruned_31features.pkl` - 31-feature pruned model
- `xgboost_tuned.pkl` - Initial tuning results
- `xgboost_with_injury_shock.pkl` - Experimental injury feature model
- `*constrained*.json` - Constrained hyperparameter attempts (rejected -2.20% AUC)
- `manifest*.json` - Old model registries

### old_data/
Superseded training datasets:
- `training_data_with_features.csv` - Original 36-feature "dirty" dataset (0.5508 AUC)
  - Issues: 286-day rest bug, 10.7% ELO inflation, perfect correlations
- `training_data_with_features_cleaned.csv` - Clean 36-feature dataset (0.5407 AUC)
  - Fixed bugs but lost hidden temporal signals

### old_scripts/
Obsolete tuning/testing scripts:
- `tune_clean_constrained.py` - 300-trial constrained tuning (abandoned)
- `preview_constrained_performance.py` - Single fold constrained test
- `compare_constrained_unconstrained_5fold.py` - Full 5-fold comparison script (never run)
- `single_fold_quick_test.py` - Quick comparison confirming constraints hurt -2.20%

## Current Production Files (NOT archived)

### models/
- `single_fold_best_params.json` - Trial 98 best hyperparameters (0.66407 AUC on Fold 5)
- `xgboost_final_trial98.json` - Final XGBoost model (pending)
- `isotonic_calibrator_final.pkl` - Final isotonic calibrator (pending)
- `final_model_metadata.json` - Complete training metadata (pending)

### data/
- `training_data_with_temporal_features.csv` - Production dataset (43 features, 12,205 games)
  - Clean 36 features + 7 explicit temporal features
  - Baseline: 0.5570 AUC uncalibrated

### scripts/
- `single_fold_tuning.py` - Trial 98 discovery script
- `train_and_calibrate_final.py` - Final 5-fold training + isotonic calibration
- `add_temporal_features.py` - Temporal feature engineering
- `test_temporal_features.py` - 43-feature validation script

## Timeline

1. **Nov 2025**: Original 36-feature model (dirty data, 0.5508 AUC)
2. **Dec 2025**: Data cleaning → performance drop to 0.5407 AUC
3. **Dec 12, 2025**: Added 7 temporal features → 0.5570 AUC baseline
4. **Dec 12, 2025**: Tested constraints → hurt -2.20%, rejected
5. **Dec 12, 2025**: Single fold tuning (100 trials) → Trial 98 best (0.66407 AUC)
6. **Dec 12, 2025**: Final 5-fold training + isotonic calibration (CURRENT)

## Key Learnings

- Data "bugs" contained hidden temporal signals (season openers, league evolution)
- Explicit temporal features (season_year, season_progress) recovered lost performance
- Monotonic constraints hurt NBA predictions (-2.20%) due to complex nonlinear relationships
- Clean data + aggressive tuning beats constrained models
- Slower learning (lr=0.008) + shallow trees (depth=4) + strong regularization optimal
