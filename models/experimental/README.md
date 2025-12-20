# Experimental Models Directory

This directory contains experimental model variations and archived models.

## Safety Protocol
- **NEVER overwrite production models** in the parent `models/` directory
- All experimental training should output here first
- Only promote to production after thorough validation

## Current Contents

### Archived Models
- `xgboost_final_trial98_REFERENCE_43features.json` - Previous 43-feature production model (archived Dec 19, 2025)

### Trial 1306 Baseline (for testing)
- `xgboost_22features_trial1306_baseline.json` - Copy of current production model
- `trial1306_params_baseline.json` - Hyperparameters
- `trial1306_results_baseline.json` - Training metrics
- `trial1306_config_baseline.py` - Configuration
- **Performance**: 49.7% ROI, 67.69% accuracy, 0.6330 log loss

## Active Experiments
(Document your experiments here as you create them)

### Feature Pruning Experiments
Testing removal of collinear features based on VIF analysis (Dec 19, 2025)
- Target: Reduce from 22 to ~17 features
- Issues to address:
  - ewma_orb_diff + projected_possession_margin (r=0.89, VIF=999)
  - ewma_tov_diff (VIF=999)
  - off_elo_diff + composite ELO redundancy (VIF=95)
  - Foul feature group overlap (VIF=7.4)
