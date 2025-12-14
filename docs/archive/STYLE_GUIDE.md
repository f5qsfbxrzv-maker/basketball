# Basketball Model Style Guide

## Naming Conventions
- Modules & functions: snake_case (e.g. `train_models`, `update_manifest`).
- Classes: PascalCase (e.g. `FeatureBuilder`).
- Constants: UPPER_SNAKE_CASE (e.g. `DATA_DIR`).
- Private helpers: prefix with single underscore (e.g. `_load_config`).
- Filenames: concise, action or domain oriented (`retrain_pipeline.py`, `nightly_tasks.py`).

## Directory Layout
- `data/` raw & processed inputs.
- `models/` artifacts + `production/` and `staging/` subfolders + `manifest.json`.
- `scripts/` operational scripts (retrain, nightly tasks, ad-hoc utilities).
- `logs/` structured JSON / CSV / MD summaries.
- `notebooks/` (optional) exploratory analysis; outputs should be converted to scripts for production.

## Logging
- Use structured prints (prefix tags) or a logging wrapper: `[Retrain]`, `[Nightly]`, `[Predict]`.
- Levels: INFO for pipeline stage transitions; WARNING for recoverable issues; ERROR for fatal aborts.
- Avoid noisy debug unless gated by an env var `DEBUG=1`.

## Error Handling
- Fail fast in critical pipeline steps; raise exceptions rather than silently continuing.
- Use atomic writes (temp file rename) for manifest & model promotion.
- Wrap `__main__` entrypoints with a try/except capturing high-level errors.

## Reproducibility
- Pin all dependencies in `requirements.txt` (no `>=` for core libs).
- Record `dependencies_hash` in manifest entries.
- Include `data_slice`, `feature_version`, and training timestamps for provenance.

## Versioning
- Semantic Model Versioning: `major.minor.patch` stored within model_id or separate metadata (e.g. `basket_ats_model_v1.2.0`).
- Bump minor for feature engineering changes; patch for calibration-only updates; major for architecture shifts.
- Keep calibration-specific version if recalibration without retraining.

## Code Style
- Prefer functions over large inline scripts; keep each function <50 lines when feasible.
- Avoid deep nesting; early returns for validation failures.
- Explicit is better than implicit; no wildcard imports.
- Type hints for public functions & return values.

## Performance Considerations
- Batch data access; avoid per-row network calls.
- Cache expensive feature computations (e.g. ELO recalculations) where deterministic.
- Profile before optimizing; add timers around critical sections if they become bottlenecks.

## Testing (Future)
- Unit test pure feature transforms & calibration math.
- Smoke test pipeline (end-to-end) with a tiny synthetic dataset.
- Regression snapshots for key metrics (Brier, ROC AUC) to catch drift.

## Security & Safety
- Never embed API keys in code; use environment variables or a `.env` loaded securely.
- Validate external data before processing (schema, required columns).
- Include responsible gambling warnings in user-facing outputs where wagers are displayed.

## Documentation
- Docstrings: first line summary, blank line, argument descriptions.
- High-level pipeline README: stages, inputs, outputs, failure modes.
- Update this guide when introducing new architectural patterns.
