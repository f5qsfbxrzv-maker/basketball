import sqlite3, logging, json
from typing import Optional, Tuple
from datetime import datetime, timedelta
from src.constants import (
    MIN_CALIBRATION_SAMPLES,
    CALIBRATION_REFIT_INTERVAL_DAYS,
    TARGET_BRIER_SCORE,
    MAX_BRIER_FOR_BETTING,
    ISOTONIC_MIN_SAMPLES
)

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    import numpy as np
except Exception as e:
    IsotonicRegression = None
    LogisticRegression = None
    np = None
    logging.warning(f"Calibration fitter limited: {e}")

class CalibrationFitter:
    """
    Probability calibration system for NBA betting predictions using Isotonic Regression and Platt Scaling.
    
    This class implements two calibration methods:
    1. **Isotonic Regression**: Non-parametric, monotonic calibration that learns arbitrary calibration curves
    2. **Platt Scaling**: Parametric logistic calibration that fits a sigmoid to map raw probabilities
    
    The calibration process:
    - Loads historical predictions and outcomes from `calibration_outcomes` table
    - Fits both isotonic and Platt models when sufficient samples (default 250) are available
    - Stores calibration parameters in `calibration_models` table for persistence
    - Automatically refits nightly based on `should_refit()` conditions
    - Tracks Brier score improvement (before/after calibration) for model selection
    
    **CRITICAL**: All betting probabilities MUST pass through `apply()` before Kelly sizing.
    Using uncalibrated probabilities will cause systematic position sizing errors.
    
    Attributes:
        db_path (str): Path to SQLite database containing calibration_outcomes and calibration_models tables
        min_samples (int): Minimum samples required for calibration fit (default: MIN_CALIBRATION_SAMPLES=250)
        model_version (str): Version string for base prediction model (update when retraining occurs)
    
    Examples:
        >>> fitter = CalibrationFitter(db_path='data/database/data/database/nba_betting_data.db', min_samples=250)
        >>> # Fit both calibration methods
        >>> iso_result = fitter.fit_isotonic()
        >>> platt_result = fitter.fit_platt()
        >>> # Apply calibration to raw probability
        >>> raw_prob = 0.65
        >>> calibrated_prob = fitter.apply(raw_prob)  # Returns calibrated probability
        >>> # Auto-refit nightly
        >>> result = fitter.auto_refit_nightly()
    
    See Also:
        - `calibration_metrics.py`: Brier score computation and reliability diagnostics
        - `kelly_optimizer.py`: Uses calibrated probabilities for bet sizing
        - `constants.py`: MIN_CALIBRATION_SAMPLES, MAX_BRIER_FOR_BETTING thresholds
    """
    def __init__(self, db_path: str = 'data/database/data/database/nba_betting_data.db', min_samples: int = MIN_CALIBRATION_SAMPLES):
        """
        Initialize calibration fitter with database connection and minimum sample threshold.
        
        Args:
            db_path (str): Path to SQLite database. Default: 'data/database/data/database/nba_betting_data.db'
            min_samples (int): Minimum predictions required to fit calibration. Default: MIN_CALIBRATION_SAMPLES (250)
        
        Raises:
            sqlite3.Error: If database connection or table creation fails
        """
        self.db_path = db_path
        self.min_samples = min_samples
        self.model_version = "v6.0"  # Update when base model changes
        self.VERSION = "v6"
        self._ensure_tables()

    def _ensure_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS calibration_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                fitted_at TEXT,
                params TEXT,
                model_version TEXT DEFAULT 'v5.0',
                calibration_version TEXT,
                brier_before REAL,
                brier_after REAL,
                sample_count INTEGER
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS calibration_schedule (
                id INTEGER PRIMARY KEY,
                last_refit_date TEXT
            )""")
            conn.commit()

    def _load_calibration_data(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""SELECT prob_over, over_result FROM calibration_outcomes
                                   WHERE over_result IS NOT NULL""").fetchall()
        if not rows:
            return None, None
        probs = [r[0] for r in rows]
        outcomes = [r[1] for r in rows]
        return probs, outcomes

    def can_fit(self) -> bool:
        probs, outcomes = self._load_calibration_data()
        return probs is not None and len(probs) >= self.min_samples

    def fit_isotonic(self) -> Optional[Tuple[str, dict, str]]:
        if not IsotonicRegression or not np:
            return None
        probs, outcomes = self._load_calibration_data()
        if probs is None or len(probs) < self.min_samples:
            return None
        # Compute Brier before calibration
        brier_before = np.mean((np.array(probs) - np.array(outcomes)) ** 2)
        ir = IsotonicRegression(out_of_bounds='clip')
        y_ = ir.fit_transform(probs, outcomes)
        # Compute Brier after
        brier_after = np.mean((y_ - np.array(outcomes)) ** 2)
        # Store thresholds and predictions
        params = {
            'thresholds': ir.fitted_.tolist() if hasattr(ir.fitted_, 'tolist') else list(ir.fitted_),
            'increasing': bool(ir.increasing),
            'y_': list(map(float, y_))
        }
        cal_version = self._persist_model('isotonic', params, brier_before, brier_after, len(probs))
        return 'isotonic', params, cal_version

    def fit_platt(self) -> Optional[Tuple[str, dict, str]]:
        if not LogisticRegression or not np:
            return None
        probs, outcomes = self._load_calibration_data()
        if probs is None or len(probs) < self.min_samples:
            return None
        X = np.array(probs).reshape(-1,1)
        brier_before = np.mean((np.array(probs) - np.array(outcomes)) ** 2)
        lr = LogisticRegression()
        lr.fit(X, outcomes)
        calibrated = lr.predict_proba(X)[:, 1]
        brier_after = np.mean((calibrated - np.array(outcomes)) ** 2)
        params = {'coef': lr.coef_.tolist(), 'intercept': lr.intercept_.tolist()}
        cal_version = self._persist_model('platt', params, brier_before, brier_after, len(probs))
        return 'platt', params, cal_version

    def _persist_model(self, model_type: str, params: dict, brier_before: float = 0.0, brier_after: float = 0.0, sample_count: int = 0):
        import datetime, json
        cal_version = f"{model_type}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""INSERT INTO calibration_models 
                         (model_type, fitted_at, params, model_version, calibration_version, brier_before, brier_after, sample_count) 
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (model_type, datetime.datetime.utcnow().isoformat(), json.dumps(params), 
                       self.model_version, cal_version, brier_before, brier_after, sample_count))
            conn.commit()
        return cal_version

    def load_latest(self) -> Optional[Tuple[str, dict, str]]:
        """Returns (model_type, params, calibration_version)"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            row = c.execute("SELECT model_type, params, calibration_version FROM calibration_models ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        return row[0], json.loads(row[1]), row[2] if row[2] else 'unknown'
    
    def should_refit(self) -> bool:
        """Check if conditions met for nightly refit: min samples + hasn't run today."""
        if not self.can_fit():
            return False
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            row = c.execute("SELECT last_refit_date FROM calibration_schedule WHERE id=1").fetchone()
        if not row:
            return True
        last_date = datetime.fromisoformat(row[0]).date()
        return datetime.utcnow().date() > last_date
    
    def auto_refit_nightly(self) -> dict:
        """Run both isotonic and Platt refit if conditions met. Returns summary dict."""
        if not self.should_refit():
            return {'status': 'skipped', 'reason': 'already_refitted_today_or_insufficient_samples'}
        
        results = {'status': 'completed', 'isotonic': None, 'platt': None}
        
        iso_result = self.fit_isotonic()
        if iso_result:
            results['isotonic'] = {'version': iso_result[2], 'brier_improvement': 'tracked'}
        
        platt_result = self.fit_platt()
        if platt_result:
            results['platt'] = {'version': platt_result[2], 'brier_improvement': 'tracked'}
        
        # Update schedule
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO calibration_schedule (id, last_refit_date) VALUES (1, ?)",
                      (datetime.utcnow().isoformat(),))
            conn.commit()
        
        # Log Brier trend
        from src.core.calibration_metrics import compute_brier_score, log_brier_trend
        brier = compute_brier_score(self.db_path)
        latest = self.load_latest()
        cal_ver = latest[2] if latest else 'none'
        log_brier_trend(self.db_path, brier, self.model_version, cal_ver)
        
        return results

    def is_ready(self) -> bool:
        """
        Check if calibration model is fitted and ready to use.
        
        Returns:
            bool: True if calibration model exists and can be applied, False otherwise
        """
        latest = self.load_latest()
        return latest is not None

    def apply(self, prob_over: float) -> float:
        latest = self.load_latest()
        if not latest:
            return prob_over
        model_type, params, cal_version = latest
        if model_type == 'platt' and 'coef' in params and 'intercept' in params and np:
            coef = np.array(params['coef'])[0]
            intercept = np.array(params['intercept'])[0]
            logit = coef * prob_over + intercept
            import math
            calibrated = 1/(1+math.exp(-logit))
            return float(max(0.001, min(0.999, calibrated)))
        if model_type == 'isotonic' and 'y_' in params and 'thresholds' in params and np:
            # Reconstruct isotonic mapping using stored fitted values
            thresholds = np.array(params['thresholds'])
            y_values = np.array(params['y_'])
            # Use numpy interp for piecewise linear interpolation
            calibrated = float(np.interp(prob_over, thresholds, y_values))
            return float(max(0.001, min(0.999, calibrated)))
        return prob_over

# Backward compatibility alias
CalibrationFitter = CalibrationFitter
