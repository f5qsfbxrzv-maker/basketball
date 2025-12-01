import sqlite3, json, logging, datetime

class CalibrationLogger:
    """Persist prediction probabilities and eventual outcomes for calibration.
    Schema supports later isotonic/Platt scaling and feature drift analysis.
    """
    VERSION = "v6"
    
    def __init__(self, db_path: str = "data/database/nba_betting_data.db"):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    """CREATE TABLE IF NOT EXISTS calibration_outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_date TEXT,
                        home_team TEXT,
                        away_team TEXT,
                        market_line REAL,
                        predicted_total REAL,
                        prob_over REAL,
                        actual_total REAL,
                        over_result INTEGER,
                        feature_snapshot TEXT,
                        logged_at TEXT,
                        model_version TEXT DEFAULT 'v6.0',
                        calibration_version TEXT
                    )"""
                )
                conn.commit()
        except Exception as e:
            logging.error(f"Calibration table create failed: {e}")

    def log_prediction(self, game_date: str, home: str, away: str, market_line: float,
                       predicted_total: float, prob_over: float, features: dict | None = None,
                       model_version: str = 'v6.0', calibration_version: str = 'none'):
        """Insert prediction row (without outcome) with version traceability."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    """INSERT INTO calibration_outcomes
                        (game_date, home_team, away_team, market_line, predicted_total, prob_over,
                         actual_total, over_result, feature_snapshot, logged_at, model_version, calibration_version)
                        VALUES (?,?,?,?,?,?,NULL,NULL,?,?,?,?)""",
                    (
                        game_date, home, away, market_line, predicted_total, prob_over,
                        json.dumps(features or {}), datetime.datetime.utcnow().isoformat(),
                        model_version, calibration_version
                    )
                )
                conn.commit()
        except Exception as e:
            logging.error(f"Calibration prediction log failed: {e}")

    def log_outcome(self, game_date: str, home: str, away: str, actual_total: float):
        """Update existing row with actual total + outcome flag.
        Outcome row selection by earliest prediction for that matchup/day lacking outcome.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    """SELECT id, market_line FROM calibration_outcomes
                        WHERE game_date=? AND home_team=? AND away_team=? AND actual_total IS NULL
                        ORDER BY id ASC LIMIT 1""",
                    (game_date, home, away)
                )
                row = c.fetchone()
                if not row:
                    return False
                rec_id, market_line = row
                over_result = 1 if actual_total > market_line else 0
                c.execute(
                    "UPDATE calibration_outcomes SET actual_total=?, over_result=? WHERE id=?",
                    (actual_total, over_result, rec_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Calibration outcome log failed: {e}")
        return False

    def fetch_unresolved(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute(
                    "SELECT * FROM calibration_outcomes WHERE actual_total IS NULL"
                ).fetchall()
        except Exception:
            return []

    def fetch_all(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute(
                    "SELECT * FROM calibration_outcomes"
                ).fetchall()
        except Exception:
            return []

# Backward compatibility alias
CalibrationLogger = CalibrationLogger
