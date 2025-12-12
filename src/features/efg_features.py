"""Rolling and EWMA eFG% differentials.
Computes team eFG% per game and returns:
- efg_l10_diff: 10-game rolling eFG% (home - away)
- efg_ewma_diff: EWMA (span=12) eFG% (home - away)
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Dict
import pandas as pd

try:
    from v2.database_paths import NBA_BETTING_DB as DEFAULT_DB
except Exception:
    DEFAULT_DB = Path(__file__).parent / ".." / "data" / "nba_betting_data.db"


def _load_team_logs(conn: sqlite3.Connection, team: str, before_date: str) -> pd.DataFrame:
    q = (
        "SELECT GAME_DATE, FGM, FGA, FG3M FROM game_logs "
        "WHERE TEAM_ABBREVIATION = ? AND GAME_DATE < ? ORDER BY GAME_DATE"
    )
    df = pd.read_sql_query(q, conn, params=(team, before_date))
    if df.empty:
        return df
    df['FGA'] = df['FGA'].replace(0, pd.NA)
    df['efg'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
    df = df.dropna(subset=['efg'])
    return df


def compute_efg_diffs(home_team: str, away_team: str, game_date: str, db_path: str | Path = DEFAULT_DB) -> Dict[str, float]:
    db_path = str(db_path)
    try:
        conn = sqlite3.connect(db_path)
    except Exception:
        return {'efg_l10_diff': 0.0, 'efg_ewma_diff': 0.0}
    try:
        h = _load_team_logs(conn, home_team, game_date)
        a = _load_team_logs(conn, away_team, game_date)
    finally:
        conn.close()

    def rolling(df: pd.DataFrame, n: int) -> float:
        if df.empty:
            return 0.0
        return float(df['efg'].tail(n).mean())

    def ewma(df: pd.DataFrame, span: int = 12) -> float:
        if df.empty:
            return 0.0
        return float(df['efg'].ewm(span=span, adjust=False).mean().iloc[-1])

    return {
        'efg_l10_diff': rolling(h, 10) - rolling(a, 10),
        'efg_ewma_diff': ewma(h) - ewma(a),
    }


__all__ = ["compute_efg_diffs"]
