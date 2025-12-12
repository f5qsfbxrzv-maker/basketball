"""Compute player-level expected value (EV) / player 'ELO' for injury valuation.

Provides functions to compute a per-player numeric value (net-rating scale) using
available player impact stats (`player_stats`) and league baselines. The value is
meant to be used by the injury pipeline to assign more weight to stars.

Algorithm (summary):
- Baseline = mean(PIE) * 100 (maps PIE to net-rating-ish units)
- Usage multiplier: rewards higher usage players relative to league average
- Starter/star multiplier: boosts star-caliber players (by PIE) and regular starters
- Positional scarcity multiplier: from `injury_replacement_model.POSITION_SCARCITY`

Functions:
- compute_league_avgs(db_path, season=None)
- compute_player_value(db_path, player_name, season=None)
- compute_all_player_values(db_path, season=None) -> writes `player_values` table
"""
from typing import Optional, Dict, Tuple
import sqlite3
from statistics import mean

from src.features.injury_replacement_model import POSITION_SCARCITY
from config.constants import STAR_PLAYER_PIE_THRESHOLD


def compute_league_avgs(db_path: str, season: Optional[str] = None) -> Dict[str, float]:
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()
    if season:
        cur.execute("SELECT AVG(pie), AVG(usg_pct) FROM player_stats WHERE season = ?", (season,))
    else:
        cur.execute("SELECT AVG(pie), AVG(usg_pct) FROM player_stats")
    row = cur.fetchone() or (0.0, 0.0)
    conn.close()
    avg_pie = row[0] or 0.0
    avg_usg = (row[1] or 0.0) / 100.0
    return {'avg_pie': avg_pie, 'avg_usg': avg_usg}


def compute_player_value(db_path: str, player_name: str, season: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()
    if season:
        cur.execute('SELECT pie, usg_pct, team_abbreviation FROM player_stats WHERE player_name = ? AND season = ? ORDER BY season DESC LIMIT 1', (player_name, season))
    else:
        cur.execute('SELECT pie, usg_pct, team_abbreviation FROM player_stats WHERE player_name = ? ORDER BY season DESC LIMIT 1', (player_name,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return 0.0, {'reason': 'no_player_stats'}

    pie, usg_pct, team_abbr = row
    pie = pie or 0.0
    usg = (usg_pct or 0.0) / 100.0

    # league averages
    avgs = compute_league_avgs(db_path, season)
    league_usg = max(avgs.get('avg_usg', 0.12), 0.01)

    # baseline: map PIE to net-rating-ish units
    baseline = pie * 100.0

    # usage multiplier: relative to league (soft scaling)
    usage_delta = (usg - league_usg) / league_usg
    usage_mult = 1.0 + 0.5 * usage_delta
    usage_mult = max(0.6, min(1.6, usage_mult))

    # starter/star multiplier
    if pie >= STAR_PLAYER_PIE_THRESHOLD:
        starter_mult = 1.35
    else:
        starter_mult = 1.05 if usg >= 0.12 else 0.85

    # positional scarcity fallback: unknown position assumed 'G'
    # try to find position in player_stats if present
    cur.execute('PRAGMA table_info(player_stats)')
    cols = [r[1] for r in cur.fetchall()]
    position = 'G'
    if 'position' in cols:
        cur.execute('SELECT position FROM player_stats WHERE player_name = ? ORDER BY season DESC LIMIT 1', (player_name,))
        pos_row = cur.fetchone()
        if pos_row and pos_row[0]:
            position = pos_row[0]

    scarcity = POSITION_SCARCITY.get(position, 1.0)

    value = baseline * usage_mult * starter_mult * scarcity

    conn.close()
    meta = {'pie': pie, 'usg': usg, 'baseline': baseline, 'usage_mult': usage_mult, 'starter_mult': starter_mult, 'scarcity': scarcity}
    return value, meta


def compute_all_player_values(db_path: str, season: Optional[str] = None) -> int:
    """Compute values for all players in player_stats and persist to `player_values` table.

    Returns number of rows written.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()
    # create table if missing
    cur.execute('''
        CREATE TABLE IF NOT EXISTS player_values (
            player_name TEXT,
            team_abbreviation TEXT,
            season TEXT,
            player_value REAL,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

    if season:
        cur.execute('SELECT DISTINCT player_name, team_abbreviation FROM player_stats WHERE season = ?', (season,))
    else:
        cur.execute('SELECT DISTINCT player_name, team_abbreviation, season FROM player_stats')
    rows = cur.fetchall()

    # normalize iteration depending on select
    players = []
    if rows and len(rows[0]) == 3:
        # had season in select
        players = [(r[0], r[1], r[2]) for r in rows]
    else:
        players = [(r[0], r[1], season) for r in rows]

    written = 0
    for player_name, team_abbr, s in players:
        val, meta = compute_player_value(db_path, player_name, s)
        cur.execute('INSERT INTO player_values (player_name, team_abbreviation, season, player_value) VALUES (?, ?, ?, ?)', (player_name, team_abbr, s, val))
        written += 1

    conn.commit()
    conn.close()
    return written


if __name__ == '__main__':
    import os, sys
    db = os.path.normpath(sys.argv[1]) if len(sys.argv) > 1 else 'v2/data/nba_betting_data.db'
    count = compute_all_player_values(db)
    print(f'Wrote {count} player values to player_values table in {db}')
