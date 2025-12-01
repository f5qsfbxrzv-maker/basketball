"""
Replacement-Level Injury Impact Model
Extends basic PIE-weighted injury impact with position-specific replacement values.
Accounts for positional scarcity and games missed weighting.
"""

import sqlite3
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta

from v2.constants import (
    STATUS_PLAY_PROBABILITIES,
    CHEMISTRY_LAG_FACTOR,
    MAX_LAG_ABSENCES,
    STAR_PLAYER_PIE_THRESHOLD,
    POSITION_SCARCITY_OVERRIDES,
)


# Position replacement PIE baselines (league average by position)
POSITION_REPLACEMENT_PIE = {
    'PG': 0.095,   # Point Guard
    'SG': 0.088,   # Shooting Guard
    'SF': 0.092,   # Small Forward
    'PF': 0.090,   # Power Forward
    'C': 0.093,    # Center
    'G': 0.091,    # Generic Guard
    'F': 0.091,    # Generic Forward
    'F-C': 0.091,  # Forward-Center
    'G-F': 0.090,  # Guard-Forward
}

# Positional scarcity multipliers (higher = more scarce)
POSITION_SCARCITY = {
    'PG': 1.15,  # Elite PGs hard to replace
    'C': 1.12,   # Two-way centers scarce
    'SG': 1.0,
    'SF': 1.0,
    'PF': 1.05,
    'G': 1.0,
    'F': 1.0,
    'F-C': 1.08,
    'G-F': 1.0,
}


def get_player_position(player_name: str, team: str, db_path: str) -> str:
    """Fetch player position from player_stats table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Try season-aware lookup in player_season_metrics if available (best-effort)
    try:
        cur.execute('''
            SELECT position FROM player_season_metrics
            WHERE player_name = ? AND team = ?
            ORDER BY season DESC
            LIMIT 1
        ''', (player_name, team))
        row = cur.fetchone()
        if row and row[0]:
            conn.close()
            return row[0]
    except Exception:
        # Table may not exist; fall back to player_stats
        pass

    cur.execute('''
        SELECT position FROM player_stats
        WHERE player_name = ? AND team = ?
        ORDER BY season DESC
        LIMIT 1
    ''', (player_name, team))

    result = cur.fetchone()
    conn.close()
    return result[0] if result else 'G'  # Default to Guard


def _estimate_consecutive_absences(player_name: str, team: str, game_date: str, db_path: str) -> int:
    """Estimate consecutive absences by counting prior sequential 'Out' statuses.
    Falls back to 0 if insufficient history.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute('''
            SELECT game_date FROM active_injuries
            WHERE player_name = ? AND team = ? AND status = 'Out' AND game_date < ?
            ORDER BY game_date DESC
            LIMIT ?
        ''', (player_name, team, game_date, MAX_LAG_ABSENCES))
        rows = cur.fetchall()
    except Exception:
        rows = []
    conn.close()
    if not rows:
        return 0
    # Count consecutive days (tolerant to schedule gaps); simple length proxy
    return len(rows)


def calculate_replacement_level_impact(
    player_name: str,
    team: str,
    pie: float,
    position: str,
    status: str,
    game_date: str,
    db_path: str
) -> float:
    """Enhanced replacement-level injury impact with:
    - Positional scarcity weighting
    - Replacement-level PIE baselines
    - Probability of playing (status-based)
    - Chemistry lag penalty for consecutive absences of star players
    
    Formula:
    absence_prob = 1 - P(play | status)
    lag_multiplier = 1 + CHEMISTRY_LAG_FACTOR * min(consecutive_absences, MAX_LAG_ABSENCES) if star
    impact = (player_PIE - replacement_PIE) * scarcity * absence_prob * lag_multiplier
    
    Returns positive expected loss. Non-positive returns 0.
    """
    replacement_pie = POSITION_REPLACEMENT_PIE.get(position, 0.090)
    base_scarcity = POSITION_SCARCITY.get(position, 1.0)
    scarcity_override = POSITION_SCARCITY_OVERRIDES.get(position, 1.0)
    scarcity = base_scarcity * scarcity_override

    play_prob = STATUS_PLAY_PROBABILITIES.get(status, 0.5)
    absence_prob = 1.0 - play_prob

    # Chemistry lag only for star-caliber players
    consecutive = _estimate_consecutive_absences(player_name, team, game_date, db_path) if absence_prob > 0 else 0
    is_star = pie >= STAR_PLAYER_PIE_THRESHOLD
    lag_multiplier = 1.0 + (CHEMISTRY_LAG_FACTOR * min(consecutive, MAX_LAG_ABSENCES)) if is_star and consecutive > 0 else 1.0

    raw_impact = (pie - replacement_pie) * scarcity * absence_prob * lag_multiplier
    # Enrich impact by considering minutes played (starter vs bench) and usage where available.
    # Fetch recent minutes per game (mpg) and usage (usg) if present to scale impact.
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # player_stats uses 'usg_pct' and may not include MPG; only use available fields
        cur.execute('''
            SELECT usg_pct FROM player_stats
            WHERE player_name = ?
            ORDER BY season DESC
            LIMIT 1
        ''', (player_name,))
        row = cur.fetchone()
        conn.close()
        mpg = 12.0
        usg = float(row[0]) / 100.0 if row and row[0] is not None else 0.12
    except Exception:
        mpg = 12.0
        usg = 0.12

    # Starter multiplier: increases weight for regular starters
    starter_multiplier = max(0.6, min(1.5, 0.6 + (mpg / 40.0)))

    # Minutes scale: linear-ish scaling capped to avoid extreme multipliers
    minutes_scale = max(0.5, min(1.25, mpg / 30.0))

    # Usage scale: small adjustment for high-usage players (guards/primary creators)
    usage_scale = max(0.8, min(1.4, 1.0 + (usg - 0.12)))

    enriched_impact = raw_impact * starter_multiplier * minutes_scale * usage_scale

    # Convert PIE-like units into net-rating-ish units (multiply by 100)
    net_rating_impact = max(0.0, enriched_impact) * 100.0
    return net_rating_impact


def calculate_team_injury_impact_advanced(
    team: str,
    game_date: str,
    db_path: str
) -> float:
    """
    Calculate total team injury impact using replacement-level model.
    
    Returns:
        Total injury impact for team (sum of all injured players' replacement-adjusted impact)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get current injuries for team
    cur.execute('''
        SELECT player_name, status, injury_detail
        FROM active_injuries
        WHERE team = ?
        AND status IN ('Out', 'Questionable', 'Doubtful', 'Probable')
    ''', (team,))
    
    injuries = cur.fetchall()
    
    total_impact = 0.0
    
    for player_name, status, injury_detail in injuries:
        # Get player PIE
        # Determine season string from game_date (e.g., '2016-11-01' -> '2016-17')
        def _season_from_date(dt_str: str) -> str:
            try:
                from datetime import datetime
                d = datetime.fromisoformat(dt_str)
                year = d.year
                month = d.month
                if month >= 10:
                    start = year
                    end = (year + 1) % 100
                else:
                    start = year - 1
                    end = year % 100
                return f"{start}-{end:02d}"
            except Exception:
                return None

        season_str = _season_from_date(game_date) if game_date else None

        player_data = None
        # Try season-aware table first
        if season_str:
            try:
                cur.execute('''
                    SELECT pie, position FROM player_season_metrics
                    WHERE player_name = ? AND team = ? AND season = ?
                    LIMIT 1
                ''', (player_name, team, season_str))
                player_data = cur.fetchone()
            except Exception:
                player_data = None

        # Fallback to career/latest player_stats if season-specific missing
        if not player_data:
            cur.execute('''
                SELECT pie, position FROM player_stats
                WHERE player_name = ? AND team = ?
                ORDER BY season DESC
                LIMIT 1
            ''', (player_name, team))
            player_data = cur.fetchone()

        if not player_data:
            continue

        pie, position = player_data
        if pie is None:
            # If pie missing in both, skip this player (or could use DEFAULT_PIE_FALLBACK)
            continue
        
        # Enhanced replacement-impact with probability weighting & lag
        impact = calculate_replacement_level_impact(
            player_name=player_name,
            team=team,
            pie=pie,
            position=position,
            status=status,
            game_date=game_date,
            db_path=db_path
        )
        
        total_impact += impact
    
    conn.close()
    return total_impact


def calculate_injury_impact_differential_advanced(
    home_team: str,
    away_team: str,
    game_date: str,
    db_path: str,
    breakdown: bool = False
) -> float | Tuple[float, Dict[str, float]]:
    """Calculate injury impact differential using enhanced replacement-level model.
    
    Differential = home_impact - away_impact
    Positive value => greater home losses (advantage shifts to away).
    
    If breakdown=True returns (differential, {'home_impact': x, 'away_impact': y}).
    """
    home_impact = calculate_team_injury_impact_advanced(home_team, game_date, db_path)
    away_impact = calculate_team_injury_impact_advanced(away_team, game_date, db_path)
    diff = home_impact - away_impact
    if breakdown:
        return diff, {'home_impact': home_impact, 'away_impact': away_impact}
    return diff


# Schema migration to add position column if missing
def ensure_position_column(db_path: str):
    """Add position column to player_stats if not present."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Check if position column exists
    cur.execute("PRAGMA table_info(player_stats)")
    cols = [r[1] for r in cur.fetchall()]
    
    if 'position' not in cols:
        try:
            cur.execute("ALTER TABLE player_stats ADD COLUMN position TEXT DEFAULT 'G'")
            conn.commit()
            print("Added 'position' column to player_stats")
        except Exception as e:
            print(f"Position column migration failed: {e}")
    
    conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python injury_replacement_model.py <db_path> <home_team> <away_team>")
        sys.exit(1)
    
    db = sys.argv[1]
    home = sys.argv[2]
    away = sys.argv[3]
    
    ensure_position_column(db)
    
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    
    home_impact = calculate_team_injury_impact_advanced(home, today, db)
    away_impact = calculate_team_injury_impact_advanced(away, today, db)
    diff = home_impact - away_impact
    
    print(f"\nInjury Impact Analysis:")
    print(f"  {home} impact: {home_impact:.4f}")
    print(f"  {away} impact: {away_impact:.4f}")
    print(f"  Differential: {diff:.4f} {'(favors away)' if diff > 0 else '(favors home)'}")
