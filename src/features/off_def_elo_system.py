"""
Off/Def ELO System
Provides separate offensive and defensive Elo ratings with:
- Adaptive K-factor (regular vs playoffs)
- Margin of victory scaling
- Season-to-season regression (25% toward baseline)
- Composite rating = (off_elo + (2000 - def_elo)) / 2  (map lower defensive ELO to higher strength)
- Exportable rating history for diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import sqlite3
from datetime import datetime

from config.constants import (
    OFF_ELO_BASELINE, DEF_ELO_BASELINE, SEASON_REGRESSION_FACTOR,
    REGULAR_SEASON_BASE_K, PLAYOFF_BASE_K, ELO_MARGIN_SCALE,
    ELO_POINT_EXPECTATION_SCALE, LEAGUE_AVG_POINTS,
    INJURY_OFF_SHARE, INJURY_DEF_SHARE, INJURY_ELO_SCALER
)

@dataclass
class TeamElo:
    team: str
    season: str
    off_elo: float
    def_elo: float
    last_updated: str

    @property
    def composite(self) -> float:
        # Defensive elo: lower number should represent worse defense.
        # To make composite intuitive (higher = stronger), invert defensive component.
        return (self.off_elo + (2000 - self.def_elo)) / 2.0

class OffDefEloSystem:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS elo_ratings (
                team TEXT NOT NULL,
                season TEXT NOT NULL,
                game_date TEXT NOT NULL,
                off_elo REAL NOT NULL,
                def_elo REAL NOT NULL,
                composite_elo REAL NOT NULL,
                is_playoffs INTEGER DEFAULT 0,
                PRIMARY KEY(team, game_date)
            )
        ''')
        conn.commit()
        conn.close()

    # ---------------- Initialization & Regression -----------------
    def initialize_season(self, season: str, teams: List[str]):
        """Initialize missing teams for a season (apply regression if prior season exists)."""
        for team in teams:
            self._initialize_team_for_season(team, season)

    def _initialize_team_for_season(self, team: str, season: str):
        # Check if team already has first record this season
        latest = self.get_latest(team, season)
        if latest:
            return
        # Attempt to fetch last season's final ratings
        prior_season = self._prior_season(season)
        prior_latest = self.get_latest(team, prior_season) if prior_season else None
        if prior_latest:
            off_elo = OFF_ELO_BASELINE + (prior_latest.off_elo - OFF_ELO_BASELINE) * SEASON_REGRESSION_FACTOR
            def_elo = DEF_ELO_BASELINE + (prior_latest.def_elo - DEF_ELO_BASELINE) * SEASON_REGRESSION_FACTOR
        else:
            off_elo = OFF_ELO_BASELINE
            def_elo = DEF_ELO_BASELINE
        self._insert_rating(team, season, self._today(), off_elo, def_elo, is_playoffs=0)

    def _prior_season(self, season: str) -> Optional[str]:
        try:
            start_year = int(season.split('-')[0])
            return f"{start_year-1}-{str(start_year)[-2:]}"  # naive pattern
        except Exception:
            return None

    # ---------------- Core Helpers -----------------
    def _today(self) -> str:
        return datetime.utcnow().strftime('%Y-%m-%d')

    def get_latest(self, team: str, season: str, before_date: Optional[str] = None) -> Optional[TeamElo]:
        """
        Return the latest ELO record for `team` in `season` strictly before `before_date` if provided.
        If `before_date` is None, return the latest available record (legacy behavior).
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        if before_date:
            # Ensure we only return ratings *before* the given game date to avoid look-ahead leakage
            cur.execute('''
                SELECT team, season, game_date, off_elo, def_elo FROM elo_ratings
                WHERE team = ? AND season = ? AND game_date < ?
                ORDER BY game_date DESC LIMIT 1
            ''', (team, season, before_date))
        else:
            cur.execute('''
                SELECT team, season, game_date, off_elo, def_elo FROM elo_ratings
                WHERE team = ? AND season = ?
                ORDER BY game_date DESC LIMIT 1
            ''', (team, season))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return TeamElo(team=row[0], season=row[1], last_updated=row[2], off_elo=row[3], def_elo=row[4])

    def _insert_rating(self, team: str, season: str, game_date: str, off_elo: float, def_elo: float, is_playoffs: int):
        composite = (off_elo + def_elo) / 2.0  # FIXED: removed inversion
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO elo_ratings (team, season, game_date, off_elo, def_elo, composite_elo, is_playoffs)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (team, season, game_date, off_elo, def_elo, composite, is_playoffs))
        conn.commit(); conn.close()

    # ---------------- Update Logic -----------------
    def update_game(self, season: str, game_date: str, home_team: str, away_team: str,
                    home_points: int, away_points: int, is_playoffs: bool=False,
                    home_injury_impact: float = 0.0, away_injury_impact: float = 0.0):
        """Update off/def Elo ratings based on game result and scoring.
        Each game affects four interactions:
        - Home offense vs Away defense
        - Away offense vs Home defense
        Margin scaling increases impact for decisive results.
        
        Args:
            home_injury_impact: Total replacement-level point impact from home team injuries (negative = weakened)
            away_injury_impact: Total replacement-level point impact from away team injuries (negative = weakened)
        """
        home_latest = self.get_latest(home_team, season) or TeamElo(home_team, season, OFF_ELO_BASELINE, DEF_ELO_BASELINE, game_date)
        away_latest = self.get_latest(away_team, season) or TeamElo(away_team, season, OFF_ELO_BASELINE, DEF_ELO_BASELINE, game_date)

        base_k = PLAYOFF_BASE_K if is_playoffs else REGULAR_SEASON_BASE_K
        margin = abs(home_points - away_points)
        
        # BLOWOUT DAMPENING: Reduce impact of margins > 20 points
        # Large margins often reflect garbage time and don't indicate true skill gap
        BLOWOUT_THRESHOLD = 20
        BLOWOUT_DAMPENING = 0.5  # Reduce blowout impact by 50%
        
        if margin > BLOWOUT_THRESHOLD:
            # Cap effective margin: base margin + dampened excess
            excess = margin - BLOWOUT_THRESHOLD
            effective_margin = BLOWOUT_THRESHOLD + (excess * BLOWOUT_DAMPENING)
        else:
            effective_margin = margin
        
        margin_factor = min(effective_margin / ELO_MARGIN_SCALE, 1.5)  # cap
        k_off = base_k * (1 + 0.5 * margin_factor)
        k_def = base_k * (1 + 0.5 * margin_factor)

        # Expected points using difference of offense and defense elo
        # Apply injury lag adjustments: injuries depress offensive output and weaken defensive quality
        # Injury impact sign convention: negative = team weakened, positive = strengthened
        home_injury_off_delta = home_injury_impact * INJURY_OFF_SHARE * INJURY_ELO_SCALER
        home_injury_def_delta = home_injury_impact * INJURY_DEF_SHARE * INJURY_ELO_SCALER
        away_injury_off_delta = away_injury_impact * INJURY_OFF_SHARE * INJURY_ELO_SCALER
        away_injury_def_delta = away_injury_impact * INJURY_DEF_SHARE * INJURY_ELO_SCALER
        
        # Effective Elo incorporating injury context
        home_eff_off = home_latest.off_elo + home_injury_off_delta
        home_eff_def = home_latest.def_elo + home_injury_def_delta  # worse def (higher elo allows more pts)
        away_eff_off = away_latest.off_elo + away_injury_off_delta
        away_eff_def = away_latest.def_elo + away_injury_def_delta
        
        exp_home_pts = LEAGUE_AVG_POINTS + (home_eff_off - away_eff_def) / ELO_POINT_EXPECTATION_SCALE
        exp_away_pts = LEAGUE_AVG_POINTS + (away_eff_off - home_eff_def) / ELO_POINT_EXPECTATION_SCALE

        # Errors
        home_off_error = home_points - exp_home_pts
        away_off_error = away_points - exp_away_pts

        # Defensive error is negative of opponent offensive error
        home_def_error = -away_off_error
        away_def_error = -home_off_error

        # Update ratings proportional to error
        new_home_off = home_latest.off_elo + k_off * (home_off_error / ELO_POINT_EXPECTATION_SCALE)
        new_away_def = away_latest.def_elo - k_def * (home_off_error / ELO_POINT_EXPECTATION_SCALE)
        new_away_off = away_latest.off_elo + k_off * (away_off_error / ELO_POINT_EXPECTATION_SCALE)
        new_home_def = home_latest.def_elo - k_def * (away_off_error / ELO_POINT_EXPECTATION_SCALE)

        self._insert_rating(home_team, season, game_date, new_home_off, new_home_def, int(is_playoffs))
        self._insert_rating(away_team, season, game_date, new_away_off, new_away_def, int(is_playoffs))

        return {
            'home_team': home_team,
            'away_team': away_team,
            'season': season,
            'game_date': game_date,
            'home_off_before': home_latest.off_elo,
            'home_def_before': home_latest.def_elo,
            'away_off_before': away_latest.off_elo,
            'away_def_before': away_latest.def_elo,
            'home_off_after': new_home_off,
            'home_def_after': new_home_def,
            'away_off_after': new_away_off,
            'away_def_after': new_away_def,
            'margin': margin,
            'k_off': k_off,
            'k_def': k_def,
            'exp_home_pts': exp_home_pts,
            'exp_away_pts': exp_away_pts,
            'home_points': home_points,
            'away_points': away_points
        }

    # ---------------- Feature Access -----------------
    def get_differentials(self, season: str, home_team: str, away_team: str, game_date: Optional[str] = None) -> Dict[str, float]:
        """
        Return differential features for a matchup using ELO ratings strictly prior to `game_date` when provided.
        """
        home = self.get_latest(home_team, season, before_date=game_date) or TeamElo(home_team, season, OFF_ELO_BASELINE, DEF_ELO_BASELINE, self._today())
        away = self.get_latest(away_team, season, before_date=game_date) or TeamElo(away_team, season, OFF_ELO_BASELINE, DEF_ELO_BASELINE, self._today())
        return {
            'off_elo_diff': home.off_elo - away.off_elo,
            'def_elo_diff': (2000 - home.def_elo) - (2000 - away.def_elo),  # defensive strength differential
            'composite_elo_diff': home.composite - away.composite
        }

    def update_game_result(self, *args, **kwargs):
        """Alias for update_game for backward compatibility"""
        return self.update_game(*args, **kwargs)
    
    def get_team_history(self, team: str, season: Optional[str]=None) -> List[Dict[str, float]]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        if season:
            cur.execute('''
                SELECT game_date, off_elo, def_elo, composite_elo FROM elo_ratings
                WHERE team = ? AND season = ? ORDER BY game_date ASC
            ''', (team, season))
        else:
            cur.execute('''
                SELECT game_date, off_elo, def_elo, composite_elo, season FROM elo_ratings
                WHERE team = ? ORDER BY game_date ASC
            ''', (team,))
        rows = cur.fetchall(); conn.close()
        history = []
        for r in rows:
            if season:
                history.append({'game_date': r[0], 'off_elo': r[1], 'def_elo': r[2], 'composite_elo': r[3]})
            else:
                history.append({'game_date': r[0], 'off_elo': r[1], 'def_elo': r[2], 'composite_elo': r[3], 'season': r[4]})
        return history

if __name__ == '__main__':
    import random
    db = 'nba_betting.db'
    system = OffDefEloSystem(db)
    teams = ['LAL','BOS','GSW','DEN','MIA']
    system.initialize_season('2025-26', teams)
    # Simulate games
    for d in range(1,6):
        system.update_game('2025-26', f'2025-11-0{d}', 'LAL','BOS', random.randint(100,120), random.randint(95,115))
    print(system.get_differentials('2025-26','LAL','BOS'))
    print(system.get_team_history('LAL','2025-26')[:3])
