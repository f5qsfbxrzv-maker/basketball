"""
Data Models & Type Definitions for NBA Betting System
Centralized dataclasses for type safety and clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class GameInfo:
    """Core game identification and scheduling information."""
    game_id: str
    home_team: str
    away_team: str
    start_time: Optional[str] = None  # ISO 8601 string
    game_date: Optional[str] = None  # YYYY-MM-DD format
    total_line: Optional[float] = None
    spread_line: Optional[float] = None
    home_ml_odds: Optional[float] = None
    away_ml_odds: Optional[float] = None


@dataclass
class GameFeatures:
    """Feature vector container for ML model inputs."""
    raw: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export raw features as dict."""
        return self.raw.copy()
    
    def get(self, key: str, default: Any = 0) -> Any:
        """Safe feature accessor with default."""
        return self.raw.get(key, default)


@dataclass
class PredictionResult:
    """Complete prediction output with probabilities, edges, and metadata."""
    game_id: str
    over_prob: float
    calibrated_over_prob: float
    total_line: Optional[float]
    model_edge: float  # calibrated_prob - effective_cost
    heuristic_prob: float
    model_prob: float
    applied_calibration: bool
    effective_yes_price: float  # Kalshi price after commission
    
    # Advanced model enrichment (optional)
    poisson_over_prob: Optional[float] = None
    poisson_expected_total: Optional[float] = None
    poisson_total_std: Optional[float] = None
    bivariate_parlay_prob: Optional[float] = None
    bivariate_parlay_edge: Optional[float] = None
    bayesian_home_win_prob: Optional[float] = None
    bayesian_expected_margin: Optional[float] = None
    
    # Metadata
    prediction_timestamp: Optional[str] = None
    model_version: Optional[str] = None
    calibration_version: Optional[str] = None


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics for reliability assessment."""
    brier_score: float
    sample_count: int
    max_decile_gap: float
    kelly_calibration_factor: float
    factor_brier_component: float
    factor_gap_component: float
    timestamp: Optional[str] = None
    model_version: Optional[str] = None
    calibration_version: Optional[str] = None


@dataclass
class BetRecommendation:
    """Kelly-optimized bet sizing recommendation."""
    game_id: str
    market_type: str  # 'total', 'spread', 'moneyline'
    position: str  # 'over', 'under', 'home', 'away'
    model_probability: float
    market_probability: float  # implied from odds
    edge: float
    kelly_fraction: float
    recommended_stake: float
    max_stake_cap: float
    confidence: float
    notes: str = ""


@dataclass
class InjuryImpact:
    """Injury report impact assessment for a single player."""
    player_name: str
    team: str
    status: str  # 'Out', 'Doubtful', 'Questionable', 'Probable', 'Active'
    play_probability: float
    offensive_impact: float  # points
    defensive_impact: float  # points
    total_impact: float  # points (net)
    pie_score: Optional[float] = None
    position: Optional[str] = None


@dataclass
class EloRating:
    """Team ELO rating snapshot."""
    team: str
    composite_elo: float
    offensive_elo: float
    defensive_elo: float
    date: str  # ISO date
    games_played: int = 0


@dataclass
class ModelMetrics:
    """Model performance evaluation metrics."""
    model_name: str
    brier_score: float
    log_loss: float
    roc_auc: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mae: Optional[float] = None  # for regressors
    rmse: Optional[float] = None  # for regressors
    sample_count: int = 0
    timestamp: Optional[str] = None


@dataclass
class RiskMetrics:
    """Bankroll and risk management metrics."""
    current_bankroll: float
    initial_bankroll: float
    total_profit_loss: float
    total_bets: int
    win_rate: float
    avg_edge: float
    kelly_utilization: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    sharpe_ratio: Optional[float] = None
    roi: float = 0.0


@dataclass
class BacktestResult:
    """Single backtest simulation result."""
    start_date: str
    end_date: str
    total_bets: int
    win_count: int
    loss_count: int
    final_bankroll: float
    total_profit_loss: float
    roi: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_edge: float
    avg_stake_pct: float
