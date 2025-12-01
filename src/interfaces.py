"""
Interface definitions for core system components.
Enables dependency injection and modular swapping of implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd


# ==============================================================================
# DATA TRANSFER OBJECTS
# ==============================================================================

@dataclass
class GameSchedule:
    """Raw game schedule information"""
    game_id: str
    home_team: str
    away_team: str
    game_date: str  # YYYY-MM-DD
    game_time: str  # HH:MM
    season: str
    is_playoffs: bool = False


@dataclass
class GameFeatures:
    """Engineered features for a single game"""
    game_id: str
    features: Dict[str, float]
    
    def to_dataframe(self, feature_order: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert to single-row DataFrame with specified column order"""
        if feature_order:
            data = {k: self.features.get(k, 0.0) for k in feature_order}
        else:
            data = self.features
        return pd.DataFrame([data])


@dataclass
class MarketOdds:
    """Market pricing for a game"""
    game_id: str
    total_line: Optional[float] = None
    yes_price_cents: Optional[int] = None  # Kalshi: 0-100 cents
    no_price_cents: Optional[int] = None
    timestamp: Optional[str] = None
    source: str = "kalshi"


@dataclass
class ModelPrediction:
    """Raw model output before calibration"""
    game_id: str
    probability_over: float
    predicted_total: Optional[float] = None
    model_confidence: Optional[float] = None  # e.g., margin or entropy


@dataclass
class CalibratedPrediction:
    """Prediction after calibration transform"""
    game_id: str
    raw_probability: float
    calibrated_probability: float
    calibration_method: str  # "isotonic", "platt", "none"
    reliability_score: Optional[float] = None


@dataclass
class EdgeCalculation:
    """Edge and betting recommendation"""
    game_id: str
    model_probability: float
    market_probability: float
    edge_pct: float
    kelly_fraction: float
    recommended_stake_pct: float
    bet_side: str  # "over", "under", "none"
    confidence_tier: str  # "strong", "medium", "weak"


@dataclass
class InjuryReport:
    """Player injury status"""
    player_name: str
    team: str
    status: str  # "Out", "Doubtful", "Questionable", "Probable"
    position: str
    pie_impact: float  # Player Impact Estimate
    games_missed: int = 0


# ==============================================================================
# CORE INTERFACES
# ==============================================================================

class IDataCollector(ABC):
    """Interface for collecting raw data (schedules, stats, injuries)"""
    
    @abstractmethod
    def fetch_schedule(self, date: str) -> List[GameSchedule]:
        """Get games scheduled for a specific date"""
        pass
    
    @abstractmethod
    def fetch_injuries(self, date: str) -> List[InjuryReport]:
        """Get active injuries as of a date"""
        pass
    
    @abstractmethod
    def fetch_team_stats(self, team: str, season: str) -> Dict[str, Any]:
        """Get team statistics for a season"""
        pass


class IFeatureEngine(ABC):
    """Interface for feature engineering"""
    
    @abstractmethod
    def calculate_features(self, game: GameSchedule, context: Dict[str, Any]) -> GameFeatures:
        """
        Calculate all features for a game.
        
        Args:
            game: Game schedule info
            context: Additional context (injuries, recent stats, etc.)
            
        Returns:
            GameFeatures with all engineered features
        """
        pass
    
    @abstractmethod
    def batch_calculate(self, games: List[GameSchedule], context: Dict[str, Any]) -> List[GameFeatures]:
        """Calculate features for multiple games efficiently"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        pass


class IPredictor(ABC):
    """Interface for generating predictions"""
    
    @abstractmethod
    def predict(self, features: GameFeatures) -> ModelPrediction:
        """Generate raw prediction from features"""
        pass
    
    @abstractmethod
    def batch_predict(self, features_list: List[GameFeatures]) -> List[ModelPrediction]:
        """Batch prediction for efficiency"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata (version, training date, performance)"""
        pass


class ICalibration(ABC):
    """Interface for probability calibration"""
    
    @abstractmethod
    def calibrate(self, prediction: ModelPrediction) -> CalibratedPrediction:
        """Apply calibration transform to raw prediction"""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if calibration model is fitted and ready"""
        pass
    
    @abstractmethod
    def fit(self, predictions: List[float], outcomes: List[int]) -> None:
        """Fit calibration model on historical data"""
        pass
    
    @abstractmethod
    def get_reliability_curve(self) -> Dict[str, List[float]]:
        """Get reliability diagram data for visualization"""
        pass


class IRiskManager(ABC):
    """Interface for position sizing and risk management"""
    
    @abstractmethod
    def calculate_edge(self, prediction: CalibratedPrediction, odds: MarketOdds) -> EdgeCalculation:
        """
        Calculate edge and Kelly sizing.
        
        Args:
            prediction: Calibrated model prediction
            odds: Market pricing
            
        Returns:
            EdgeCalculation with sizing recommendation
        """
        pass
    
    @abstractmethod
    def validate_bet(self, edge: EdgeCalculation, bankroll: float, config: Dict[str, Any]) -> bool:
        """Validate bet meets risk criteria (min edge, max stake, etc.)"""
        pass
    
    @abstractmethod
    def calculate_kelly_stake(self, edge_pct: float, odds_decimal: float, bankroll: float) -> float:
        """Calculate Kelly criterion stake"""
        pass


class IOddsProvider(ABC):
    """Interface for fetching market odds"""
    
    @abstractmethod
    def get_odds(self, game_id: str) -> Optional[MarketOdds]:
        """Get current odds for a game"""
        pass
    
    @abstractmethod
    def get_odds_history(self, game_id: str) -> List[MarketOdds]:
        """Get historical odds snapshots"""
        pass
    
    @abstractmethod
    def get_line_movement(self, game_id: str) -> Optional[float]:
        """Get line movement (current - opening)"""
        pass


class IExecutionEngine(ABC):
    """Interface for bet execution (paper or live)"""
    
    @abstractmethod
    def place_bet(self, game_id: str, side: str, stake: float, odds: float) -> Dict[str, Any]:
        """Place a bet and return confirmation"""
        pass
    
    @abstractmethod
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get currently open positions"""
        pass
    
    @abstractmethod
    def close_position(self, position_id: str) -> Dict[str, Any]:
        """Close an open position"""
        pass


# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================

class IPredictionPipeline(ABC):
    """Orchestrates the complete prediction workflow"""
    
    @abstractmethod
    def run_pipeline(self, date: str) -> List[EdgeCalculation]:
        """
        Execute full pipeline for a date:
        1. Fetch schedule
        2. Collect context (injuries, stats)
        3. Engineer features
        4. Generate predictions
        5. Apply calibration
        6. Calculate edges
        7. Filter by criteria
        
        Returns:
            List of EdgeCalculations for actionable bets
        """
        pass
    
    @abstractmethod
    def run_single_game(self, game_id: str) -> Optional[EdgeCalculation]:
        """Run pipeline for a single game"""
        pass
