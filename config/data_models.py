"""
Data models for NBA betting system
Simple dataclasses for type safety
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class GameInfo:
    """Basic game information"""
    game_id: str
    game_date: datetime
    home_team: str
    away_team: str
    season: str = "2024-25"
    
@dataclass
class GameFeatures:
    """Computed features for a game"""
    features: Dict[str, float]
    feature_names: list[str]
    
    def to_dict(self) -> Dict[str, float]:
        return self.features

@dataclass
class PredictionResult:
    """Model prediction result"""
    game_id: str
    game_date: datetime
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    home_edge: Optional[float] = None
    away_edge: Optional[float] = None
    recommended_bet: Optional[str] = None
    stake_size: Optional[float] = None
    model_version: Optional[str] = None
    calibrated: bool = True
