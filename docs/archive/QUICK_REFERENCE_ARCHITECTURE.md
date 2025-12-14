# Quick Reference: New Architecture

## Import Paths

```python
# Interfaces
from interfaces import (
    IDataCollector, IFeatureEngine, IPredictor, ICalibration,
    IRiskManager, IOddsProvider, IPredictionPipeline,
    GameSchedule, GameFeatures, MarketOdds, EdgeCalculation
)

# Services
from services.prediction_service import PredictionService
from services.risk_manager import RiskManager
from services.pipeline_orchestrator import PredictionPipeline

# Constants
from constants import (
    KELLY_FRACTION_MULTIPLIER, MIN_EDGE_FOR_BET,
    KALSHI_BUY_COMMISSION, HEURISTIC_BLEND_WEIGHT
)
```

---

## Common Patterns

### Pattern 1: Run Full Pipeline
```python
# Setup
pipeline = PredictionPipeline(
    data_collector=data_collector,
    feature_engine=feature_engine,
    predictor=predictor,
    calibration=calibration,
    risk_manager=risk_manager,
    odds_provider=odds_provider,
    config=config
)

# Execute
edges = pipeline.run_pipeline("2024-11-19")

# Display
for edge in edges:
    print(f"{edge.game_id}: {edge.bet_side} @ {edge.edge_pct:+.2%}")
```

### Pattern 2: Single Game Prediction
```python
# Create service
service = PredictionService(predictor, calibration, risk_manager, config)

# Generate prediction
edge = service.generate_prediction(features, odds)

# Check if actionable
if edge.edge_pct >= MIN_EDGE_FOR_BET and edge.bet_side != "none":
    print(f"BET: {edge.bet_side} {edge.recommended_stake_pct:.2%}")
```

### Pattern 3: Calculate Kelly Stake
```python
# Create risk manager
risk_manager = RiskManager(config)

# Calculate stake
stake = risk_manager.calculate_kelly_stake(
    edge_pct=0.05,        # 5% edge
    odds_decimal=2.0,     # Even money
    bankroll=10000.0      # $10k bankroll
)
# stake ≈ $125 (0.05 * 0.25 * 10000 capped at 5%)
```

### Pattern 4: Validate Bet
```python
# Check if bet meets criteria
is_valid = risk_manager.validate_bet(
    edge=edge,
    bankroll=10000.0,
    config={"min_stake_amount": 10, "max_stake_amount": 500}
)

if is_valid:
    place_bet(edge)
```

---

## Interface Implementations Needed

### NBADataCollector
```python
class NBADataCollector(IDataCollector):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
    
    def fetch_schedule(self, date: str) -> List[GameSchedule]:
        # Query game_results WHERE game_date = date
        # Return [GameSchedule(...), ...]
        pass
```

### FeatureEngineV5
```python
class FeatureEngineV5(IFeatureEngine):
    def __init__(self, db_path: str):
        from feature_calculator_v5 import FeatureCalculatorV5
        self.calculator = FeatureCalculatorV5(db_path)
    
    def calculate_features(self, game: GameSchedule, context: Dict) -> GameFeatures:
        features = self.calculator.calculate_game_features(
            game.home_team, game.away_team, game.game_date
        )
        return GameFeatures(game_id=game.game_id, features=features)
```

### XGBoostPredictor
```python
class XGBoostPredictor(IPredictor):
    def __init__(self, model_path: str):
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
    
    def predict(self, features: GameFeatures) -> ModelPrediction:
        df = features.to_dataframe(self.feature_order)
        prob = self.model.predict_proba(df)[0, 1]
        return ModelPrediction(
            game_id=features.game_id,
            probability_over=prob
        )
```

### IsotonicCalibration
```python
class IsotonicCalibration(ICalibration):
    def __init__(self, db_path: str):
        from calibration_fitter import CalibrationFitter
        self.fitter = CalibrationFitter(db_path)
    
    def calibrate(self, prediction: ModelPrediction) -> CalibratedPrediction:
        calibrated = self.fitter.apply(prediction.probability_over)
        return CalibratedPrediction(
            game_id=prediction.game_id,
            raw_probability=prediction.probability_over,
            calibrated_probability=calibrated,
            calibration_method="isotonic"
        )
    
    def is_ready(self) -> bool:
        return self.fitter.is_ready()
```

---

## EdgeCalculation Fields

```python
@dataclass
class EdgeCalculation:
    game_id: str
    model_probability: float        # 0.58 = 58% chance of over
    market_probability: float       # 0.52 = market implies 52%
    edge_pct: float                 # 0.06 = 6% edge
    kelly_fraction: float           # 0.0462 = full Kelly
    recommended_stake_pct: float    # 0.0115 = 1.15% of bankroll (quarter Kelly)
    bet_side: str                   # "over", "under", "none"
    confidence_tier: str            # "weak", "medium", "strong"
```

---

## Config Structure

```yaml
prediction:
  heuristic_weight: 0.35
  model_weight: 0.65

risk:
  kelly_fraction: 0.25        # Quarter Kelly
  max_stake_pct: 0.05         # 5% max
  min_stake_pct: 0.01         # 1% min
  min_edge: 0.03              # 3% minimum
  strong_edge: 0.08           # 8% = strong

kalshi:
  buy_commission: 0.02
  sell_commission: 0.02
  expiry_commission: 0.0

heuristic:
  pace_weight: 0.002
  offense_weight: 0.001
  injury_weight: 0.01
  line_weight: 0.01
```

---

## Testing Examples

### Unit Test Service
```python
def test_prediction_service():
    mock_predictor = Mock(IPredictor)
    mock_predictor.predict.return_value = ModelPrediction(
        game_id="test", probability_over=0.58
    )
    
    service = PredictionService(mock_predictor, mock_cal, mock_risk, config)
    edge = service.generate_prediction(features, odds)
    
    assert edge.edge_pct > 0
    mock_predictor.predict.assert_called_once()
```

### Integration Test Pipeline
```python
def test_pipeline():
    pipeline = PredictionPipeline(...)
    edges = pipeline.run_pipeline("2024-11-19")
    
    assert len(edges) >= 0
    for edge in edges:
        assert edge.bet_side in ["over", "under", "none"]
        assert 0 <= edge.model_probability <= 1
```

---

## Dashboard Integration

### Before (Legacy)
```python
def _load_predictions_for_date(self):
    games = self._get_games_from_db(date)
    for game in games:
        total, prob = self._model_total_and_prob(game, market_line)
        self._create_game_widget(game, total, prob)
```

### After (Service Layer)
```python
def _load_predictions_for_date(self):
    date = self.date_selector.currentText()
    edges = self.pipeline.run_pipeline(date)
    for edge in edges:
        self._create_edge_widget(edge)

def _create_edge_widget(self, edge: EdgeCalculation):
    # Display edge.bet_side, edge.edge_pct, edge.recommended_stake_pct
    # All logic already computed by service layer
```

---

## Debugging Commands

```python
# Check component status
status = pipeline.get_pipeline_status()
print(status)
# {'data_collector': 'ready', 'predictor': 'ready', ...}

# Get model info
info = predictor.get_model_info()
print(info)
# {'model_type': 'XGBoost', 'accuracy': 0.58, ...}

# Get reliability curve
curve = calibration.get_reliability_curve()
# {'predicted': [...], 'observed': [...], 'counts': [...]}

# Check if calibration ready
is_ready = calibration.is_ready()
# True/False
```

---

## Common Mistakes to Avoid

❌ **Don't mix business logic in UI**
```python
# BAD: Heuristics in dashboard
def _create_game_widget(self, game):
    prob = 0.5 + 0.002 * pace + 0.01 * injury  # ❌
```

✅ **Use service layer**
```python
# GOOD: Service handles logic
edge = self.pipeline.prediction_service.generate_prediction(features, odds)
```

❌ **Don't hardcode configuration**
```python
kelly_fraction = 0.25  # ❌ Magic number
```

✅ **Use constants or config**
```python
from constants import KELLY_FRACTION_MULTIPLIER
kelly_fraction = KELLY_FRACTION_MULTIPLIER  # ✅
```

❌ **Don't skip validation**
```python
place_bet(edge)  # ❌ What if edge too small?
```

✅ **Validate before betting**
```python
if risk_manager.validate_bet(edge, bankroll, config):
    place_bet(edge)  # ✅
```

---

**Last Updated**: 2024-11-19  
**Reference**: ARCHITECTURE_REFACTOR.md, IMPLEMENTATION_COMPLETE.md
