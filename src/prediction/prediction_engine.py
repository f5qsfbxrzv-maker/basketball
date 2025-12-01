from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src.core.calibration_fitter import CalibrationFitter as CalibrationFitter
from src.core.calibration_logger import CalibrationLogger as CalibrationLogger
from src.constants import (
    MIN_PROBABILITY, MAX_PROBABILITY,
    KALSHI_BUY_COMMISSION,
    HEURISTIC_PACE_COEFFICIENT,
    HEURISTIC_COMPOSITE_ELO_COEFFICIENT,
    HEURISTIC_OFF_ELO_COEFFICIENT,
    HEURISTIC_DEF_ELO_COEFFICIENT,
    HEURISTIC_INJURY_COEFFICIENT,
    HEURISTIC_TOTAL_LINE_COEFFICIENT
)
from src.data_models import GameInfo, GameFeatures, PredictionResult
from src.logger_setup import get_structured_adapter, classify_error


class PredictionEngine:
    """
    Central prediction orchestration system for NBA game totals with hybrid ML + heuristic approach.
    
    This engine coordinates the complete prediction workflow:
    1. **Feature Extraction**: Receives GameFeatures from FeatureCalculatorV5V5 (120+ signals)
    2. **Model Prediction**: XGBoost classifier produces raw Over probability
    3. **Heuristic Fallback**: If model fails, uses weighted heuristic from ELO/pace/injuries
    4. **Calibration**: Applies isotonic or Platt calibration to raw probability (MANDATORY)
    5. **Edge Calculation**: Compares calibrated probability vs market price (minus commission)
    6. **Advanced Models**: Optional Poisson/Bivariate/Bayesian for scenario simulation
    
    **CRITICAL WORKFLOW**:
    - Raw ML probability → Calibration (isotonic/Platt) → Edge calculation → Kelly sizing
    - NEVER skip calibration step (causes systematic overbetting)
    - Commission-adjusted edge = calibrated_prob - (market_price * (1 + KALSHI_BUY_COMMISSION))
    
    Heuristic coefficients (when XGBoost unavailable):
    - HEURISTIC_PACE_COEFFICIENT: 0.0016 (pace impact on total)
    - HEURISTIC_COMPOSITE_ELO_COEFFICIENT: 0.0022 (ELO strength impact)
    - HEURISTIC_OFF/DEF_ELO_COEFFICIENT: Separate offensive/defensive contributions
    - HEURISTIC_INJURY_COEFFICIENT: Expected points lost to injuries
    - HEURISTIC_TOTAL_LINE_COEFFICIENT: Baseline adjustment from line
    
    Attributes:
        cfg (Dict[str, Any]): Configuration dictionary with model paths, thresholds
        calibration_fitter (CalibrationFitter): Calibration system (isotonic/Platt)
        calibration_logger (CalibrationLogger): Logs predictions for future calibration updates
        model (xgb.XGBClassifier): Trained XGBoost model for totals Over/Under
        feature_order (List[str]): Feature names in model training order
        event_logger (StructuredLoggerAdapter): Structured logging with prediction context
        _poisson_model (PoissonTotalModel): Optional Poisson/Negative Binomial model
        _bivariate_model (BivariateSpreadTotalModel): Optional correlated spread-total model
        _bayesian_model (BayesianHierarchicalModel): Optional Bayesian hierarchical model
        _advanced_ready (bool): True if advanced models initialized successfully
    
    Examples:
        >>> from src.core.calibration_fitter import CalibrationFitter
        >>> from src.core.calibration_logger import CalibrationLogger
        >>> config = {'advanced_models': {'enabled': True, 'use_negative_binomial': True}}
        >>> fitter = CalibrationFitter(db_path='data/database/data/database/nba_betting_data.db')
        >>> logger = CalibrationLogger(db_path='data/database/data/database/nba_betting_data.db')
        >>> engine = PredictionEngine(
        ...     config=config,
        ...     model_path='models/model_v5_total.xgb',
        ...     calibration_fitter=fitter,
        ...     calibration_logger=logger
        ... )
        >>> # Make prediction
        >>> from src.core.feature_calculator_v5 import FeatureCalculatorV5V5
        >>> calc = FeatureCalculatorV5V5()
        >>> features = calc.calculate_game_features(game_id='0022400123')
        >>> result = engine.predict_total(
        ...     game_info=game_info,
        ...     features=features,
        ...     total_line=225.5,
        ...     market_yes_price=0.50
        ... )
        >>> print(f"Over prob: {result.over_prob:.3f}")
        >>> print(f"Calibrated: {result.calibrated_over_prob:.3f}")
        >>> print(f"Edge: {result.model_edge:.3f}")
    
    See Also:
        - `feature_calculator_v5.py`: Feature engineering (120+ signals)
        - `calibration_fitter.py`: Isotonic/Platt calibration
        - `calibration_logger.py`: Prediction tracking for calibration updates
        - `advanced_models.py`: Poisson, Bayesian models
        - `bivariate_model.py`: Correlated spread-total modeling
        - `constants.py`: Heuristic coefficients, commission rates
    """
    def __init__(self, config: Dict[str, Any], model_path: Optional[str], calibration_fitter: CalibrationFitter, calibration_logger: CalibrationLogger, system_version: str = '6.0') -> None:
        """
        Initialize prediction engine with configuration, model, and calibration systems.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary with keys:
                - 'advanced_models': {'enabled': bool, 'use_negative_binomial': bool, 'enable_bayesian': bool}
            model_path (Optional[str]): Path to trained XGBoost model file (.xgb or .json)
            calibration_fitter (CalibrationFitter): Calibration system for probability adjustment
            calibration_logger (CalibrationLogger): Logger for tracking predictions
            system_version (str): System version identifier (default: '6.0')
        
        Raises:
            Exception: If model loading fails (falls back to heuristic predictions)
        """
        self.cfg = config
        self.calibration_fitter = calibration_fitter
        self.calibration_logger = calibration_logger
        self.VERSION = "v6"
        self.system_version = system_version
        self.model = None
        # Advanced model suite (lazy init)
        self._poisson_model = None
        self._bivariate_model = None
        self._bayesian_model = None
        self._advanced_ready = False
        # Structured logger
        pred_ver = getattr(calibration_fitter, 'model_version', 'v5.0') if calibration_fitter else 'v5.0'
        self.event_logger = get_structured_adapter(component='prediction', prediction_version=pred_ver)
        if model_path and xgb:
            p = Path(model_path)
            if p.exists():
                try:
                    self.model = xgb.XGBClassifier()
                    self.model.load_model(str(p))
                except Exception as e:
                    self.event_logger.event('warning', f"Model load failed: {e}", category=classify_error(e))
                    self.model = None

        self.feature_order = None
        # Attempt advanced model initialization if enabled
        self._init_advanced_models()

    def _init_advanced_models(self) -> None:
        try:
            adv_cfg = self.cfg.get('advanced_models', {}) if isinstance(self.cfg, dict) else {}
            if not adv_cfg.get('enabled', False):
                return
            from advanced_models import PoissonTotalModel, BayesianHierarchicalModel
            from bivariate_model import BivariateSpreadTotalModel
            # Initialize Poisson model (will require external fit call elsewhere)
            self._poisson_model = PoissonTotalModel(use_negative_binomial=adv_cfg.get('use_negative_binomial', True))
            # Bivariate model (fit requires historical game frame)
            self._bivariate_model = BivariateSpreadTotalModel()
            # Bayesian model (player level) optional heavy dependency
            if adv_cfg.get('enable_bayesian', True):
                self._bayesian_model = BayesianHierarchicalModel()
            self._advanced_ready = True
            self.event_logger.event('info', 'advanced_models_initialized', category='init')
        except Exception as e:
            self.event_logger.event('warning', f'Advanced model init failed: {e}', category=classify_error(e))
            self._advanced_ready = False

    def _heuristic_total_prob(self, features: GameFeatures, total_line: float) -> float:
        """Heuristic probability calculation - fallback when model unavailable"""
        pace = features.raw.get("expected_pace", 0)
        injury = features.raw.get("injury_impact_diff", 0)
        # Prefer advanced composite/off/def Elo signals if available
        composite = features.raw.get("composite_elo_diff")
        off_diff = features.raw.get("off_elo_diff")
        def_diff = features.raw.get("def_elo_diff")
        legacy_elo = features.raw.get("elo_diff", 0)

        if composite is not None and off_diff is not None and def_diff is not None:
            # Offensive differential increases over probability; defensive differential (higher = better home defense) lowers scoring
            # Composite captures overall strength context.
            base = 0.5 
            base += HEURISTIC_PACE_COEFFICIENT * pace
            base += HEURISTIC_COMPOSITE_ELO_COEFFICIENT * composite
            base += HEURISTIC_OFF_ELO_COEFFICIENT * off_diff
            base -= HEURISTIC_DEF_ELO_COEFFICIENT * def_diff
            base += HEURISTIC_INJURY_COEFFICIENT * injury
        else:
            # Fallback legacy behavior
            base = 0.5 + 0.002 * (pace + legacy_elo) + HEURISTIC_INJURY_COEFFICIENT * injury
        # adjust vs line as very rough anchor
        if total_line:
            base += HEURISTIC_TOTAL_LINE_COEFFICIENT * (pace - (total_line - 210))
        return max(MIN_PROBABILITY, min(MAX_PROBABILITY, base))

    def _model_total_prob(self, features: GameFeatures) -> Optional[float]:
        if not self.model:
            return None
        if self.feature_order is None:
            self.feature_order = list(features.raw.keys())
        # Convert to DataFrame for xgboost predict_proba
        data = {k: features.raw.get(k, 0) for k in self.feature_order}
        vec = pd.DataFrame([data])
        try:
            proba = self.model.predict_proba(vec)[0, 1]
            return float(proba)
        except Exception as e:
            self.event_logger.event('warning', f"predict_proba failed: {e}", category=classify_error(e))
            return None

    def predict_total(self, game: GameInfo, features: GameFeatures, market_yes_price: Optional[float] = None, market_no_price: Optional[float] = None) -> PredictionResult:
        """Generate total prediction with calibration and edge calculation.
        
        Edge Computation:
        1. Blend heuristic + model probability (weighted by config)
        2. Apply calibration if available (isotonic/platt fitted curve)
        3. Calculate effective cost: market_yes_price * (1 + KALSHI_BUY_COMMISSION)
        4. Edge = calibrated_probability - (effective_cost / 100)
        
        Args:
            game: GameInfo with game_id, teams, total_line
            features: GameFeatures dict containing model inputs
            market_yes_price: Kalshi YES price in cents (0-100)
            market_no_price: Kalshi NO price in cents (0-100)
        
        Returns:
            PredictionResult with probabilities, edge, and optional advanced model outputs
        """
        total_line = game.total_line
        heuristic_prob = self._heuristic_total_prob(features, total_line or 0)
        model_raw_prob = self._model_total_prob(features)
        
        # Blend heuristic and model probabilities using configured weights
        heuristic_weight = self.cfg["prediction"]["heuristic_weight"]
        model_weight = self.cfg["prediction"]["model_weight"] if model_raw_prob is not None else 0
        weight_sum = heuristic_weight + model_weight
        if weight_sum > 0:
            blended_prob = (heuristic_prob * heuristic_weight + (model_raw_prob or 0) * model_weight) / weight_sum
        else:
            blended_prob = heuristic_prob

        # Apply calibration transform if fitter is ready
        calibrated_prob = blended_prob
        applied_calibration = False
        if self.calibration_fitter.is_ready():
            calibrated_prob = self.calibration_fitter.apply(blended_prob)
            applied_calibration = True

        # Calculate effective cost accounting for Kalshi commission
        # Kalshi prices are in cents (0-100); buy commission increases effective cost
        vig_removed_prob = blended_prob  # fallback fair probability
        effective_yes_cost_cents = market_yes_price  # raw market price
        
        if market_yes_price is not None and market_no_price is not None:
            # Remove vig to get true fair probability from market prices
            vig_removed_prob = self._remove_vig_kalshi(market_yes_price, market_no_price)
            # Adjust for entry commission: buyer pays price * (1 + commission_rate)
            effective_yes_cost_cents = market_yes_price * (1.0 + KALSHI_BUY_COMMISSION)

        # Edge = calibrated model probability - effective market probability (cost / 100)
        # Positive edge means model believes YES is underpriced
        if effective_yes_cost_cents is not None:
            model_edge = calibrated_prob - (effective_yes_cost_cents / 100.0)
        else:
            # No market price available; compare calibrated vs vig-removed fair
            model_edge = calibrated_prob - vig_removed_prob

        result = PredictionResult(
            game_id=game.game_id,
            over_prob=blended_prob,
            calibrated_over_prob=calibrated_prob,
            total_line=total_line,
            model_edge=model_edge,
            heuristic_prob=heuristic_prob,
            model_prob=model_raw_prob or heuristic_prob,
            applied_calibration=applied_calibration,
            effective_yes_price=effective_yes_cost_cents if effective_yes_cost_cents else market_yes_price or 0,
        )
        # Attach advanced model probabilities if available and line provided
        if self._advanced_ready and total_line is not None:
            try:
                if self._poisson_model and hasattr(self._poisson_model, 'team_strengths'):
                    # Assume Poisson model already fitted externally
                    dist = self._poisson_model.predict_total_probability(
                        home_team=features.raw.get('home_team', ''),
                        away_team=features.raw.get('away_team', ''),
                        total_line=total_line,
                        pace=features.raw.get('expected_pace', 100),
                        n_simulations=self.cfg.get('advanced_models', {}).get('poisson_simulations', 5000)
                    )
                    result.poisson_over_prob = dist.get('over_prob')
                    result.poisson_expected_total = dist.get('expected_total')
                    result.poisson_total_std = dist.get('std')
                if self._bivariate_model and self._poisson_model and result.poisson_expected_total is not None:
                    # Provide synthetic expected spread from features if present
                    exp_spread = features.raw.get('projected_spread')
                    if exp_spread is not None:
                        joint = self._bivariate_model.predict_joint_probability(
                            spread_line=exp_spread,
                            total_line=total_line,
                            expected_spread=exp_spread,
                            expected_total=result.poisson_expected_total
                        )
                        result.bivariate_parlay_prob = joint.get('cover_and_over')
                        # Edge vs naive independence (placeholder calculation)
                        naive = (0.5 * result.poisson_over_prob) if result.poisson_over_prob else None
                        if naive and joint.get('cover_and_over') is not None:
                            result.bivariate_parlay_edge = joint['cover_and_over'] - naive
                if self._bayesian_model and hasattr(self._bayesian_model, 'player_effects'):
                    # Require roster info from features (if available)
                    home_roster = features.raw.get('home_roster') or []
                    away_roster = features.raw.get('away_roster') or []
                    if home_roster and away_roster:
                        bayes_pred = self._bayesian_model.predict_with_roster(
                            home_roster=home_roster,
                            away_roster=away_roster,
                            home_team=features.raw.get('home_team',''),
                            away_team=features.raw.get('away_team','')
                        )
                        result.bayesian_home_win_prob = bayes_pred.get('home_win_prob')
                        result.bayesian_expected_margin = bayes_pred.get('expected_margin')
            except Exception as e:
                self.event_logger.event('warning', f'Advanced attach failed: {e}', category=classify_error(e))
        # Emit structured log for prediction
        try:
            self.event_logger.event('info', 'prediction', category='prediction', game_id=game.game_id, context={
                'over_prob': result.over_prob,
                'calibrated': result.calibrated_over_prob,
                'edge': result.model_edge,
                'total_line': result.total_line,
                'heuristic': result.heuristic_prob,
                'model_prob': result.model_prob,
                'applied_calibration': result.applied_calibration,
                'effective_yes_price': result.effective_yes_price,
            })
        except Exception:
            pass
        return result

    def _remove_vig_kalshi(self, yes_price: float, no_price: float) -> float:
        # Kalshi prices are in cents (0-100). Normalize to remove over-round.
        # User note: Kalshi charges 2% commission on buy and sell, 0% on expiry.
        # For edge calculation, we account for entry cost (buy commission).
        total = yes_price + no_price
        if total <= 0:
            return 0.5
        fair_prob = yes_price / total
        return fair_prob

    def log_prediction(self, result: PredictionResult, features: GameFeatures) -> None:
        snapshot = json.dumps(features.raw)
        self.calibration_logger.log_prediction(result.game_id, result.over_prob, snapshot)

    def log_outcome(self, game_id: str, did_go_over: int) -> None:
        self.calibration_logger.log_outcome(game_id, did_go_over)


# --- CLV TRACKER ENGINE ---
import os
from datetime import datetime


class CLVTracker:
    """
    Closing Line Value (CLV) Tracker - The Ultimate Truth Serum for Betting Edge.
    
    CLV measures whether you're beating the market by comparing:
    - **Bet Odds**: The price you locked in when placing the bet
    - **Closing Odds**: The final market price just before game starts
    
    **Why CLV Matters:**
    - Positive CLV (+5%) = You beat the market (Sharp)
    - Negative CLV (-5%) = Market moved against you (Soft)
    - Consistent +CLV = Long-term profitability (even if you lose individual bets)
    - Negative CLV = Your wins are luck; casino takes it back eventually
    
    **Example:**
    You bet Knicks +110 (2.10 decimal) → Line closes at +100 (2.00)
    CLV = (2.10 / 2.00) - 1 = +5% → You captured value before sharp money moved it
    
    **Workflow:**
    1. `log_bet()` - Record bet immediately with your locked-in odds
    2. `update_closing_lines()` - After games start, backfill closing odds
    3. System auto-calculates CLV% and reports Sharp vs Soft status
    
    Attributes:
        log_path (str): CSV file path for bet tracking (default: logs/bet_tracker_with_clv.csv)
    
    CSV Schema:
        - Date: Timestamp of bet placement
        - Game_ID: Unique game identifier
        - Team: Team you bet on
        - Wager: Stake amount ($)
        - Bet_Odds: Decimal odds when you placed bet (e.g., 1.909 for -110)
        - Closing_Odds: Final market odds before game starts
        - CLV_Percent: (Bet_Odds / Closing_Odds - 1) * 100
        - Result: Win/Loss (populated after game settles)
        - Profit: Net profit/loss ($)
    
    Examples:
        >>> # Initialize tracker
        >>> clv = CLVTracker()
        >>> 
        >>> # Log bet when placing
        >>> clv.log_bet(
        ...     game_id='0022400123',
        ...     team='BOS Celtics',
        ...     wager=100.0,
        ...     bet_odds=1.909  # -110 American = 1.909 decimal
        ... )
        >>> 
        >>> # After games start, update with closing lines
        >>> closing_map = {
        ...     ('0022400123', 'BOS Celtics'): 1.833  # Closed at -120 (worse)
        ... }
        >>> clv.update_closing_lines(closing_map)
        >>> # Output: CLV = (1.909 / 1.833 - 1) = +4.1% (SHARP - you beat market)
    
    See Also:
        - `kelly_optimizer.py`: Uses CLV health factor in sizing (penalizes negative CLV)
        - `calibration_metrics.py`: Brier score + CLV = complete edge validation
    """
    def __init__(self, log_path: str = 'logs/bet_tracker_with_clv.csv') -> None:
        """
        Initialize CLV Tracker with specified log file path.
        
        Args:
            log_path (str): Path to CSV tracking file (creates if doesn't exist)
        """
        self.log_path = log_path
        self._init_log()

    def _init_log(self) -> None:
        """Creates the tracking file if it doesn't exist."""
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        if not os.path.exists(self.log_path):
            # We track Bet Odds (what you got) vs Closing Odds (Truth)
            df = pd.DataFrame(columns=[
                'Date', 'Game_ID', 'Team', 'Wager', 'Bet_Odds', 
                'Closing_Odds', 'CLV_Percent', 'Result', 'Profit'
            ])
            df.to_csv(self.log_path, index=False)

    def log_bet(self, game_id: str, team: str, wager: float, bet_odds: float) -> None:
        """
        Logs a bet immediately after placement. Closing odds are pending.
        
        **CRITICAL**: bet_odds must be in DECIMAL format, not American.
        - American -110 → Decimal 1.909
        - American +150 → Decimal 2.50
        - Conversion: Negative odds = (100 / abs(odds)) + 1
        -              Positive odds = (odds / 100) + 1
        
        Args:
            game_id (str): Unique game identifier (e.g., '0022400123')
            team (str): Team name you bet on (e.g., 'BOS Celtics')
            wager (float): Stake amount in dollars (e.g., 100.0)
            bet_odds (float): Decimal odds locked in (e.g., 1.909 for -110)
        
        Examples:
            >>> clv.log_bet('0022400123', 'LAL Lakers', 50.0, 2.10)  # +110 American
            📝 [CLV Tracker] Bet Logged: LAL Lakers @ 2.10
        """
        new_row = {
            'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Game_ID': game_id,
            'Team': team,
            'Wager': wager,
            'Bet_Odds': bet_odds,
            'Closing_Odds': None,  # Pending market close
            'CLV_Percent': 0.0,
            'Result': None,
            'Profit': 0.0
        }
        
        try:
            # Efficient append
            df = pd.DataFrame([new_row])
            df.to_csv(self.log_path, mode='a', header=not os.path.exists(self.log_path), index=False)
            print(f"   📝 [CLV Tracker] Bet Logged: {team} @ {bet_odds:.2f}")
        except Exception as e:
            print(f"   ❌ [CLV Tracker] Error logging bet: {e}")

    def update_closing_lines(self, closing_data_map: Dict[tuple, float]) -> None:
        """
        Updates past bets with final closing odds and calculates CLV.
        
        **CLV Formula**: (Bet_Odds / Closing_Odds) - 1
        - Positive CLV: You got better odds than closing (Sharp)
        - Negative CLV: Odds worsened after you bet (Soft/Public)
        
        Args:
            closing_data_map (Dict[tuple, float]): Map of (game_id, team_name) → closing decimal odds
                Example: {('0022400123', 'BOS Celtics'): 1.833}
        
        Examples:
            >>> # Collect closing lines from odds provider
            >>> closing_map = {
            ...     ('0022400123', 'BOS Celtics'): 1.833,  # Closed worse than your 1.909
            ...     ('0022400124', 'LAL Lakers'): 2.20     # Closed better than your 2.10
            ... }
            >>> clv.update_closing_lines(closing_map)
            🔄 Updating CLV for past bets...
            ✅ Updated BOS Celtics: Bet 1.909 vs Close 1.833 (CLV: +4.1%)
            ✅ Updated LAL Lakers: Bet 2.10 vs Close 2.20 (CLV: -4.5%)
            
            📊 CLV PERFORMANCE REPORT
               Total Bets Tracked: 2
               Average CLV: -0.2%
               Beating Market: 1/2 (50.0%)
               ⚠️ STATUS: SOFT (Market is moving against you)
        """
        if not os.path.exists(self.log_path):
            return
        
        df = pd.read_csv(self.log_path)
        updated = False
        
        print("\n🔄 Updating CLV for past bets...")
        
        for idx, row in df.iterrows():
            # Check for missing closing odds (handles both NaN and empty string)
            closing_val = row['Closing_Odds']
            if pd.isna(closing_val) or closing_val == '' or closing_val == 0.0:
                key = (str(row['Game_ID']), row['Team'])
                
                # Check if we have closing data for this specific bet
                if key in closing_data_map:
                    close = float(closing_data_map[key])
                    bet_price = float(row['Bet_Odds'])
                    
                    # CLV Formula: (Bet_Price / Closing_Price) - 1
                    clv = (bet_price / close) - 1
                    
                    df.at[idx, 'Closing_Odds'] = close
                    df.at[idx, 'CLV_Percent'] = round(clv * 100, 2)
                    updated = True
                    print(f"   ✅ Updated {row['Team']}: Bet {bet_price:.3f} vs Close {close:.3f} (CLV: {clv:+.1%})")
        
        if updated:
            df.to_csv(self.log_path, index=False)
            self._print_performance(df)
        else:
            print("   💤 No new closing lines found for pending bets.")

    def _print_performance(self, df: pd.DataFrame) -> None:
        """Internal report generator - shows Sharp vs Soft status."""
        closed = df.dropna(subset=['Closing_Odds'])
        if closed.empty:
            return
        
        avg_clv = closed['CLV_Percent'].mean()
        beating_market = len(closed[closed['CLV_Percent'] > 0])
        total = len(closed)
        
        print(f"\n📊 CLV PERFORMANCE REPORT")
        print(f"   Total Bets Tracked: {total}")
        print(f"   Average CLV: {avg_clv:+.2f}%")
        print(f"   Beating Market: {beating_market}/{total} ({beating_market/total:.1%})")
        
        if avg_clv > 0.0:
            print("   🚀 STATUS: SHARP (You are beating the closing line)")
        else:
            print("   ⚠️ STATUS: SOFT (Market is moving against you)")


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Backward compatibility alias
PredictionEngine = PredictionEngine

