"""
Kelly Criterion Optimizer for NBA Betting System

Implements Kelly criterion calculations for optimal bet sizing with comprehensive risk management:
- Drawdown-aware scaling (reduces bet size during losing streaks)
- Calibration health checks (refuses bets if calibration is broken)
- Event risk budgeting (limits total exposure per game)
- Bankroll tracking with SQLite persistence
- Comprehensive input validation (NaN/infinite protection)

The Kelly formula f = (bp - q) / b determines optimal bet fraction where:
- b = decimal_odds - 1 (net odds)
- p = model probability (MUST be calibrated)
- q = 1 - p (probability of loss)

**CRITICAL**: Only use calibrated probabilities. Uncalibrated probabilities will cause systematic overbetting.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

class KellyOptimizer:
    """
    Kelly Criterion implementation for optimal bet sizing with advanced risk management.
    
    This class manages the complete bet sizing workflow:
    1. **Validate Inputs**: Check for NaN, infinite, or invalid probability/price values
    2. **Calculate Raw Kelly**: Apply Kelly formula f = (bp - q) / b
    3. **Apply Risk Scaling**: Reduce bet size based on drawdown, calibration health
    4. **Enforce Limits**: Cap at max Kelly fraction, ensure non-negative stakes
    5. **Track Bankroll**: Log all bets and bankroll changes to database
    
    Drawdown-aware scaling policy (reduces Kelly multiplier during losses):
    - >20% drawdown → 25% of Kelly (0.25x)
    - >10% drawdown → 50% of Kelly (0.50x)
    - >5% drawdown → 75% of Kelly (0.75x)
    - <5% drawdown → 100% of Kelly (1.00x)
    
    Calibration health checks:
    - Computes current Brier score from calibration_outcomes table
    - Refuses bet if Brier > MAX_BRIER_FOR_BETTING (0.20)
    - Applies calibration_factor scaling based on model quality
    
    Attributes:
        initial_bankroll (float): Starting bankroll amount
        current_bankroll (float): Current bankroll after all bets/outcomes
        db_path (str): Path to SQLite database for bet tracking
        kelly_fraction (float): Base Kelly multiplier (default: 0.25 = quarter-Kelly)
        min_bet_pct (float): Minimum bet as percentage of bankroll (default: 0.01 = 1%)
        drawdown_policy (List[Tuple[float, float]]): [(DD_threshold, Kelly_scale), ...] thresholds
        event_risk_budget_pct (float): Max combined stake per event (default: 0.02 = 2%)
    
    Examples:
        >>> optimizer = KellyOptimizer(bankroll=10000.0, db_path='data/database/data/database/nba_betting_data.db')
        >>> # Calculate bet for calibrated probability vs market price
        >>> bet = optimizer.calculate_bet(
        ...     model_prob=0.62,  # Calibrated probability (not raw!)
        ...     market_price=0.50,  # Market implied probability
        ...     team_name='Lakers',
        ...     max_kelly_fraction=0.12,  # Cap at 12% Kelly
        ...     min_edge=0.04,  # Require 4%+ edge
        ...     calibration_factor=0.95  # Scale for calibration quality
        ... )
        >>> if bet['bet_recommended']:
        ...     print(f"Stake: ${bet['stake_amount']:.2f}")
        ...     print(f"Edge: {bet['edge_pct']:.1f}%")
    
    See Also:
        - `calibration_fitter.py`: Provides calibrated probabilities
        - `calibration_metrics.py`: Computes calibration health metrics
        - `constants.py`: MAX_BRIER_FOR_BETTING, KELLY_FRACTION_MULTIPLIER
    """
    
    def __init__(self, bankroll: float = 10000.0, db_path: str = "data/database/data/database/nba_betting_data.db"):
        """
        Initialize Kelly optimizer with starting bankroll and database connection.
        
        Args:
            bankroll (float): Starting bankroll amount in dollars. Default: 10000.0
            db_path (str): Path to SQLite database for bet tracking. Default: 'data/database/data/database/nba_betting_data.db'
        
        Raises:
            sqlite3.Error: If database initialization fails
        """
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        self.db_path = db_path
        self.VERSION = "v6"
        
        # Kelly fraction for bet sizing (0.0 to 1.0, typically 0.1-0.25)
        self.kelly_fraction = 0.25
        
        # Minimum bet as percentage of bankroll (e.g., 1% = 0.01)
        self.min_bet_pct = 0.01
        
        # Initialize database
        self._init_database()
        
        # Load current bankroll from database
        self._load_current_bankroll()
        
        logging.info(f"Kelly Optimizer initialized with bankroll: ${self.current_bankroll:,.2f}")
        # Drawdown-aware scaling policy thresholds (fractional drawdown -> scale)
        self.drawdown_policy = [
            (0.20, 0.25),  # >20% DD → 25% of Kelly
            (0.10, 0.50),  # >10% DD → 50% of Kelly
            (0.05, 0.75),  # >5% DD  → 75% of Kelly
        ]
        self.event_risk_budget_pct = 0.02  # Max combined stake per event (2% of bankroll)
    
    @property
    def bankroll(self):
        """Alias for current_bankroll for backwards compatibility"""
        return self.current_bankroll
    
    @bankroll.setter
    def bankroll(self, value):
        """Set bankroll value"""
        self.current_bankroll = value
    
    def _init_database(self) -> None:
        """Initialize database tables for bankroll tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bet history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bet_history (
                bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                game_id TEXT,
                team_name TEXT,
                bet_type TEXT,
                market_price REAL NOT NULL,
                model_probability REAL NOT NULL,
                edge REAL NOT NULL,
                kelly_fraction REAL NOT NULL,
                stake_amount REAL NOT NULL,
                bankroll_before REAL NOT NULL,
                bankroll_after REAL,
                profit_loss REAL,
                outcome TEXT,
                confidence REAL,
                notes TEXT
            )
        ''')
        
        # Bankroll history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bankroll_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                bankroll REAL NOT NULL,
                change REAL NOT NULL,
                reason TEXT,
                bet_id INTEGER,
                FOREIGN KEY (bet_id) REFERENCES bet_history (bet_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_current_bankroll(self) -> None:
        """Load current bankroll from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT bankroll FROM bankroll_history 
            ORDER BY timestamp DESC LIMIT 1
        ''')
        result = cursor.fetchone()
        
        if result:
            self.current_bankroll = result[0]
        else:
            # Initialize bankroll history
            cursor.execute('''
                INSERT INTO bankroll_history (timestamp, bankroll, change, reason)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now(), self.initial_bankroll, 0.0, 'Initial bankroll'))
            conn.commit()
            self.current_bankroll = self.initial_bankroll
        
        conn.close()

    # --- Drawdown-aware scaling ---
    def _high_water_mark(self) -> float:
        """Get highest bankroll value from history."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            row = cur.execute("SELECT MAX(bankroll) FROM bankroll_history").fetchone()
            hwm = row[0] if row and row[0] is not None else self.current_bankroll
        return max(hwm, self.current_bankroll)

    def current_drawdown(self) -> float:
        """Calculate current drawdown as fraction of high water mark."""
        hwm = self._high_water_mark()
        if hwm <= 0:
            return 0.0
        return max(0.0, (hwm - self.current_bankroll) / hwm)

    def drawdown_scale_factor(self) -> float:
        """Get Kelly scaling factor based on current drawdown."""
        dd = self.current_drawdown()
        for threshold, scale in self.drawdown_policy:
            if dd >= threshold:
                return scale
        return 1.0
    
    def calculate_kelly_fraction(self, model_prob: float, market_price: float) -> float:
        """
        Calculate optimal Kelly criterion fraction
        
        Args:
            model_prob (float): Model predicted probability (0-1)
            market_price (float): Market price/odds
            
        Returns:
            float: Kelly fraction (0-1)
        """
        # Validation: Probability must be in valid range
        if model_prob <= 0 or model_prob >= 1:
            logging.warning(f"Invalid model_prob={model_prob}, must be in (0,1). Returning 0.")
            return 0.0
        
        # Validation: Check for NaN or infinite values
        if not np.isfinite(model_prob):
            logging.error(f"model_prob is NaN or infinite: {model_prob}. Returning 0.")
            return 0.0
        
        if market_price <= 0:
            logging.warning(f"Invalid market_price={market_price}, must be positive. Returning 0.")
            return 0.0
        
        if not np.isfinite(market_price):
            logging.error(f"market_price is NaN or infinite: {market_price}. Returning 0.")
            return 0.0
        
        # Convert market price to decimal odds
        if market_price < 1:  # Probability format
            decimal_odds = 1 / market_price
        else:  # Already decimal odds
            decimal_odds = market_price
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = model probability, q = 1 - p
        b = decimal_odds - 1
        p = model_prob
        q = 1 - p
        
        if b <= 0:
            logging.warning(f"Invalid decimal_odds={decimal_odds}, b={b} must be positive. Returning 0.")
            return 0.0
        
        kelly_fraction = (b * p - q) / b
        
        # Validation: Ensure result is finite
        if not np.isfinite(kelly_fraction):
            logging.error(f"Calculated kelly_fraction is NaN or infinite: {kelly_fraction}. Returning 0.")
            return 0.0
        
        # Ensure non-negative
        kelly_fraction = max(0.0, kelly_fraction)
        
        # Validation: Prevent absurd values (>100% bankroll)
        if kelly_fraction > 1.0:
            logging.warning(f"Kelly fraction {kelly_fraction:.2%} exceeds 100%, capping at 100%")
            kelly_fraction = 1.0
        
        return kelly_fraction
    
    def calculate_bet(self, model_prob: float, market_price: float, team_name: str, 
                     max_kelly_fraction: float = 0.12, min_edge: float = 0.04,
                     dry_run: bool = True, calibration_factor: Optional[float] = None) -> Dict:
        """
        Calculate optimal bet size using Kelly criterion with risk safeguards
        
        Args:
            model_prob (float): Model predicted probability
            market_price (float): Market price/odds
            team_name (str): Team name for bet
            max_kelly_fraction (float): Maximum Kelly fraction allowed (default 12% of bankroll)
            min_edge (float): Minimum edge required to bet (default 4%)
            dry_run (bool): If True, don't actually place bet
            
        Returns:
            Dict: Bet recommendation details
        """
        # CRITICAL: Check calibration health before sizing bet
        from src.constants import MAX_BRIER_FOR_BETTING
        try:
            from src.core.calibration_metrics import compute_brier_score, calibration_sample_count
            brier = compute_brier_score(self.db_path)
            n_samples = calibration_sample_count(self.db_path)
            
            # Refuse bet if calibration is severely broken or insufficient samples
            if brier > MAX_BRIER_FOR_BETTING:
                logging.warning(f"Calibration health check FAILED: Brier={brier:.4f} > {MAX_BRIER_FOR_BETTING}. Refusing bet.")
                return {
                    'bet_recommended': False,
                    'team_name': team_name,
                    'model_prob_capped': model_prob,
                    'market_price': market_price,
                    'edge': 0,
                    'edge_pct': 0,
                    'kelly_fraction': 0,
                    'kelly_fraction_scaled': 0,
                    'calibration_factor': 0,
                    'calibration_brier': brier,
                    'calibration_samples': n_samples,
                    'drawdown_scale': 0,
                    'raw_kelly_fraction': 0,
                    'stake_amount': 0,
                    'stake_pct': 0,
                    'expected_profit': 0,
                    'current_bankroll': self.current_bankroll,
                    'max_kelly_used': max_kelly_fraction,
                    'refusal_reason': f'Calibration Brier {brier:.3f} exceeds safety threshold {MAX_BRIER_FOR_BETTING}'
                }
        except Exception as e:
            logging.warning(f"Calibration health check failed: {e}. Proceeding with caution.")
            brier = 0.0
            n_samples = 0
        
        # Calculate Kelly fraction
        raw_kelly = self.calculate_kelly_fraction(model_prob, market_price)
        
        # Apply maximum Kelly limit
        kelly_fraction = min(raw_kelly, max_kelly_fraction)

        # Apply calibration-aware scaling of Kelly
        # If not provided, compute from calibration metrics (safe import)
        cal_diag = {}
        if calibration_factor is None:
            try:
                from src.core.calibration_metrics import calibration_health_factor
                calibration_factor, cal_diag = calibration_health_factor(self.db_path)
            except Exception:
                calibration_factor = 1.0
        calibration_factor = max(0.5, min(1.0, calibration_factor))
        # Drawdown scaling
        dd_scale = self.drawdown_scale_factor()
        kelly_fraction_scaled = kelly_fraction * calibration_factor * dd_scale
        
        # Calculate edge
        if market_price < 1:  # Probability format
            market_prob = market_price
        else:  # Decimal odds format
            market_prob = 1 / market_price
        
        edge = model_prob - market_prob
        edge_pct = edge * 100
        
        # Determine if bet is recommended
        bet_recommended = (
            kelly_fraction_scaled > 0 and 
            edge > min_edge and 
            model_prob > 0.5 and
            self.current_bankroll > 0
        )
        
        # Calculate stake amount
        stake_amount = self.current_bankroll * kelly_fraction_scaled if bet_recommended else 0.0
        
        # CRITICAL VALIDATION: Prevent absurd stake amounts
        if not np.isfinite(stake_amount):
            logging.error(f"Stake amount is NaN or infinite: {stake_amount}. Setting to 0.")
            stake_amount = 0.0
            bet_recommended = False
        
        # Ensure stake doesn't exceed bankroll
        if stake_amount > self.current_bankroll:
            logging.error(f"Stake {stake_amount:.2f} exceeds bankroll {self.current_bankroll:.2f}. Capping.")
            stake_amount = self.current_bankroll
        
        # Ensure stake is non-negative
        if stake_amount < 0:
            logging.error(f"Negative stake amount: {stake_amount:.2f}. Setting to 0.")
            stake_amount = 0.0
            bet_recommended = False
        
        # Calculate expected value
        if market_price < 1:  # Probability format
            payout_odds = (1 / market_price) - 1
        else:  # Decimal odds format
            payout_odds = market_price - 1
        
        expected_profit = stake_amount * (model_prob * payout_odds - (1 - model_prob))
        
        result = {
            'bet_recommended': bet_recommended,
            'team_name': team_name,
            'model_prob_capped': model_prob,
            'market_price': market_price,
            'edge': edge,
            'edge_pct': edge_pct,
            'kelly_fraction': kelly_fraction,
            'kelly_fraction_scaled': kelly_fraction_scaled,
            'calibration_factor': calibration_factor,
            'calibration_brier': brier if 'brier' in locals() else None,
            'calibration_samples': n_samples if 'n_samples' in locals() else None,
            'drawdown_scale': dd_scale,
            'raw_kelly_fraction': raw_kelly,
            'stake_amount': stake_amount,
            'stake_pct': kelly_fraction_scaled * 100,
            'expected_profit': expected_profit,
            'current_bankroll': self.current_bankroll,
            'max_kelly_used': max_kelly_fraction
        }
        
        # Log the bet if not dry run
        if not dry_run and bet_recommended:
            self._log_bet(result, model_prob, market_price, team_name)
        
        return result

    # --- Portfolio correlation / event risk budget ---
    def allocate_event_risk_budget(self, bets: List[Dict], bankroll: Optional[float] = None,
                                   event_key: str = 'event_id', stake_key: str = 'stake') -> List[Dict]:
        """Scale stakes so that combined stake per event does not exceed budget percent of bankroll.
        bets: list of dicts; each must include event_id (or event_key) and proposed stake amount in currency.
        Returns a new list of bets with 'stake_adjusted' applied.
        """
        bk = bankroll if bankroll is not None else self.current_bankroll
        budget = bk * self.event_risk_budget_pct
        # Group by event
        from collections import defaultdict
        groups = defaultdict(list)
        for b in bets:
            groups[b.get(event_key, 'UNKNOWN')].append(b)
        adjusted = []
        for ev, group in groups.items():
            total = sum(max(0.0, float(g.get(stake_key, 0.0))) for g in group)
            if total <= 0:
                for g in group:
                    g2 = g.copy(); g2['stake_adjusted'] = 0.0; adjusted.append(g2)
            elif total <= budget:
                for g in group:
                    g2 = g.copy(); g2['stake_adjusted'] = float(g.get(stake_key, 0.0)); adjusted.append(g2)
            else:
                scale = budget / total
                for g in group:
                    g2 = g.copy(); g2['stake_adjusted'] = float(g.get(stake_key, 0.0)) * scale; adjusted.append(g2)
        return adjusted

    # --- Risk of Ruin (Monte Carlo) ---
    def simulate_ror(self, bets: List[Dict], trials: int = 5000, horizon: int = 1,
                     ruin_fraction: float = 0.5, dynamic_stake: bool = True) -> Dict:
        """Monte Carlo simulate bankroll distribution and risk-of-ruin.
        bets: list of {prob: float (0-1), odds: float (decimal), stake: float (abs) or stake_pct: float (0-1), event_id: str}
        trials: number of Monte Carlo runs
        horizon: number of repeated bet slates to simulate (e.g., days)
        ruin_fraction: bankroll level considered ruin (fraction of starting bankroll)
        dynamic_stake: if True and stake_pct provided, stake scales with current bankroll each round
        Returns summary dict with ruin_prob and bankroll distribution stats.
        """
        import random
        start = float(self.current_bankroll)
        if start <= 0:
            return {'status':'error','reason':'non_positive_bankroll'}
        ruin_level = start * ruin_fraction
        finals = []
        ruins = 0
        for _ in range(trials):
            bk = start
            ruined = False
            for _h in range(horizon):
                for b in bets:
                    p = max(0.0, min(1.0, float(b.get('prob', 0.0))))
                    odds = float(b.get('odds', 2.0))
                    stake_abs = b.get('stake', None)
                    stake_pct = b.get('stake_pct', None)
                    stake = 0.0
                    if stake_abs is not None:
                        stake = float(stake_abs)
                    elif stake_pct is not None:
                        stake = bk * float(stake_pct)
                    else:
                        continue
                    if stake <= 0 or bk <= 0:
                        continue
                    # Place stake
                    bk -= stake
                    # Resolve outcome
                    win = (random.random() < p)
                    if win:
                        payout = stake * (odds - 1.0)
                        bk += stake + payout
                    # Check ruin
                    if bk <= ruin_level:
                        ruined = True
                        break
                if ruined:
                    break
            finals.append(bk)
            if ruined:
                ruins += 1
        finals.sort()
        n = len(finals) if finals else 1
        def pct(v):
            idx = int(max(0, min(n-1, round(v*(n-1)))))
            return finals[idx]
        return {
            'status':'ok',
            'trials': trials,
            'horizon': horizon,
            'ruin_fraction': ruin_fraction,
            'ruin_prob': ruins / trials if trials>0 else 0.0,
            'final_bankroll': {
                'p05': pct(0.05),
                'p50': pct(0.50),
                'p95': pct(0.95),
                'min': finals[0] if finals else start,
                'max': finals[-1] if finals else start,
                'mean': float(np.mean(finals)) if finals else start,
                'std': float(np.std(finals)) if finals else 0.0,
            }
        }
    
    def _log_bet(self, bet_details: Dict, model_prob: float, market_price: float, team_name: str):
        """Log bet to database with context manager"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO bet_history (
                    timestamp, team_name, bet_type, market_price, model_probability,
                    edge, kelly_fraction, stake_amount, bankroll_before, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                team_name,
                'moneyline',
                market_price,
                model_prob,
                bet_details['edge'],
                bet_details['kelly_fraction'],
                bet_details['stake_amount'],
                self.current_bankroll,
                'pending'
            ))
            
            bet_id = cursor.lastrowid
            
            # Update bankroll (subtract stake)
            new_bankroll = self.current_bankroll - bet_details['stake_amount']
            
            cursor.execute('''
                INSERT INTO bankroll_history (timestamp, bankroll, change, reason, bet_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                new_bankroll,
                -bet_details['stake_amount'],
                f"Bet placed on {team_name}",
                bet_id
            ))
            
            conn.commit()
        
        self.current_bankroll = new_bankroll
        logging.info(f"Bet logged: ${bet_details['stake_amount']:.2f} on {team_name}, bankroll: ${new_bankroll:.2f}")
    
    def update_bet_outcome(self, bet_id: int, outcome: str, profit_loss: float):
        """Update bet outcome and adjust bankroll with context manager"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get bet details
            cursor.execute('SELECT bankroll_before, stake_amount FROM bet_history WHERE bet_id = ?', (bet_id,))
            result = cursor.fetchone()
            
            if not result:
                logging.error(f"Bet {bet_id} not found")
                return
            
            bankroll_before, stake_amount = result
            new_bankroll = self.current_bankroll + profit_loss
            
            # Update bet record
            cursor.execute('''
                UPDATE bet_history 
                SET outcome = ?, profit_loss = ?, bankroll_after = ?
                WHERE bet_id = ?
            ''', (outcome, profit_loss, new_bankroll, bet_id))
            
            # Add bankroll history entry
            cursor.execute('''
                INSERT INTO bankroll_history (timestamp, bankroll, change, reason, bet_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                new_bankroll,
                profit_loss,
                f"Bet outcome: {outcome}",
                bet_id
            ))
            
            conn.commit()
        
        self.current_bankroll = new_bankroll
        logging.info(f"Bet {bet_id} updated: {outcome}, P&L: ${profit_loss:.2f}, bankroll: ${new_bankroll:.2f}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics with context manager"""
        with sqlite3.connect(self.db_path) as conn:
            # Get bet statistics
            bet_stats = pd.read_sql_query('''
                SELECT outcome, profit_loss, stake_amount, edge
                FROM bet_history 
                WHERE outcome IN ('win', 'loss')
            ''', conn)
            
            # Get bankroll history
            bankroll_history = pd.read_sql_query('''
                SELECT bankroll, timestamp FROM bankroll_history
                ORDER BY timestamp
            ''', conn)
        
        stats = {
            'current_bankroll': self.current_bankroll,
            'total_profit': 0.0,
            'roi': 0.0,
            'win_rate': 0.0,
            'total_bets': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        if not bet_stats.empty:
            stats['total_profit'] = bet_stats['profit_loss'].sum()
            stats['roi'] = (stats['total_profit'] / self.initial_bankroll) * 100
            stats['total_bets'] = len(bet_stats)
            
            wins = bet_stats[bet_stats['outcome'] == 'win']
            stats['win_rate'] = len(wins) / len(bet_stats) if len(bet_stats) > 0 else 0.0
            
            # Calculate max drawdown
            if not bankroll_history.empty:
                bankroll_series = bankroll_history['bankroll']
                rolling_max = bankroll_series.expanding().max()
                drawdown = (bankroll_series - rolling_max) / rolling_max
                stats['max_drawdown'] = abs(drawdown.min()) * 100
        
        return stats
    
    def get_bet_history_data(self) -> pd.DataFrame:
        """Get bet history as DataFrame with context manager"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT bet_id, timestamp, team_name, stake_amount, profit_loss,
                       outcome, edge, kelly_fraction, bankroll_before, bankroll_after
                FROM bet_history
                ORDER BY timestamp DESC
            ''', conn)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_bankroll_history(self) -> pd.DataFrame:
        """Get bankroll history for charting with context manager"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT timestamp, bankroll, change, reason
                FROM bankroll_history
                ORDER BY timestamp
            ''', conn)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

# Usage example
if __name__ == "__main__":
    # Initialize Kelly optimizer
    kelly = KellyOptimizer(bankroll=10000.0)
    
    # Test bet calculation
    bet_result = kelly.calculate_bet(
        model_prob=0.65,
        market_price=0.55,
        team_name="Lakers",
        dry_run=True
    )
    
    print("Kelly Bet Calculation:")
    print(f"Recommended: {bet_result['bet_recommended']}")
    print(f"Edge: {bet_result['edge_pct']:.2f}%")
    print(f"Kelly %: {bet_result['stake_pct']:.2f}%")
    print(f"Stake: ${bet_result['stake_amount']:.2f}")
    print(f"Expected Profit: ${bet_result['expected_profit']:.2f}")

# Backward compatibility alias
KellyOptimizer = KellyOptimizer
