"""
Kalshi API Client for NBA Betting
Handles authentication, market data retrieval, and automated betting
with comprehensive error handling and rate limiting
"""

import requests
import json
import time
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from src.logger_setup import get_structured_adapter, classify_error

@dataclass
class BettingOpportunity:
    """Data class for betting opportunities"""
    market_id: str
    event_title: str
    team_a: str
    team_b: str
    bet_type: str  # 'spread', 'moneyline', 'total'
    line: float
    yes_price: float
    no_price: float
    predicted_probability: float
    kelly_fraction: float
    recommended_bet_size: float
    expected_value: float
    confidence: float

class KalshiClient:
    def __init__(self, api_key: str, api_secret: str, environment: str = 'demo', *, auth_on_init: bool = True, request_timeout: int = 10, max_retries: int = 2):
        """
        Initialize Kalshi API client
        
        Args:
            api_key (str): Kalshi API key
            api_secret (str): Kalshi API secret
            environment (str): 'demo' or 'prod'
        """
        self.api_key = api_key
        # CRITICAL: Decode \\n escape sequences from JSON
        # JSON stores "\\n" as literal backslash-n, but we need actual newlines for PEM parsing
        if api_secret and '\\n' in api_secret:
            api_secret = api_secret.replace('\\n', '\n')
        self.api_secret = api_secret
        self.environment = environment
        
        # Set API base URL based on environment
        if environment == 'prod':
            self.base_url = 'https://api.elections.kalshi.com'
        else:
            self.base_url = 'https://demo-api.kalshi.co'
        self.api_path = '/trade-api/v2'
        
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        # timeouts and retries
        self.request_timeout = max(3, int(request_timeout))
        self.max_retries = max(0, int(max_retries))
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second max
        
        # Authentication
        self.access_token = None
        self.token_expires_at = None
        
        # Market data cache
        self.markets_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Structured logger
        self.event_logger = get_structured_adapter(component='kalshi', prediction_version='v5.0')
        # Initialize authentication (optionally non-blocking for UI callers)
        if auth_on_init:
            self.authenticate()
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _create_signature(self, method: str, path: str, body: str = "") -> str:
        """Create RSA-PSS signature for Kalshi API authentication"""
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.backends import default_backend
        
        timestamp = str(int(time.time() * 1000))
        
        # Create message to sign (Kalshi format: timestamp + method + path without query params)
        path_parts = path.split('?')
        message = timestamp + method.upper() + path_parts[0]
        
        try:
            # Parse the private key (handle both string and bytes)
            key_data = self.api_secret
            if isinstance(key_data, str):
                key_data = key_data.encode('utf-8')
            private_key = serialization.load_pem_private_key(
                key_data,
                password=None,
                backend=default_backend()
            )
            
            # Sign using RSA-PSS (Kalshi's required method)
            signature = private_key.sign(
                message.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            return timestamp, signature_b64
            
        except Exception as e:
            self.event_logger.event('error', f"Error creating Kalshi signature: {e}", category=classify_error(e))
            raise
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None) -> Dict:
        """
        Make authenticated API request
        
        Args:
            method (str): HTTP method
            endpoint (str): API endpoint
            data (dict): Request body data
            params (dict): Query parameters
            
        Returns:
            dict: API response
        """
        self._rate_limit()
        
        # Construct full URL
        if not endpoint.startswith('/trade-api/v2'):
            endpoint = f"{self.api_path}{endpoint}"
        url = f"{self.base_url}{endpoint}"
        
        # Prepare request body
        body = json.dumps(data) if data else ""
        
        # Create signature
        timestamp, signature = self._create_signature(method, endpoint, body)
        
        # Set headers (Kalshi uses header-based auth, no separate login)
        headers = {
            'Content-Type': 'application/json',
            'KALSHI-ACCESS-KEY': self.api_key,
            'KALSHI-ACCESS-SIGNATURE': signature,
            'KALSHI-ACCESS-TIMESTAMP': timestamp
        }
        
        attempt = 0
        last_err = None
        while attempt <= self.max_retries:
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_err = e
                self.event_logger.event('warning', f"API request failed (attempt {attempt+1}/{self.max_retries+1}): {e}", category='network')
                if hasattr(e, 'response') and e.response is not None:
                    self.event_logger.event('warning', f"Response: {e.response.text}", category='network')
                attempt += 1
                time.sleep(min(1.0 * attempt, 3.0))
        # After retries
        self.event_logger.event('error', f"API request failed after retries: {last_err}", category='network')
        raise last_err
    
    def authenticate(self) -> bool:
        """
        Test Kalshi API authentication by getting exchange status
        
        Returns:
            bool: Success status
        """
        try:
            response = self._make_request('GET', '/exchange/status')
            if response.get('exchange_active') is not None:
                self.event_logger.event('info', "Successfully authenticated with Kalshi API", category='network')
                return True
            self.event_logger.event('error', "Unexpected response from Kalshi API", category='network')
            return False
        except Exception as e:
            self.event_logger.event('warning', f"Authentication failed (non-fatal): {e}", category=classify_error(e))
            return False

    def ping(self) -> bool:
        """Lightweight connectivity check without raising exceptions."""
        try:
            self._make_request('GET', '/exchange/status')
            return True
        except Exception:
            return False
    
    def refresh_token_if_needed(self):
        """Refresh access token if expired"""
        if (self.token_expires_at is None or 
            datetime.now() >= self.token_expires_at - timedelta(minutes=5)):
            self.authenticate()
    
    def get_account_info(self) -> Dict:
        """
        Get account information including balance
        
        Returns:
            dict: Account information
        """
        self.refresh_token_if_needed()
        
        try:
            response = self._make_request('GET', '/portfolio/balance')
            return response
        except Exception as e:
            self.event_logger.event('error', f"Failed to get account info: {e}", category=classify_error(e))
            return {}
    
    def get_nba_markets(self, status: str = 'open') -> List[Dict]:
        """
        Get NBA betting markets
        
        Args:
            status (str): Market status ('open', 'closed', 'settled')
            
        Returns:
            list: NBA markets
        """
        self.refresh_token_if_needed()
        
        # Check cache first
        cache_key = f"nba_markets_{status}"
        if (cache_key in self.markets_cache and 
            time.time() - self.markets_cache[cache_key].get('timestamp', 0) < self.cache_expiry):
            return self.markets_cache[cache_key]['data']
        
        try:
            params = {
                'event_ticker': 'NBA',
                'status': status,
                'limit': 200
            }
            
            response = self._make_request('GET', '/events', params=params)
            markets = response.get('events', [])
            
            # Cache results
            self.markets_cache[cache_key] = {
                'data': markets,
                'timestamp': time.time()
            }
            
            self.event_logger.event('info', f"Retrieved {len(markets)} NBA markets", category='network')
            return markets
            
        except Exception as e:
            logger.error(f"Failed to get NBA markets: {e}")
            return []
    
    def get_market_details(self, market_id: str) -> Dict:
        """
        Get detailed market information
        
        Args:
            market_id (str): Market identifier
            
        Returns:
            dict: Market details
        """
        self.refresh_token_if_needed()
        
        try:
            response = self._make_request('GET', f'/events/{market_id}')
            return response.get('event', {})
        except Exception as e:
            logger.error(f"Failed to get market details for {market_id}: {e}")
            return {}
    
    def get_market_orderbook(self, market_id: str) -> Dict:
        """
        Get market orderbook with current prices
        
        Args:
            market_id (str): Market identifier
            
        Returns:
            dict: Orderbook data
        """
        self.refresh_token_if_needed()
        
        try:
            response = self._make_request('GET', f'/events/{market_id}/markets')
            markets = response.get('markets', [])
            
            if markets:
                market = markets[0]
                orderbook_response = self._make_request('GET', f'/markets/{market["ticker"]}/orderbook')
                return orderbook_response
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get orderbook for {market_id}: {e}")
            return {}
    
    def place_bet(self, market_ticker: str, side: str, quantity: int, 
                  price: int, bet_type: str = 'limit') -> Dict:
        """
        Place a bet on Kalshi
        
        Args:
            market_ticker (str): Market ticker symbol
            side (str): 'yes' or 'no'
            quantity (int): Number of contracts (in cents)
            price (int): Price per contract (in cents, 1-99)
            bet_type (str): Order type ('limit', 'market')
            
        Returns:
            dict: Order response
        """
        self.refresh_token_if_needed()
        
        order_data = {
            'ticker': market_ticker,
            'client_order_id': f"nba_bet_{int(time.time())}",
            'side': side,
            'action': 'buy',
            'count': quantity,
            'type': bet_type
        }
        
        if bet_type == 'limit':
            order_data['yes_price'] = price if side == 'yes' else None
            order_data['no_price'] = price if side == 'no' else None
        
        try:
            response = self._make_request('POST', '/portfolio/orders', data=order_data)
            logger.info(f"Placed bet: {market_ticker} {side} {quantity} @ {price}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to place bet: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions
        
        Returns:
            list: Current positions
        """
        self.refresh_token_if_needed()
        
        try:
            response = self._make_request('GET', '/portfolio/positions')
            return response.get('market_positions', [])
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_order_history(self, limit: int = 100) -> List[Dict]:
        """
        Get order history
        
        Args:
            limit (int): Maximum number of orders to retrieve
            
        Returns:
            list: Order history
        """
        self.refresh_token_if_needed()
        
        try:
            params = {'limit': limit}
            response = self._make_request('GET', '/portfolio/orders', params=params)
            return response.get('orders', [])
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order
        
        Args:
            order_id (str): Order identifier
            
        Returns:
            bool: Success status
        """
        self.refresh_token_if_needed()
        
        try:
            self._make_request('DELETE', f'/portfolio/orders/{order_id}')
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def calculate_kelly_bet_size(self, predicted_prob: float, market_price: float, 
                               bankroll: float, max_fraction: float = 0.02) -> float:
        """
        Calculate optimal bet size using Kelly criterion
        
        Args:
            predicted_prob (float): Model predicted probability (0-1)
            market_price (float): Market price in cents (1-99)
            bankroll (float): Current bankroll
            max_fraction (float): Maximum fraction of bankroll to bet
            
        Returns:
            float: Recommended bet size
        """
        # Convert market price to probability
        market_prob = market_price / 100
        
        # Calculate Kelly fraction
        if market_prob >= predicted_prob:
            return 0  # No edge
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = predicted prob, q = 1-p
        b = (1 - market_prob) / market_prob  # Odds
        p = predicted_prob
        q = 1 - predicted_prob
        
        kelly_fraction = (b * p - q) / b
        
        # Apply maximum fraction limit
        kelly_fraction = min(kelly_fraction, max_fraction)
        kelly_fraction = max(kelly_fraction, 0)  # No negative bets
        
        return bankroll * kelly_fraction
    
    def find_betting_opportunities(self, predictions: List[Dict], 
                                 min_edge: float = 0.05,
                                 min_confidence: float = 0.60) -> List[BettingOpportunity]:
        """
        Find betting opportunities based on model predictions
        
        Args:
            predictions (list): Model predictions with probabilities
            min_edge (float): Minimum edge required
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            list: Betting opportunities
        """
        opportunities = []
        nba_markets = self.get_nba_markets()
        
        for prediction in predictions:
            if prediction.get('confidence', 0) < min_confidence:
                continue
            
            # Find matching market
            matching_market = None
            for market in nba_markets:
                if (prediction['home_team'] in market.get('title', '') and
                    prediction['away_team'] in market.get('title', '')):
                    matching_market = market
                    break
            
            if not matching_market:
                continue
            
            # Get market prices
            orderbook = self.get_market_orderbook(matching_market['event_ticker'])
            if not orderbook:
                continue
            
            best_yes = orderbook.get('yes', [{}])[0].get('price', 50)
            best_no = orderbook.get('no', [{}])[0].get('price', 50)
            
            # Calculate edges for different bet types
            for bet_type in ['moneyline', 'spread', 'total']:
                if bet_type not in prediction:
                    continue
                
                pred_prob = prediction[bet_type]['probability']
                market_prob = best_yes / 100
                
                edge = pred_prob - market_prob
                
                if edge >= min_edge:
                    # Calculate Kelly bet size
                    account_info = self.get_account_info()
                    bankroll = account_info.get('portfolio_balance', 10000) / 100  # Convert from cents
                    
                    kelly_size = self.calculate_kelly_bet_size(pred_prob, best_yes, bankroll)
                    
                    opportunity = BettingOpportunity(
                        market_id=matching_market['event_ticker'],
                        event_title=matching_market['title'],
                        team_a=prediction['home_team'],
                        team_b=prediction['away_team'],
                        bet_type=bet_type,
                        line=prediction[bet_type].get('line', 0),
                        yes_price=best_yes,
                        no_price=best_no,
                        predicted_probability=pred_prob,
                        kelly_fraction=kelly_size / bankroll if bankroll > 0 else 0,
                        recommended_bet_size=kelly_size,
                        expected_value=edge * kelly_size,
                        confidence=prediction['confidence']
                    )
                    
                    opportunities.append(opportunity)
        
        # Sort by expected value
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        
        return opportunities
    
    def execute_betting_strategy(self, opportunities: List[BettingOpportunity],
                               max_simultaneous_bets: int = 5,
                               dry_run: bool = True) -> List[Dict]:
        """
        Execute betting strategy based on opportunities
        
        Args:
            opportunities (list): Betting opportunities
            max_simultaneous_bets (int): Maximum simultaneous positions
            dry_run (bool): If True, only simulate bets
            
        Returns:
            list: Executed trades
        """
        executed_trades = []
        current_positions = len(self.get_positions())
        
        for opportunity in opportunities[:max_simultaneous_bets - current_positions]:
            if opportunity.recommended_bet_size < 50:  # Minimum $0.50 bet
                continue
            
            # Convert bet size to contracts (cents)
            bet_size_cents = int(opportunity.recommended_bet_size * 100)
            price_cents = int(opportunity.yes_price)
            
            if dry_run:
                trade_result = {
                    'market_id': opportunity.market_id,
                    'event_title': opportunity.event_title,
                    'bet_type': opportunity.bet_type,
                    'side': 'yes',
                    'quantity': bet_size_cents,
                    'price': price_cents,
                    'expected_value': opportunity.expected_value,
                    'status': 'simulated',
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(f"SIMULATED BET: {trade_result}")
            else:
                # Execute real bet
                trade_result = self.place_bet(
                    market_ticker=opportunity.market_id,
                    side='yes',
                    quantity=bet_size_cents,
                    price=price_cents
                )
                
                trade_result.update({
                    'expected_value': opportunity.expected_value,
                    'bet_type': opportunity.bet_type,
                    'timestamp': datetime.now().isoformat()
                })
            
            executed_trades.append(trade_result)
            
            # Small delay between bets
            time.sleep(1)
        
        return executed_trades
    
    def get_game_markets(self, home_team: str, away_team: str, game_date: str = None) -> Optional[Dict[str, Any]]:
        """Retrieve Kalshi moneyline markets for NBA game using /events and /markets endpoints."""
        self.refresh_token_if_needed()
        try:
            # Get KXNBAGAME events
            events_resp = self._make_request('GET', '/events', params={'limit': 200, 'series_ticker': 'KXNBAGAME'})
            events = events_resp.get('events', [])

            team_name_map = {
                'LAL': ['Lakers', 'Los Angeles L', 'LA L'],
                'LAC': ['Clippers', 'Los Angeles C', 'LA C'],
                'BOS': ['Celtics', 'Boston'],
                'GSW': ['Warriors', 'Golden State'],
                'MIA': ['Heat', 'Miami'],
                'PHX': ['Suns', 'Phoenix'],
                'DEN': ['Nuggets', 'Denver'],
                'MIL': ['Bucks', 'Milwaukee'],
                'PHI': ['Philadelphia', '76ers', 'Sixers'],
                'BKN': ['Nets', 'Brooklyn'],
                'DAL': ['Mavericks', 'Dallas', 'Mavs'],
                'MEM': ['Grizzlies', 'Memphis'],
                'SAC': ['Kings', 'Sacramento'],
                'NYK': ['Knicks', 'New York'],
                'CLE': ['Cavaliers', 'Cleveland', 'Cavs'],
                'MIN': ['Timberwolves', 'Minnesota', 'Wolves'],
                'NOP': ['Pelicans', 'New Orleans'],
                'OKC': ['Thunder', 'Oklahoma City'],
                'ATL': ['Hawks', 'Atlanta'],
                'CHI': ['Bulls', 'Chicago'],
                'TOR': ['Raptors', 'Toronto'],
                'WAS': ['Wizards', 'Washington'],
                'IND': ['Pacers', 'Indiana'],
                'CHA': ['Hornets', 'Charlotte'],
                'DET': ['Pistons', 'Detroit'],
                'HOU': ['Rockets', 'Houston'],
                'ORL': ['Magic', 'Orlando'],
                'POR': ['Trail Blazers', 'Portland', 'Blazers'],
                'SAS': ['Spurs', 'San Antonio'],
                'UTA': ['Jazz', 'Utah']
            }

            home_names = team_name_map.get(home_team.upper(), [home_team])
            away_names = team_name_map.get(away_team.upper(), [away_team])

            # Find matching event by checking if both teams are in title
            matching_event = None
            for event in events:
                title = (event.get('title') or '').lower()
                if any(n.lower() in title for n in home_names) and any(n.lower() in title for n in away_names):
                    matching_event = event
                    break

            if not matching_event:
                self.event_logger.event('info', f'No event found for {home_team} vs {away_team}', category='market_data')
                return None

            # Get markets for this event using event_ticker parameter
            event_ticker = matching_event.get('event_ticker')
            markets_resp = self._make_request('GET', '/markets', params={'event_ticker': event_ticker})
            markets = markets_resp.get('markets', [])

            if not markets:
                self.event_logger.event('info', f'No markets for event {event_ticker}', category='market_data')
                return None

            result: Dict[str, Any] = {}
            for m in markets:
                ticker = m.get('ticker', '').upper()
                yes_price = m.get('yes_bid', 50)
                no_price = m.get('no_bid', 50)

                # Kalshi NBA markets are moneyline only
                # Ticker format: KXNBAGAME-25NOV21PORGSW-POR (home) or -GSW (away)
                if any(n.upper() in ticker for n in home_names):
                    prob = (yes_price or 50) / 100
                    american = self._kalshi_to_american_odds(prob)
                    result['home_ml_yes_price'] = yes_price
                    result['home_ml_no_price'] = no_price
                    result['home_ml'] = american
                elif any(n.upper() in ticker for n in away_names):
                    prob = (yes_price or 50) / 100
                    american = self._kalshi_to_american_odds(prob)
                    result['away_ml_yes_price'] = yes_price
                    result['away_ml_no_price'] = no_price
                    result['away_ml'] = american

            self.event_logger.event('info', f"Found moneyline for {home_team} vs {away_team}", category='market_data')
            return result if result else None
        except Exception as e:
            self.event_logger.event('error', f'Game market retrieval failed: {e}', category=classify_error(e))
            return None

    def _kalshi_to_american_odds(self, probability: float) -> int:
        """Convert Kalshi probability (0-1) to American odds format"""
        if probability >= 0.5:
            # Favorite: negative odds
            return int(-100 * probability / (1 - probability))
        else:
            # Underdog: positive odds
            return int(100 * (1 - probability) / probability)
    
    def get_performance_metrics(self, days: int = 30) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            dict: Performance metrics
        """
        # Get order history
        orders = self.get_order_history(limit=1000)
        
        # Filter to recent orders
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_orders = [
            order for order in orders
            if datetime.fromisoformat(order.get('created_time', '').replace('Z', '+00:00')) > cutoff_date
        ]
        
        if not recent_orders:
            return {}
        
        # Calculate metrics
        total_pnl = sum(order.get('pnl', 0) for order in recent_orders) / 100  # Convert from cents
        total_volume = sum(order.get('count', 0) for order in recent_orders) / 100
        
        winning_trades = [order for order in recent_orders if order.get('pnl', 0) > 0]
        losing_trades = [order for order in recent_orders if order.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(recent_orders) if recent_orders else 0
        avg_win = sum(order.get('pnl', 0) for order in winning_trades) / len(winning_trades) / 100 if winning_trades else 0
        avg_loss = sum(order.get('pnl', 0) for order in losing_trades) / len(losing_trades) / 100 if losing_trades else 0
        
        return {
            'total_pnl': total_pnl,
            'total_volume': total_volume,
            'total_trades': len(recent_orders),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'roi': (total_pnl / total_volume) if total_volume > 0 else 0
        }

# Example usage
if __name__ == "__main__":
    # Initialize client (demo environment)
    client = KalshiClient(
        api_key="your_api_key_here",
        api_secret="your_api_secret_here",
        environment="demo"
    )
    
    # Get account info
    account = client.get_account_info()
    print(f"Account balance: ${account.get('portfolio_balance', 0) / 100:.2f}")
    
    # Get NBA markets
    markets = client.get_nba_markets()
    print(f"Found {len(markets)} NBA markets")
    
    # Example predictions (normally from ML model)
    sample_predictions = [
        {
            'home_team': 'Lakers',
            'away_team': 'Celtics',
            'confidence': 0.67,
            'moneyline': {'probability': 0.58, 'line': 0},
            'spread': {'probability': 0.62, 'line': -3.5},
            'total': {'probability': 0.55, 'line': 218.5}
        }
    ]
    
    # Find betting opportunities
    opportunities = client.find_betting_opportunities(sample_predictions)
    print(f"Found {len(opportunities)} betting opportunities")
    
    # Execute strategy (dry run)
    if opportunities:
        trades = client.execute_betting_strategy(opportunities, dry_run=True)
        print(f"Simulated {len(trades)} trades")
    
    # Get performance metrics
    metrics = client.get_performance_metrics()
    if metrics:
        print(f"30-day performance: ROI {metrics['roi']:.1%}, Win Rate {metrics['win_rate']:.1%}")
