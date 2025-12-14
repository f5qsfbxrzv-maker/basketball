"""
Multi-Source Odds Service - Kalshi API + The Odds API fallback
Priority: Kalshi (for betting) > The Odds API (for comparison)
"""

from typing import Dict, Optional
from datetime import datetime
import os
import json
from pathlib import Path


class MultiSourceOddsService:
    """Fetch odds from multiple sources with Kalshi as primary"""
    
    def __init__(self, kalshi_api_key: str = None, kalshi_api_secret: str = None, odds_api_key: str = None):
        self.kalshi_api_key = kalshi_api_key
        self.kalshi_api_secret = kalshi_api_secret
        self.odds_api_key = odds_api_key
        
        # Try to load credentials from config file if not provided
        config_path = Path(__file__).parent / "config" / "api_credentials.json"
        if config_path.exists() and (not kalshi_api_key or not odds_api_key):
            try:
                with open(config_path) as f:
                    creds = json.load(f)
                    self.kalshi_api_key = self.kalshi_api_key or creds.get('kalshi_api_key')
                    self.kalshi_api_secret = self.kalshi_api_secret or creds.get('kalshi_api_secret')
                    self.odds_api_key = self.odds_api_key or creds.get('odds_api_key')
            except Exception:
                pass
        
        # Initialize Kalshi client if credentials available
        self.kalshi_client = None
        if self.kalshi_api_key and self.kalshi_api_secret:
            try:
                from src.services._OLD_kalshi_client import KalshiClient
                self.kalshi_client = KalshiClient(
                    api_key=self.kalshi_api_key,
                    api_secret=self.kalshi_api_secret,
                    environment='prod',
                    auth_on_init=True
                )
                print("[SUCCESS] Kalshi API client initialized")
            except Exception as e:
                print(f"[WARNING] Could not initialize Kalshi client: {e}")
                self.kalshi_client = None
        
    def get_game_odds(self, home_team: str, away_team: str, game_date: datetime) -> Dict:
        """
        Fetch odds for a specific game
        Priority: Kalshi > The Odds API > Default
        
        Args:
            home_team: Home team abbreviation (e.g., 'LAL')
            away_team: Away team abbreviation (e.g., 'BOS')
            game_date: Game date
            
        Returns:
            Dict with:
                - home_ml_odds: American odds for home team
                - away_ml_odds: American odds for away team  
                - kalshi_home_prob: Kalshi market probability (primary)
                - kalshi_away_prob: Kalshi market probability (primary)
                - source: 'Kalshi', 'OddsAPI', or 'Default'
        """
        
        # Try Kalshi first (this is where we bet!)
        if self.kalshi_client:
            try:
                kalshi_odds = self._fetch_from_kalshi(home_team, away_team, game_date)
                if kalshi_odds['kalshi_home_prob'] is not None:
                    return kalshi_odds
            except Exception as e:
                print(f"[WARNING] Kalshi fetch failed: {e}")
        
        # Try The Odds API as fallback
        if self.odds_api_key:
            try:
                odds_api_data = self._fetch_from_odds_api(home_team, away_team, game_date)
                if odds_api_data['home_ml_odds']:
                    return odds_api_data
            except Exception as e:
                print(f"[WARNING] Odds API fetch failed: {e}")
        
        # Default fallback (50/50 odds)
        return {
            'home_ml_odds': -110,
            'away_ml_odds': -110,
            'kalshi_home_prob': None,
            'kalshi_away_prob': None,
            'source': 'Default',
            'spread': None,
            'total': None
        }
    
    def _fetch_from_kalshi(self, home_team: str, away_team: str, game_date: datetime) -> Dict:
        """
        Fetch odds from Kalshi API
        Returns market probabilities which are the REAL prices for betting
        """
        if not self.kalshi_client:
            return {'kalshi_home_prob': None, 'kalshi_away_prob': None, 'source': 'Kalshi'}
        
        try:
            # Use get_game_markets which searches KXNBAGAME series
            date_str = game_date.strftime('%Y-%m-%d') if game_date else None
            market_data = self.kalshi_client.get_game_markets(home_team, away_team, date_str)
            
            if market_data:
                # Extract yes prices (cents per contract)
                home_yes_price = market_data.get('home_ml_yes_price', 50)
                away_yes_price = market_data.get('away_ml_yes_price', 50)
                
                # Validate prices - reject if 0 (closed/settled market with no liquidity)
                if home_yes_price == 0 or away_yes_price == 0:
                    print(f"[WARNING] Kalshi market for {home_team} vs {away_team} has no liquidity (prices=0)")
                    return {
                        'home_ml_odds': None,
                        'away_ml_odds': None,
                        'kalshi_home_prob': None,
                        'kalshi_away_prob': None,
                        'source': 'Kalshi'
                    }
                
                # Convert to probabilities (cents / 100)
                kalshi_home_prob = home_yes_price / 100.0
                kalshi_away_prob = away_yes_price / 100.0
                
                # Get American odds (already calculated by get_game_markets)
                home_ml_odds = market_data.get('home_ml', -110)
                away_ml_odds = market_data.get('away_ml', -110)
                
                return {
                    'home_ml_odds': home_ml_odds,
                    'away_ml_odds': away_ml_odds,
                    'kalshi_home_prob': kalshi_home_prob,
                    'kalshi_away_prob': kalshi_away_prob,
                    'source': 'Kalshi'
                }
            
            # No matching market found
            return {
                'home_ml_odds': None,
                'away_ml_odds': None,
                'kalshi_home_prob': None,
                'kalshi_away_prob': None,
                'source': 'Kalshi'
            }
            
        except Exception as e:
            print(f"[ERROR] Kalshi API error: {e}")
            return {
                'kalshi_home_prob': None,
                'kalshi_away_prob': None,
                'source': 'Kalshi'
            }
    
    def _fetch_from_odds_api(self, home_team: str, away_team: str, game_date: datetime) -> Dict:
        """
        Fetch odds from The Odds API (backup source)
        """
        if not self.odds_api_key:
            return {'home_ml_odds': None, 'away_ml_odds': None, 'source': 'OddsAPI'}
        
        try:
            import requests
            
            # The Odds API endpoint
            url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h',  # Head to head (moneyline)
                'oddsFormat': 'american'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Find matching game
            date_str = game_date.strftime('%Y-%m-%d')
            for game in data:
                game_date_api = game.get('commence_time', '')[:10]
                if game_date_api == date_str:
                    teams = [team.lower() for team in [game.get('home_team', ''), game.get('away_team', '')]]
                    if home_team.lower() in str(teams) and away_team.lower() in str(teams):
                        # Get best odds from bookmakers
                        bookmakers = game.get('bookmakers', [])
                        if bookmakers:
                            odds = bookmakers[0].get('markets', [{}])[0].get('outcomes', [])
                            home_odds = next((o['price'] for o in odds if 'home' in o.get('name', '').lower()), -110)
                            away_odds = next((o['price'] for o in odds if 'away' in o.get('name', '').lower()), -110)
                            
                            return {
                                'home_ml_odds': home_odds,
                                'away_ml_odds': away_odds,
                                'kalshi_home_prob': None,  # Not from Kalshi
                                'kalshi_away_prob': None,
                                'source': 'OddsAPI'
                            }
            
            return {'home_ml_odds': None, 'away_ml_odds': None, 'source': 'OddsAPI'}
            
        except Exception as e:
            print(f"[ERROR] Odds API error: {e}")
            return {'home_ml_odds': None, 'away_ml_odds': None, 'source': 'OddsAPI'}
    
    def _prob_to_american_odds(self, prob: float) -> int:
        """Convert probability to American odds"""
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)


if __name__ == "__main__":
    # Test the service
    service = MultiSourceOddsService()
    odds = service.get_game_odds('LAL', 'BOS', datetime.now())
    print("Sample odds data:")
    print(json.dumps(odds, indent=2))
