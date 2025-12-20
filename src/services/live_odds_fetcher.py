"""
Live Odds Fetcher for Trial 1306
Fetches real-time odds from Kalshi API with fallback to defaults
"""

import os
import json
from typing import Dict, Optional, List
from pathlib import Path


class LiveOddsFetcher:
    """Fetch live odds from Kalshi with smart fallbacks"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'config/kalshi_config.json'
        self.kalshi_client = None
        self._load_kalshi_client()
    
    def _load_kalshi_client(self):
        """Load Kalshi client if credentials available"""
        try:
            # PRIORITY 1: Try to load from .kalshi_credentials (working file)
            creds_file = Path('.kalshi_credentials')
            if creds_file.exists():
                print("[KALSHI] Loading from .kalshi_credentials...")
                api_key, api_secret = self._parse_kalshi_credentials(creds_file)
                
                if api_key and api_secret:
                    try:
                        from .kalshi_client import KalshiClient
                    except ImportError:
                        from kalshi_client import KalshiClient
                    
                    self.kalshi_client = KalshiClient(
                        api_key=api_key,
                        api_secret=api_secret,
                        environment='prod',  # Use production
                        auth_on_init=True,
                        request_timeout=15
                    )
                    print("[KALSHI] ✅ Connected to PRODUCTION environment")
                    return
            
            # PRIORITY 2: Fallback to config/kalshi_config.json
            if Path(self.config_path).exists():
                print(f"[KALSHI] Loading from {self.config_path}...")
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                api_key = config.get('api_key')
                api_secret = config.get('api_secret')
                environment = config.get('environment', 'demo')
                
                if api_key and api_secret and not api_key.startswith('YOUR_'):
                    try:
                        from .kalshi_client import KalshiClient
                    except ImportError:
                        from kalshi_client import KalshiClient
                    
                    self.kalshi_client = KalshiClient(
                        api_key=api_key,
                        api_secret=api_secret,
                        environment=environment,
                        auth_on_init=True
                    )
                    print(f"[KALSHI] Connected to {environment} environment")
                else:
                    print("[KALSHI] No valid credentials in config, using defaults")
            else:
                print(f"[KALSHI] Config not found, using defaults")
        
        except Exception as e:
            print(f"[KALSHI] Failed to initialize: {e}, using defaults")
            self.kalshi_client = None
    
    def _parse_kalshi_credentials(self, creds_file: Path) -> tuple:
        """Parse .kalshi_credentials file"""
        try:
            content = creds_file.read_text()
            
            # Parse API_KEY_ID or API_KEY (both formats supported)
            api_key = None
            for line in content.split('\n'):
                if line.startswith('API_KEY_ID='):
                    api_key = line.split('=', 1)[1].strip()
                    break
                elif line.startswith('API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    break
            
            # Extract private key (between BEGIN and END markers)
            if '-----BEGIN RSA PRIVATE KEY-----' in content:
                start_marker = '-----BEGIN RSA PRIVATE KEY-----'
                end_marker = '-----END RSA PRIVATE KEY-----'
                
                start_idx = content.find(start_marker)
                end_idx = content.find(end_marker)
                
                if start_idx != -1 and end_idx != -1:
                    private_key = content[start_idx:end_idx + len(end_marker)]
                    return api_key, private_key
            
            return None, None
        
        except Exception as e:
            print(f"[KALSHI] Error parsing credentials: {e}")
            return None, None
    
    def get_moneyline_odds(self, home_team: str, away_team: str, game_date: str) -> Optional[Dict]:
        """
        Get moneyline odds for a game
        
        Returns:
            Dict with odds if available, None if no real market data
            {
                'home_ml': float (American odds),
                'away_ml': float (American odds),
                'source': str,
                'yes_price': float (0-100),
                'no_price': float (0-100)
            }
        """
        # Try Kalshi first
        if self.kalshi_client:
            try:
                odds = self._fetch_from_kalshi(home_team, away_team, game_date)
                if odds:
                    return odds
            except Exception as e:
                print(f"[KALSHI] Error fetching odds: {e}")
        
        # No default fallback - return None to indicate no real market data available
        print(f"[ODDS] No real market odds available for {away_team} @ {home_team} on {game_date}")
        return None
    
    def _fetch_from_kalshi(self, home_team: str, away_team: str, game_date: str) -> Optional[Dict]:
        """Fetch odds from Kalshi API using get_game_markets"""
        if not self.kalshi_client:
            return None
        
        try:
            # Use the working get_game_markets method
            markets = self.kalshi_client.get_game_markets(home_team, away_team, game_date)
            
            if markets:
                home_yes_price = markets.get('home_ml_yes_price', 0)
                away_yes_price = markets.get('away_ml_yes_price', 0)
                home_ml = markets.get('home_ml')
                away_ml = markets.get('away_ml')
                
                # Validate we got real data (all fields must be present and valid)
                if (home_yes_price > 0 and away_yes_price > 0 and 
                    home_ml is not None and away_ml is not None):
                    print(f"[KALSHI] Found odds: {home_team} {home_yes_price}c ({home_ml}) vs {away_team} {away_yes_price}c ({away_ml})")
                    return {
                        'home_ml': home_ml,
                        'away_ml': away_ml,
                        'source': 'kalshi',
                        'yes_price': home_yes_price,
                        'no_price': away_yes_price
                    }
                else:
                    print(f"[KALSHI] Market found but invalid/missing data (home_ml={home_ml}, away_ml={away_ml}, prices={home_yes_price}/{away_yes_price})")
            else:
                print(f"[KALSHI] No market found for {home_team} vs {away_team}")
            
            return None
        
        except Exception as e:
            print(f"[KALSHI] API error: {e}")
            return None
    
    def _kalshi_to_american(self, kalshi_price: float) -> float:
        """
        Convert Kalshi price (0-100) to American odds
        
        Kalshi price is cents per contract (50 = 50% = even odds)
        """
        prob = kalshi_price / 100.0
        
        if prob >= 0.5:
            # Favorite (negative odds)
            return -100 * prob / (1 - prob)
        else:
            # Underdog (positive odds)
            return 100 * (1 - prob) / prob
    
    def remove_vig(self, home_ml: float, away_ml: float) -> tuple:
        """
        Remove vig from American odds to get fair probabilities
        
        Returns:
            (home_fair_prob, away_fair_prob)
        """
        # Convert to implied probabilities
        if home_ml < 0:
            home_implied = abs(home_ml) / (abs(home_ml) + 100)
        else:
            home_implied = 100 / (home_ml + 100)
        
        if away_ml < 0:
            away_implied = abs(away_ml) / (abs(away_ml) + 100)
        else:
            away_implied = 100 / (away_ml + 100)
        
        # Normalize to remove vig
        total_implied = home_implied + away_implied
        home_fair = home_implied / total_implied
        away_fair = away_implied / total_implied
        
        return home_fair, away_fair


def create_kalshi_config():
    """Create Kalshi config file template"""
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / 'kalshi_config.json'
    
    if not config_path.exists():
        template = {
            "api_key": "YOUR_KALSHI_API_KEY_HERE",
            "api_secret": "YOUR_KALSHI_API_SECRET_HERE",
            "environment": "demo",
            "comment": "Set environment to 'prod' for live trading"
        }
        
        with open(config_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"[CONFIG] Created template: {config_path}")
        print("[CONFIG] Edit this file with your Kalshi credentials")
    else:
        print(f"[CONFIG] Already exists: {config_path}")


if __name__ == '__main__':
    # Create config template
    create_kalshi_config()
    
    # Test odds fetcher
    fetcher = LiveOddsFetcher()
    
    print("\n" + "="*60)
    print("TESTING LIVE ODDS FETCHER")
    print("="*60)
    
    # Test with sample game
    odds = fetcher.get_moneyline_odds('LAL', 'PHX', '2024-12-15')
    
    print(f"\nSample Odds (LAL @ PHX):")
    print(f"  Home ML: {odds['home_ml']}")
    print(f"  Away ML: {odds['away_ml']}")
    print(f"  Source: {odds['source']}")
    
    # Test vig removal
    home_fair, away_fair = fetcher.remove_vig(odds['home_ml'], odds['away_ml'])
    print(f"\nFair Probabilities:")
    print(f"  Home: {home_fair:.2%}")
    print(f"  Away: {away_fair:.2%}")
