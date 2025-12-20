"""
NBA Injury Scraper with Spread Penalty Calculation
Based on CBS Sports injury reports + player impact values

Implements THE SHARP FORMULA:
  penalty = player_value * (1 - play_probability)
"""
from __future__ import annotations

import sys
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Optional
import logging

# Import player impact values (optional - graceful degradation)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from player_impact_values import PLAYER_SPREAD_VALUES, calculate_injury_penalty
    PLAYER_IMPACT_AVAILABLE = True
except ImportError:
    PLAYER_IMPACT_AVAILABLE = False
    PLAYER_SPREAD_VALUES = {}
    def calculate_injury_penalty(*args, **kwargs):
        return 0.0

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Sharp bettors often treat 'Questionable' differently based on the team.
# This dictionary maps standard statuses to "Probability of Playing"
STATUS_PROB_MAP = {
    'Out': 0.0,
    'Doubtful': 0.25,           # Historically ~25% chance to play
    'Questionable': 0.50,       # Historically ~50-60% chance (variable by player)
    'Probable': 0.75,           # Historically ~85-90% chance
    'Game Time Decision': 0.50,
    'Available': 1.0,
    'Unknown': 0.0
}

# Team-specific adjustments for Questionable status
TEAM_QUESTIONABLE_ADJUSTMENTS = {
    'Los Angeles Lakers': 0.75, 'LAL': 0.75,
    'Miami Heat': 0.75, 'MIA': 0.75,
    'Philadelphia 76ers': 0.50, 'PHI': 0.50,
    'Los Angeles Clippers': 0.70, 'LAC': 0.70,
    'Milwaukee Bucks': 0.70, 'MIL': 0.70,
    'Denver Nuggets': 0.65, 'DEN': 0.65,
    'Memphis Grizzlies': 0.40, 'MEM': 0.40,
    'DEFAULT': 0.50,
}


# Map NBA team abbreviations to CBS Sports team names
TEAM_ABB_TO_CBS = {
    'ATL': 'Atlanta', 'BOS': 'Boston', 'BKN': 'Brooklyn', 'CHA': 'Charlotte',
    'CHI': 'Chicago', 'CLE': 'Cleveland', 'DAL': 'Dallas', 'DEN': 'Denver',
    'DET': 'Detroit', 'GSW': 'Golden St.', 'HOU': 'Houston', 'IND': 'Indiana',
    'LAC': 'L.A. Clippers', 'LAL': 'L.A. Lakers', 'MEM': 'Memphis', 'MIA': 'Miami',
    'MIL': 'Milwaukee', 'MIN': 'Minnesota', 'NOP': 'New Orleans', 'NYK': 'New York',
    'OKC': 'Oklahoma City', 'ORL': 'Orlando', 'PHI': 'Philadelphia', 'PHX': 'Phoenix',
    'POR': 'Portland', 'SAC': 'Sacramento', 'SAS': 'San Antonio', 'TOR': 'Toronto',
    'UTA': 'Utah', 'WAS': 'Washington'
}


class InjuryScraper:
    """Scrape NBA injuries from CBS Sports and calculate spread penalties"""
    
    def __init__(self, cache_dir: str = "data/injuries"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.url = "https://www.cbssports.com/nba/injuries/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def scrape_injuries(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Scrape current NBA injury reports with spread penalty calculation
        
        Args:
            use_cache: If True, use cached data from today if available
            
        Returns:
            DataFrame with columns: Player, Team, Position, Status, Probability, 
                                   Adjusted_Probability, Spread_Value_Max, 
                                   Adjusted_Penalty, Description, Updated
        """
        cache_file = self.cache_dir / f"nba_injuries_{datetime.now().strftime('%Y-%m-%d')}.csv"
        
        if use_cache and cache_file.exists():
            print(f"✅ Loading cached injuries from {cache_file}")
            return pd.read_csv(cache_file)
        
        print(f"🔄 Scraping injuries from {self.url}")
        
        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            injury_data = []
            
            # CBS organizes injuries by Team blocks
            team_sections = soup.find_all('div', class_='TeamLogoNameLockup-name')
            
            # The actual tables follow the team headers
            tables = soup.find_all('table', class_='TableBase-table')
            
            if len(tables) == 0:
                print("⚠️ No injury tables found. Class names may have changed.")
                return pd.DataFrame()
            
            # Extract team names for matching
            team_names = [section.text.strip() for section in team_sections]
            
            for idx, table in enumerate(tables):
                # Get team name from header
                team_name = team_names[idx] if idx < len(team_names) else "Unknown"
                
                rows = table.find_all('tr')
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        # Extract player info
                        player_elem = cols[0].find('span', class_='CellPlayerName--long')
                        if not player_elem:
                            continue
                            
                        player_name = player_elem.text.strip()
                        position = cols[1].text.strip()
                        updated = cols[2].text.strip()
                        injury_desc = cols[3].text.strip()
                        
                        # Determine Status from description text
                        status = "Unknown"
                        desc_lower = injury_desc.lower()
                        
                        if "out" in desc_lower and "doubtful" not in desc_lower:
                            status = "Out"
                        elif "doubtful" in desc_lower:
                            status = "Doubtful"
                        elif "questionable" in desc_lower:
                            status = "Questionable"
                        elif "probable" in desc_lower:
                            status = "Probable"
                        elif "game time decision" in desc_lower or "gtd" in desc_lower:
                            status = "Game Time Decision"
                        elif "available" in desc_lower or "cleared" in desc_lower:
                            status = "Available"
                        
                        # Calculate base probability
                        play_prob = STATUS_PROB_MAP.get(status, 0.0)  # Default to 0 if unknown
                        
                        # Apply team-specific adjustment if Questionable
                        if status == 'Questionable':
                            adjusted_prob = TEAM_QUESTIONABLE_ADJUSTMENTS.get(
                                team_name,
                                TEAM_QUESTIONABLE_ADJUSTMENTS['DEFAULT']
                            )
                        else:
                            adjusted_prob = play_prob
                        
                        # Get player's max spread value
                        max_value = PLAYER_SPREAD_VALUES.get(player_name, 0.3)  # Default to bench player
                        
                        # THE SHARP FORMULA:
                        # If Probability is 1.0 (Playing), Penalty is 0.
                        # If Probability is 0.0 (Out), Penalty is Max Value.
                        # If Probability is 0.5 (Questionable), Penalty is Weighted.
                        penalty = max_value * (1 - adjusted_prob)
                        
                        injury_data.append({
                            "Player": player_name,
                            "Team": team_name,
                            "Position": position,
                            "Status": status,
                            "Probability": play_prob,
                            "Adjusted_Probability": adjusted_prob,
                            "Spread_Value_Max": max_value,
                            "Adjusted_Penalty": penalty,
                            "Description": injury_desc,
                            "Updated": updated,
                            "Scraped_At": datetime.now().isoformat()
                        })
            
            df = pd.DataFrame(injury_data)
            
            # Apply manual overrides for known incorrect statuses
            df = self._apply_manual_overrides(df)
            
            # Save to cache
            if len(df) > 0:
                df.to_csv(cache_file, index=False)
                print(f"✅ Scraped {len(df)} injuries and cached to {cache_file}")
            
            return df
                    
        except Exception as e:
            print(f"❌ Error scraping data: {e}")
            logger.error(f"Scraping failed: {e}")
            return pd.DataFrame()
    
    def _apply_manual_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply manual injury status overrides for known incorrect CBS data
        
        Returns:
            DataFrame with corrected statuses and recalculated penalties
        """
        import json
        
        override_file = Path(__file__).parent.parent / "data" / "manual_injury_overrides.json"
        
        if not override_file.exists():
            return df
        
        try:
            with open(override_file) as f:
                overrides = json.load(f)
            
            today = datetime.now().strftime('%Y-%m-%d')
            if today not in overrides:
                return df
            
            today_overrides = overrides[today]
            
            for player_name, override_data in today_overrides.items():
                # Find player in dataframe
                mask = df['Player'] == player_name
                if not mask.any():
                    continue
                
                # Update status and probability
                new_status = override_data['status']
                new_prob = override_data.get('probability', STATUS_PROB_MAP.get(new_status, 0.0))
                
                df.loc[mask, 'Status'] = new_status
                df.loc[mask, 'Probability'] = new_prob
                df.loc[mask, 'Adjusted_Probability'] = new_prob
                
                # Recalculate penalty with new probability
                max_value = df.loc[mask, 'Spread_Value_Max'].iloc[0]
                new_penalty = max_value * (1 - new_prob)
                df.loc[mask, 'Adjusted_Penalty'] = new_penalty
                
                print(f"✅ Manual override applied: {player_name} → {new_status} ({new_prob:.0%}), Penalty: -{new_penalty:.1f} pts")
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to apply manual overrides: {e}")
            return df
    
    def get_team_injury_impact(self, team_name: str, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate total spread penalty for a team's injuries
        
        Args:
            team_name: Full team name or abbreviation
            df: Injury DataFrame (if None, will scrape fresh)
            
        Returns:
            Dictionary with total_penalty, star_injuries, questionable_count, out_count, injuries list
        """
        if df is None:
            df = self.scrape_injuries()
        
        # Convert abbreviation to CBS team name if needed
        search_name = TEAM_ABB_TO_CBS.get(team_name.upper(), team_name)
        
        # Filter for team (case-insensitive partial match)
        team_injuries = df[df['Team'].str.contains(search_name, case=False, na=False)]
        
        if len(team_injuries) == 0:
            return {
                'team': team_name,
                'total_penalty': 0.0,
                'star_injuries': 0,
                'questionable_count': 0,
                'out_count': 0,
                'injuries': []
            }
        
        return {
            'team': team_name,
            'total_penalty': team_injuries['Adjusted_Penalty'].sum(),
            'star_injuries': len(team_injuries[team_injuries['Spread_Value_Max'] > 3.0]),
            'questionable_count': len(team_injuries[team_injuries['Status'] == 'Questionable']),
            'out_count': len(team_injuries[team_injuries['Status'] == 'Out']),
            'injuries': team_injuries.to_dict('records')
        }
    
    def get_game_injury_differential(
        self, 
        home_team: str, 
        away_team: str,
        df: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate injury spread differential for a game
        
        Positive = home team NET ADVANTAGE (away more injured)
        Negative = away team NET ADVANTAGE (home more injured)
        
        Args:
            home_team: Home team name
            away_team: Away team name
            df: Injury DataFrame
            
        Returns:
            Net injury differential in points (apply to home spread)
        """
        home_impact = self.get_team_injury_impact(home_team, df)
        away_impact = self.get_team_injury_impact(away_team, df)
        
        # Penalties are positive values (points deducted)
        home_penalty = home_impact['total_penalty']
        away_penalty = away_impact['total_penalty']
        
        # Differential: away penalty - home penalty
        # If away has 5 pts penalty and home has 2 pts penalty:
        # 5 - 2 = +3 (home has 3 pt advantage)
        differential = away_penalty - home_penalty
        
        print(f"   📊 Injury Differential: {home_team} vs {away_team}")
        print(f"      Home penalty: {home_penalty:.2f} pts")
        print(f"      Away penalty: {away_penalty:.2f} pts")
        print(f"      Net differential: {differential:+.2f} pts (favors {'home' if differential > 0 else 'away'})")
        
        return differential


if __name__ == "__main__":
    # Example usage
    scraper = InjuryScraper()
    
    print("=" * 80)
    print("NBA INJURY SCRAPER")
    print("=" * 80)
    
    df_injuries = scraper.scrape_injuries(use_cache=False)
    
    print(f"\n📊 Scraped {len(df_injuries)} injuries.")
    
    if len(df_injuries) > 0:
        print("\n🔝 Top 10 Impact Injuries:")
        print(df_injuries.nlargest(10, 'Adjusted_Penalty')[
            ['Player', 'Team', 'Status', 'Adjusted_Probability', 'Spread_Value_Max', 'Adjusted_Penalty']
        ].to_string())
        
        # Example: Dallas Mavericks
        print("\n" + "=" * 80)
        print("EXAMPLE: Dallas Mavericks Injury Impact")
        print("=" * 80)
        dal_impact = scraper.get_team_injury_impact("Dallas", df_injuries)
        print(f"\nTotal Spread Penalty: {dal_impact['total_penalty']:.2f} points")
        print(f"Star Injuries (>3.0 pts): {dal_impact['star_injuries']}")
        print(f"Questionable: {dal_impact['questionable_count']}")
        print(f"Out: {dal_impact['out_count']}")
        
        if dal_impact['injuries']:
            print("\nDetailed Injuries:")
            for inj in dal_impact['injuries']:
                print(f"  {inj['Player']:20} {inj['Status']:12} "
                      f"Prob: {inj['Adjusted_Probability']:.0%}  "
                      f"Value: {inj['Spread_Value_Max']:4.1f}  "
                      f"Penalty: {inj['Adjusted_Penalty']:+.2f}")
        
        # Example game differential
        print("\n" + "=" * 80)
        print("EXAMPLE: Game Injury Differential")
        print("=" * 80)
        diff = scraper.get_game_injury_differential("Los Angeles Lakers", "Boston Celtics", df_injuries)
        print(f"\n💡 Apply {diff:+.2f} points to home team's spread")
    
    print("\n" + "=" * 80)
