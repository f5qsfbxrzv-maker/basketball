"""
Live Injury Updater - Updates active_injuries table with current NBA injuries
Uses ESPN API for reliable, up-to-date injury data
"""

import sqlite3
import requests
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class LiveInjuryUpdater:
    """Fetch and store current NBA injuries using ESPN API"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        # ESPN injury endpoint - more reliable than scraping
        self.injury_url = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
    def update_active_injuries(self) -> int:
        """
        Fetch current injuries from ESPN and update active_injuries table
        
        Returns:
            Number of injured players updated
        """
        print("🏥 Fetching current NBA injuries from ESPN...")
        
        try:
            response = requests.get(self.injury_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            all_injuries = []
            
            # Parse ESPN injury data - structure: {injuries: [{id, displayName, injuries: [...]}]}
            teams = data.get('injuries', [])
            
            for team_data in teams:
                team_name = team_data.get('displayName', 'Unknown')
                team_id = team_data.get('id', 'Unknown')
                
                # Get injuries for this team
                team_injuries = team_data.get('injuries', [])
                
                for injury in team_injuries:
                    athlete = injury.get('athlete', {})
                    player_name = athlete.get('displayName', '')
                    
                    # Try to get position from athlete details
                    position = 'N/A'  # ESPN injury endpoint doesn't include position
                    
                    # Injury details
                    injury_status = injury.get('status', 'Unknown')
                    injury_details = injury.get('details', {})
                    injury_type = injury_details.get('type', 'Unknown') if isinstance(injury_details, dict) else 'Unknown'
                    injury_desc = f"{injury_type}"
                    
                    if player_name and injury_status:
                        all_injuries.append({
                            'player_name': player_name,
                            'team_name': team_name,
                            'team_id': team_id,
                            'position': position,
                            'status': injury_status,
                            'injury_desc': injury_desc,
                            'source': 'ESPN',
                            'last_updated': datetime.now().isoformat()
                        })
            
            if not all_injuries:
                print("⚠️ No injuries found from ESPN")
                return 0
            
            # Update database
            count = self._save_injuries(all_injuries)
            print(f"✅ Updated {count} injured players in active_injuries table")
            return count
            
        except Exception as e:
            logger.error(f"Failed to fetch injuries: {e}")
            print(f"❌ Error fetching injuries from ESPN: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _save_injuries(self, injuries: List[Dict]) -> int:
        """Save injuries to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear old injuries
        cursor.execute("DELETE FROM active_injuries")
        
        # Insert new injuries
        for inj in injuries:
            try:
                cursor.execute("""
                    INSERT INTO active_injuries 
                    (player_name, team_name, position, status, injury_desc, source, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    inj['player_name'],
                    inj['team_name'], 
                    inj['position'],
                    inj['status'],
                    inj['injury_desc'],
                    inj['source'],
                    inj['last_updated']
                ))
            except Exception as e:
                logger.warning(f"Failed to insert injury for {inj['player_name']}: {e}")
                continue
        
        conn.commit()
        count = len(injuries)
        conn.close()
        
        return count


if __name__ == "__main__":
    updater = LiveInjuryUpdater("data/live/nba_betting_data.db")
    count = updater.update_active_injuries()
    print(f"\n{'='*60}")
    print(f"Injury update complete: {count} players")
    print(f"{'='*60}")
