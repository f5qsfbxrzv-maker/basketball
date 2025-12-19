"""
Download ACTUAL game results through today using ESPN Scoreboard API
ESPN provides real-time scores, not full-season cached data
"""
import requests
from datetime import datetime, timedelta
import sqlite3
import time

DB_PATH = r"c:\Users\d76do\OneDrive\Documents\New Basketball Model\nba_betting_data.db"

# Team name mapping
ESPN_TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

print("=" * 80)
print("DOWNLOADING REAL GAME RESULTS FROM ESPN (2025-26 Season through Dec 17, 2025)")
print("=" * 80)

# Start from 2025-26 season start (October 2025)
start_date = datetime(2025, 10, 22)
end_date = datetime(2025, 12, 17)

all_games = []
current_date = start_date

print(f"\n📥 Fetching games day by day from ESPN API...")
print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"   Total days: {(end_date - start_date).days + 1}")

dates_processed = 0
games_found = 0

while current_date <= end_date:
    date_str = current_date.strftime('%Y%m%d')
    
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        events = data.get('events', [])
        
        for event in events:
            # Only get COMPLETED games
            status = event.get('status', {}).get('type', {}).get('name', '')
            if status != 'STATUS_FINAL':
                continue
            
            competitions = event.get('competitions', [])
            if not competitions:
                continue
            
            comp = competitions[0]
            competitors = comp.get('competitors', [])
            
            if len(competitors) != 2:
                continue
            
            # Find home/away
            home = None
            away = None
            for team in competitors:
                if team.get('homeAway') == 'home':
                    home = team
                elif team.get('homeAway') == 'away':
                    away = team
            
            if not home or not away:
                continue
            
            # Extract data
            home_name = home.get('team', {}).get('displayName', '')
            away_name = away.get('team', {}).get('displayName', '')
            home_abbr = ESPN_TEAM_MAP.get(home_name, home_name)
            away_abbr = ESPN_TEAM_MAP.get(away_name, away_name)
            
            home_score = int(home.get('score', 0))
            away_score = int(away.get('score', 0))
            
            game_id = event.get('id', '')
            game_date = current_date.strftime('%Y-%m-%d')
            
            all_games.append({
                'game_id': f"ESPN_{game_id}",
                'game_date': game_date,
                'season': '2025-26',
                'home_team': home_abbr,
                'away_team': away_abbr,
                'home_score': home_score,
                'away_score': away_score,
                'home_won': 1 if home_score > away_score else 0,
                'total_points': home_score + away_score,
                'point_differential': home_score - away_score
            })
            games_found += 1
        
        dates_processed += 1
        if dates_processed % 10 == 0:
            print(f"   Progress: {dates_processed} days, {games_found} games found...")
        
        # Rate limiting
        time.sleep(0.15)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user after {dates_processed} days, {games_found} games")
        break
    except Exception as e:
        print(f"   ⚠️  Error fetching {current_date.strftime('%Y-%m-%d')}: {e}")
        time.sleep(1)  # Wait longer on error
    
    current_date += timedelta(days=1)

print(f"\n✅ Downloaded {games_found} completed games from ESPN")

if games_found == 0:
    print("\n❌ No games found! Cannot continue.")
    exit()

# Calculate team records
print("\n" + "=" * 80)
print("TEAM RECORDS FROM ESPN DATA")
print("=" * 80)

team_stats = {}
for game in all_games:
    home = game['home_team']
    away = game['away_team']
    
    if home not in team_stats:
        team_stats[home] = {'wins': 0, 'losses': 0, 'games': 0}
    if away not in team_stats:
        team_stats[away] = {'wins': 0, 'losses': 0, 'games': 0}
    
    if game['home_won']:
        team_stats[home]['wins'] += 1
        team_stats[away]['losses'] += 1
    else:
        team_stats[away]['wins'] += 1
        team_stats[home]['losses'] += 1
    
    team_stats[home]['games'] += 1
    team_stats[away]['games'] += 1

# Sort by wins
sorted_teams = sorted(team_stats.items(), key=lambda x: x[1]['wins'], reverse=True)

print(f"\n📊 TEAM RECORDS:")
for team, stats in sorted_teams:
    wins, losses, games = stats['wins'], stats['losses'], stats['games']
    if team in ['DET', 'DAL']:
        print(f"   {team}: {wins}-{losses} ({games} games) ⭐")
    else:
        print(f"   {team}: {wins}-{losses} ({games} games)")

# Verify expected records
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

if 'DET' in team_stats:
    det = team_stats['DET']
    print(f"\n✓ DET: {det['wins']}-{det['losses']} ({det['games']} games)")
    if det['wins'] == 20 and det['losses'] == 5:
        print("  ✅ MATCHES expected 20-5!")
    else:
        print(f"  ⚠️  Expected 20-5")

if 'DAL' in team_stats:
    dal = team_stats['DAL']
    print(f"\n✓ DAL: {dal['wins']}-{dal['losses']} ({dal['games']} games)")
    if dal['wins'] == 10 and dal['losses'] == 16:
        print("  ✅ MATCHES expected 10-16!")
    else:
        print(f"  ⚠️  Expected 10-16")

# Save to database
print("\n" + "=" * 80)
print("SAVE TO DATABASE")
print("=" * 80)

user_input = input(f"\n⚠️  Save {games_found} games to database? This will REPLACE game_results table. (yes/no): ")

if user_input.lower() != 'yes':
    print("❌ Cancelled")
    exit()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Clear existing data
cursor.execute("DELETE FROM game_results WHERE season = '2025-26'")

# Insert new data
for game in all_games:
    cursor.execute("""
        INSERT INTO game_results 
        (game_id, game_date, season, home_team, away_team, home_score, away_score, 
         home_won, total_points, point_differential)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game['game_id'], game['game_date'], game['season'],
        game['home_team'], game['away_team'],
        game['home_score'], game['away_score'],
        game['home_won'], game['total_points'], game['point_differential']
    ))

conn.commit()
conn.close()

print(f"\n✅ Saved {games_found} games to database")
print("\n" + "=" * 80)
print("SUCCESS! Database now has REAL game results for 2025-26 season through Dec 17, 2025")
print("=" * 80)
print("\n📋 Next step: Run ELO recalculation script")
