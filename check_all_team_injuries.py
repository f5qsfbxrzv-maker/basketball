"""
Check injury impact for all 30 NBA teams
Ensures every team has an injury value calculated (0.0 if no injuries)
"""
from datetime import datetime
from injury_impact_live import calculate_team_injury_impact_simple, TEAM_ABBREV_TO_FULL

# All 30 NBA teams
ALL_NBA_TEAMS = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

if __name__ == '__main__':
    today = datetime.now().strftime('%Y-%m-%d')
    db_path = 'data/live/nba_betting_data.db'
    
    print("\n" + "="*80)
    print(f"NBA INJURY IMPACT REPORT - {today}")
    print("="*80 + "\n")
    
    team_injuries = {}
    injured_teams = []
    healthy_teams = []
    
    for team_abbr in ALL_NBA_TEAMS:
        team_full = TEAM_ABBREV_TO_FULL[team_abbr]
        
        try:
            print(f"\n{team_abbr} - {team_full}:")
            impact = calculate_team_injury_impact_simple(team_abbr, today, db_path)
            
            team_injuries[team_abbr] = impact
            
            if impact > 0:
                injured_teams.append((team_abbr, impact))
            else:
                healthy_teams.append(team_abbr)
                print(f"  [OK] No significant injuries")
            
            print(f"  TOTAL IMPACT: {impact:.2f} pts")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            team_injuries[team_abbr] = 0.0
            healthy_teams.append(team_abbr)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n[OK] Teams with no significant injuries ({len(healthy_teams)}):")
    print(", ".join(healthy_teams))
    
    print(f"\n[INJURED] Teams with injuries ({len(injured_teams)}):")
    injured_teams.sort(key=lambda x: x[1], reverse=True)
    for team, impact in injured_teams:
        severity = "[HIGH]" if impact > 10 else "[MED]" if impact > 5 else "[LOW]"
        print(f"  {severity} {team}: {impact:.2f} pts")
    
    # Top 5 most injured
    print(f"\n[TOP 5] Most Injured Teams:")
    for i, (team, impact) in enumerate(injured_teams[:5], 1):
        team_full = TEAM_ABBREV_TO_FULL[team]
        print(f"  {i}. {team_full}: {impact:.2f} pts")
    
    # Verification
    print(f"\n[VERIFY] All {len(ALL_NBA_TEAMS)} NBA teams checked")
    missing = set(ALL_NBA_TEAMS) - set(team_injuries.keys())
    if missing:
        print(f"  [WARNING] Missing teams: {missing}")
    else:
        print(f"  [OK] All teams accounted for")
    
    print("\n" + "="*80 + "\n")
