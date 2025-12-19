"""
Calculate advanced stats (DRB%, STL%, foul rate, opponent stats) and add to game_advanced_stats.

This script:
1. Reads game_logs to get raw box score data
2. Calculates DRB%, STL%, foul_rate for each team-game
3. Calculates opponent-allowed stats (opp_efg_allowed, opp_orb_allowed)
4. Updates game_advanced_stats table with new columns
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_PATH = "nba_betting_data.db"

def calculate_advanced_stats():
    """Calculate and populate advanced stats in database"""
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Step 1: Add new columns if they don't exist
        print("Adding new columns to game_advanced_stats...")
        cursor = conn.cursor()
        
        new_columns = [
            ('drb_pct', 'REAL'),
            ('stl_pct', 'REAL'),
            ('foul_rate', 'REAL'),
            ('opp_efg_allowed', 'REAL'),
            ('opp_orb_allowed', 'REAL'),
            ('opp_drb_allowed', 'REAL')
        ]
        
        for col_name, col_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE game_advanced_stats ADD COLUMN {col_name} {col_type}")
                print(f"  ✓ Added {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"  - {col_name} already exists")
                else:
                    raise
        
        conn.commit()
        
        # Step 2: Check if game_advanced_stats has records
        print("\nChecking existing game_advanced_stats...")
        cursor.execute("SELECT COUNT(*) FROM game_advanced_stats")
        existing_count = cursor.fetchone()[0]
        print(f"Existing game_advanced_stats records: {existing_count}")
        
        # Step 3: Load game_logs for calculations
        print("\nLoading game_logs...")
        game_logs = pd.read_sql_query("""
            SELECT 
                GAME_ID,
                TEAM_ID,
                TEAM_NAME,
                GAME_DATE,
                FGM, FGA, FG3M,
                OREB, DREB, REB,
                STL, BLK, TOV, PF,
                MIN
            FROM game_logs
            WHERE GAME_DATE IS NOT NULL
        """, conn)
        
        print(f"Loaded {len(game_logs)} game log records")
        
        # Step 4: Calculate advanced stats for each game
        print("\nCalculating advanced stats...")
        
        game_logs['game_date'] = pd.to_datetime(game_logs['GAME_DATE'])
        
        advanced_stats = []
        
        for game_id in game_logs['GAME_ID'].unique():
            game_data = game_logs[game_logs['GAME_ID'] == game_id]
            
            if len(game_data) != 2:
                continue  # Skip if not exactly 2 teams
            
            for idx, team in game_data.iterrows():
                opponent = game_data[game_data['TEAM_ID'] != team['TEAM_ID']].iloc[0]
                
                # Calculate team possessions (approximation)
                team_poss = (team['FGA'] - team['OREB'] + team['TOV'] + 0.44 * team['FGA'])
                opp_poss = (opponent['FGA'] - opponent['OREB'] + opponent['TOV'] + 0.44 * opponent['FGA'])
                
                # Defensive Rebound % = DREB / (DREB + Opp_OREB)
                drb_pct = team['DREB'] / (team['DREB'] + opponent['OREB']) if (team['DREB'] + opponent['OREB']) > 0 else 0.75
                
                # Steal % = STL / Opp_Possessions
                stl_pct = team['STL'] / opp_poss if opp_poss > 0 else 0.08
                
                # Foul Rate = PF / Opp_Possessions (Personal fouls per opponent possession)
                foul_rate = team['PF'] / opp_poss if opp_poss > 0 else 0.20
                
                # Opponent eFG% allowed = (Opp_FGM + 0.5 * Opp_3PM) / Opp_FGA
                opp_efg_allowed = (opponent['FGM'] + 0.5 * opponent['FG3M']) / opponent['FGA'] if opponent['FGA'] > 0 else 0.52
                
                # Opponent ORB% allowed = Opp_OREB / (Opp_OREB + Team_DREB)
                opp_orb_allowed = opponent['OREB'] / (opponent['OREB'] + team['DREB']) if (opponent['OREB'] + team['DREB']) > 0 else 0.25
                
                # Opponent DRB% allowed = Opp_DREB / (Opp_DREB + Team_OREB)
                opp_drb_allowed = opponent['DREB'] / (opponent['DREB'] + team['OREB']) if (opponent['DREB'] + team['OREB']) > 0 else 0.75
                
                advanced_stats.append({
                    'game_id': game_id,
                    'team_id': team['TEAM_ID'],
                    'team_name': team['TEAM_NAME'],
                    'game_date': team['game_date'],
                    'drb_pct': drb_pct,
                    'stl_pct': stl_pct,
                    'foul_rate': foul_rate,
                    'opp_efg_allowed': opp_efg_allowed,
                    'opp_orb_allowed': opp_orb_allowed,
                    'opp_drb_allowed': opp_drb_allowed
                })
        
        stats_df = pd.DataFrame(advanced_stats)
        print(f"Calculated stats for {len(stats_df)} team-games")
        
        # Step 5: Update game_advanced_stats table
        print("\nUpdating game_advanced_stats table...")
        
        # If game_advanced_stats is empty, we need to insert records first
        if existing_count == 0:
            print("game_advanced_stats is empty - creating records from game_logs...")
            # Load all games with basic stats
            basic_stats = pd.read_sql_query("""
                SELECT DISTINCT
                    g.GAME_ID as game_id,
                    g.TEAM_ID as team_id,
                    g.TEAM_ABBREVIATION as team_abb,
                    g.GAME_DATE as game_date
                FROM game_logs g
                WHERE g.GAME_DATE IS NOT NULL
            """, conn)
            
            # Insert basic records
            for _, row in basic_stats.iterrows():
                cursor.execute("""
                    INSERT OR IGNORE INTO game_advanced_stats 
                    (game_id, team_id, team_abb, game_date)
                    VALUES (?, ?, ?, ?)
                """, (row['game_id'], row['team_id'], row['team_abb'], row['game_date']))
            
            conn.commit()
            print(f"Inserted {len(basic_stats)} basic records")
        
        updated = 0
        for _, row in stats_df.iterrows():
            cursor.execute("""
                UPDATE game_advanced_stats
                SET drb_pct = ?,
                    stl_pct = ?,
                    foul_rate = ?,
                    opp_efg_allowed = ?,
                    opp_orb_allowed = ?,
                    opp_drb_allowed = ?
                WHERE game_id = ? AND team_id = ?
            """, (
                row['drb_pct'],
                row['stl_pct'],
                row['foul_rate'],
                row['opp_efg_allowed'],
                row['opp_orb_allowed'],
                row['opp_drb_allowed'],
                row['game_id'],
                row['team_id']
            ))
            updated += cursor.rowcount
        
        conn.commit()
        print(f"✓ Updated {updated} records in game_advanced_stats")
        
        # Step 6: Verify results
        print("\nVerifying results...")
        verification = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total,
                AVG(drb_pct) as avg_drb_pct,
                AVG(stl_pct) as avg_stl_pct,
                AVG(foul_rate) as avg_foul_rate,
                AVG(opp_efg_allowed) as avg_opp_efg,
                AVG(opp_orb_allowed) as avg_opp_orb
            FROM game_advanced_stats
            WHERE drb_pct IS NOT NULL
        """, conn)
        
        print("\nCalculated Stats Summary:")
        print(f"  Records with new stats: {verification['total'].iloc[0]}")
        print(f"  Avg DRB%: {verification['avg_drb_pct'].iloc[0]:.3f}")
        print(f"  Avg STL%: {verification['avg_stl_pct'].iloc[0]:.3f}")
        print(f"  Avg Foul Rate: {verification['avg_foul_rate'].iloc[0]:.3f}")
        print(f"  Avg Opp eFG% Allowed: {verification['avg_opp_efg'].iloc[0]:.3f}")
        print(f"  Avg Opp ORB% Allowed: {verification['avg_opp_orb'].iloc[0]:.3f}")
        
        print("\n✓ Advanced stats calculation complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED STATS CALCULATOR")
    print("=" * 60)
    calculate_advanced_stats()
