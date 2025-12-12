"""
Merge new injury CSV files into historical_inactives table
"""

import pandas as pd
import sqlite3
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DB_PATH = "data/live/nba_betting_data.db"

def load_csv_files():
    """Load the two new injury CSV files"""
    
    # File 1: 2023-24 Regular Season
    file1 = "c:/Users/d76do/Downloads/Injury Database - 2023-24 Regular Season.csv"
    
    # File 2: 2024-25 data
    file2 = "c:/Users/d76do/Downloads/data-956Jc(2).csv"
    
    print("Loading CSV files...")
    
    try:
        df1 = pd.read_csv(file1)
        print(f"\n📁 File 1: {file1}")
        print(f"   Rows: {len(df1)}")
        print(f"   Columns: {df1.columns.tolist()}")
        if len(df1) > 0:
            # Try to find date column
            date_col = None
            for col in df1.columns:
                if 'date' in col.lower() or df1[col].dtype == 'object':
                    try:
                        pd.to_datetime(df1[col])
                        date_col = col
                        break
                    except:
                        pass
            if date_col:
                print(f"   Date range: {df1[date_col].min()} to {df1[date_col].max()}")
        print(df1.head())
    except Exception as e:
        print(f"   ❌ Error loading file 1: {e}")
        df1 = None
    
    try:
        df2 = pd.read_csv(file2)
        print(f"\n📁 File 2: {file2}")
        print(f"   Rows: {len(df2)}")
        print(f"   Columns: {df2.columns.tolist()}")
        if len(df2) > 0 and 'DATE' in df2.columns:
            print(f"   Date range: {df2['DATE'].min()} to {df2['DATE'].max()}")
        print(df2.head())
    except Exception as e:
        print(f"   ❌ Error loading file 2: {e}")
        df2 = None
    
    return df1, df2

def check_existing_coverage():
    """Check current injury data coverage in database"""
    conn = sqlite3.connect(DB_PATH)
    
    print("\n" + "="*60)
    print("CURRENT DATABASE COVERAGE")
    print("="*60)
    
    # historical_inactives
    df = pd.read_sql("""
        SELECT 
            MIN(game_date) as min_date,
            MAX(game_date) as max_date,
            COUNT(*) as total_records,
            COUNT(DISTINCT game_date) as unique_dates
        FROM historical_inactives
    """, conn)
    print("\n📊 historical_inactives:")
    print(df.to_string(index=False))
    
    # Coverage by year
    df_years = pd.read_sql("""
        SELECT 
            strftime('%Y', game_date) as year,
            COUNT(*) as records
        FROM historical_inactives
        GROUP BY year
        ORDER BY year DESC
    """, conn)
    print("\n📅 Coverage by year:")
    print(df_years.to_string(index=False))
    
    conn.close()

def normalize_and_insert(df1, df2):
    """
    Normalize the CSV data to match historical_inactives schema and insert
    
    historical_inactives schema (actual):
    - id INTEGER
    - game_id TEXT
    - player_id INTEGER
    - player_name TEXT
    - team_abbreviation TEXT
    - season TEXT
    - game_date TEXT
    """
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    records_inserted = 0
    
    # Process File 2 (2024-25 data with known schema)
    if df2 is not None and len(df2) > 0:
        print(f"\n🔄 Processing File 2 - 2024-25 season ({len(df2)} rows)...")
        
        for idx, row in df2.iterrows():
            try:
                # Parse the data
                player_name = row['PLAYER']
                status = row['STATUS']
                reason = row['REASON']
                team = row['TEAM']
                game_str = row['GAME']
                date_str = row['DATE']
                
                # Convert date format (MM/DD/YYYY -> YYYY-MM-DD)
                game_date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
                season = "2024-25"  # Based on the dates (Oct 2024 onwards)
                
                # Extract team abbreviation (e.g., "New York Knicks" -> "NYK")
                team_map = {
                    'New York Knicks': 'NYK',
                    'Boston Celtics': 'BOS',
                    'Los Angeles Lakers': 'LAL',
                    'Minnesota Timberwolves': 'MIN',
                    'Indiana Pacers': 'IND',
                    'Detroit Pistons': 'DET',
                    'Brooklyn Nets': 'BKN',
                    'Atlanta Hawks': 'ATL',
                    'Cleveland Cavaliers': 'CLE',
                    'Toronto Raptors': 'TOR',
                    'Milwaukee Bucks': 'MIL',
                    'Philadelphia 76ers': 'PHI',
                    'Orlando Magic': 'ORL',
                    'Miami Heat': 'MIA',
                    'Charlotte Hornets': 'CHA',
                    'Houston Rockets': 'HOU',
                    'Chicago Bulls': 'CHI',
                    'New Orleans Pelicans': 'NOP',
                    'Memphis Grizzlies': 'MEM',
                    'Utah Jazz': 'UTA',
                    'Golden State Warriors': 'GSW',
                    'Portland Trail Blazers': 'POR',
                    'Phoenix Suns': 'PHX',
                    'LA Clippers': 'LAC',
                    'Denver Nuggets': 'DEN',
                    'Washington Wizards': 'WAS',
                    'San Antonio Spurs': 'SAS',
                    'Dallas Mavericks': 'DAL',
                    'Sacramento Kings': 'SAC',
                    'Oklahoma City Thunder': 'OKC',
                }
                
                team_abb = team_map.get(team, team[:3].upper())
                
                # Create a pseudo game_id (we don't have real game IDs from this data)
                game_id = f"{game_date}_{game_str.replace('@', '_')}"
                
                # Insert using correct column name
                cursor.execute("""
                    INSERT OR IGNORE INTO historical_inactives 
                    (game_id, game_date, player_name, team_abbreviation, season)
                    VALUES (?, ?, ?, ?, ?)
                """, (game_id, game_date, player_name, team_abb, season))
                
                if cursor.rowcount > 0:
                    records_inserted += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error on row {idx}: {e}")
                continue
        
        conn.commit()
        print(f"   ✅ Inserted {records_inserted} new records from File 2")
    
    # Process File 1 (2023-24 Regular Season)
    if df1 is not None and len(df1) > 0:
        print(f"\n🔄 Processing File 1 - 2023-24 season ({len(df1)} rows)...")
        
        # Detect date column
        date_col = None
        for col in df1.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        # Detect other columns
        if date_col:
            file1_inserted = 0
            for idx, row in df1.iterrows():
                try:
                    # Try to extract information from available columns
                    # Schema is unknown, so we'll do best effort
                    game_date = pd.to_datetime(row[date_col]).strftime('%Y-%m-%d') if date_col else None
                    
                    # Look for player name column
                    player_name = None
                    for col in ['Player', 'PLAYER', 'player_name', 'Name', 'NAME']:
                        if col in df1.columns:
                            player_name = row[col]
                            break
                    
                    # Look for team column
                    team = None
                    team_abb = None
                    for col in ['Team', 'TEAM', 'team', 'Team Abbreviation']:
                        if col in df1.columns:
                            team = row[col]
                            team_abb = team if len(str(team)) <= 3 else None
                            break
                    
                    if not all([game_date, player_name]):
                        continue
                    
                    game_id = f"{game_date}_UNKNOWN"
                    season = "2023-24"
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO historical_inactives 
                        (game_id, game_date, player_name, team_abbreviation, season)
                        VALUES (?, ?, ?, ?, ?)
                    """, (game_id, game_date, player_name, team_abb or 'UNK', season))
                    
                    if cursor.rowcount > 0:
                        file1_inserted += 1
                        
                except Exception as e:
                    continue
            
            conn.commit()
            print(f"   ✅ Inserted {file1_inserted} new records from File 1")
            records_inserted += file1_inserted
        else:
            print("   ⚠️ Could not identify date column in File 1")
    
    conn.close()
    
    return records_inserted

def main():
    print("="*60)
    print("INJURY DATA MERGER")
    print("="*60)
    
    # Check current state
    check_existing_coverage()
    
    # Load new data
    df1, df2 = load_csv_files()
    
    if df1 is None and df2 is None:
        print("\n❌ No CSV files could be loaded")
        return
    
    # Insert new data
    total_inserted = normalize_and_insert(df1, df2)
    
    # Check updated state
    print("\n" + "="*60)
    print("UPDATED DATABASE COVERAGE")
    print("="*60)
    check_existing_coverage()
    
    print(f"\n✅ Total new records inserted: {total_inserted}")

if __name__ == "__main__":
    main()
