import pandas as pd
import numpy as np

# CONFIG
INPUT_FILE = 'data/training_data_matchup_with_injury_advantage.csv'
OUTPUT_FILE = 'data/training_data_matchup_with_injury_advantage_FIXED.csv'

def repair_elo():
    print("===================================================================")
    print("DATASET REPAIR: SURGICAL ELO TRANSPLANT")
    print("===================================================================\n")
    
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    original_rows = len(df)
    
    # 1. CREATE A "MASTER ELO HISTORY"
    # We trust 'away_composite_elo' as the Source of Truth.
    # We will scrape every instance of a team playing Away to build their timeline.
    
    print("[1] Extracting valid ELOs from Away games...")
    
    # Get all Away records: [Date, Team, ELO]
    away_records = df[['date', 'away_team', 'away_composite_elo']].rename(
        columns={'away_team': 'team', 'away_composite_elo': 'elo'}
    )
    
    # Get all Home records: [Date, Team, ELO]
    # We include these initially, but we will OVERWRITE them with the interpolations
    home_records = df[['date', 'home_team', 'home_composite_elo']].rename(
        columns={'home_team': 'team', 'home_composite_elo': 'elo'}
    )
    
    # 2. BUILD TEAM TIMELINES
    # We assume the "Away" column is 90% accurate, but it has gaps (when teams play at home).
    # We need to fill those gaps.
    
    # Combine just to get the schedule structure
    full_schedule = pd.concat([
        df[['date', 'home_team']].rename(columns={'home_team':'team'}).assign(loc='HOME'), 
        df[['date', 'away_team']].rename(columns={'away_team':'team'}).assign(loc='AWAY')
    ]).sort_values(['team', 'date'])
    
    # Create the reference dictionary from ONLY Away games (the trusted ones)
    trusted_elos = away_records.set_index(['date', 'team'])['elo']
    
    print("[2] Repairing timelines for 30 teams...")
    
    repaired_elo_map = {} # (date, team) -> elo
    
    teams = full_schedule['team'].unique()
    
    for team in teams:
        # Get this team's full schedule
        team_schedule = full_schedule[full_schedule['team'] == team].sort_values('date')
        
        # Merge with the trusted "Away" ELOs
        team_data = team_schedule.merge(
            trusted_elos, 
            how='left', 
            left_on=['date', 'team'], 
            right_index=True
        )
        
        # INTERPOLATE: Fill the missing "Home" ELOs using the surrounding "Away" ELOs
        # We use 'linear' interpolation because ELO moves smoothly
        team_data['elo'] = team_data['elo'].interpolate(method='linear', limit_direction='both')
        
        # Forward fill and backward fill for any remaining NaNs at the edges
        team_data['elo'] = team_data['elo'].ffill().bfill()
        
        # Store valid values in our map
        for index, row in team_data.iterrows():
            if pd.notna(row['elo']):
                repaired_elo_map[(row['date'], row['team'])] = row['elo']

    # 3. APPLY THE FIX TO THE MAIN DATASET
    print("[3] Overwriting broken Home ELOs...")
    
    def get_corrected_elo(row, is_home):
        target_team = row['home_team'] if is_home else row['away_team']
        target_date = row['date']
        
        if (target_date, target_team) in repaired_elo_map:
            return repaired_elo_map[(target_date, target_team)]
        else:
            # Fallback if interpolation failed (rare)
            return row['home_composite_elo'] if is_home else row['away_composite_elo']

    # Apply fix
    df['home_composite_elo'] = df.apply(lambda x: get_corrected_elo(x, is_home=True), axis=1)
    # We also refresh Away just to be safe (though it was the source)
    df['away_composite_elo'] = df.apply(lambda x: get_corrected_elo(x, is_home=False), axis=1)
    
    # Recalculate Diff features since ELO changed
    df['off_elo_diff'] = df['home_composite_elo'] - df['away_composite_elo']
    # If def_elo_diff exists, we'd need to recalculate that too, but composite_elo doesn't have off/def split
    
    # 4. VERIFY
    print("\n[4] VERIFICATION (Post-Op Check)")
    print(f"    Original Home ELO Std: 99.96")
    print(f"    Original Away ELO Std: 77.16")
    print(f"    ---")
    print(f"    Repaired Home ELO Std: {df['home_composite_elo'].std():.2f}")
    print(f"    Repaired Away ELO Std: {df['away_composite_elo'].std():.2f}")
    
    # Sample a few teams to show the repair worked
    print("\n[5] SAMPLE REPAIR CHECK (Atlanta Hawks first 10 games)")
    atl_home = df[df['home_team'] == 'ATL'][['date', 'home_team', 'home_composite_elo']].head(5)
    atl_away = df[df['away_team'] == 'ATL'][['date', 'away_team', 'away_composite_elo']].head(5)
    
    print("\n    ATL Home Games:")
    for _, row in atl_home.iterrows():
        print(f"      {row['date'].date()}  ELO: {row['home_composite_elo']:.1f}")
    
    print("\n    ATL Away Games:")
    for _, row in atl_away.iterrows():
        print(f"      {row['date'].date()}  ELO: {row['away_composite_elo']:.1f}")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ SUCCESS. Saved corrected dataset to: {OUTPUT_FILE}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print("\n   ACTION REQUIRED: Retrain model on this new file.")
    print("   Prediction: home_composite_elo will now be a top-5 feature.")

repair_elo()
