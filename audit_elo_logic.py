import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'data/training_data_matchup_with_injury_advantage.csv'

def audit_elo_logic():
    print(f"===================================================================")
    print(f"ELO FORENSICS: REVERSE ENGINEERING THE FORMULA")
    print(f"===================================================================\n")
    
    df = pd.read_csv(DATA_PATH)
    
    # 1. restructure data to track specific teams
    # We want to see how 'BOS' looks when they are Home vs when they are Away
    
    # Extract Home games
    home_df = df[['date', 'home_team', 'home_composite_elo']].rename(columns={
        'home_team': 'team_id', 'home_composite_elo': 'elo_value'
    })
    home_df['location'] = 'HOME'
    
    # Extract Away games
    away_df = df[['date', 'away_team', 'away_composite_elo']].rename(columns={
        'away_team': 'team_id', 'away_composite_elo': 'elo_value'
    })
    away_df['location'] = 'AWAY'
    
    # Combine and sort
    elo_history = pd.concat([home_df, away_df]).sort_values(['team_id', 'date'])
    
    # 2. ANALYSIS: COMPARE THE SPREAD
    print(f"[1] Global Stats")
    print(f"    Home ELO Mean: {home_df['elo_value'].mean():.2f} | Std: {home_df['elo_value'].std():.2f}")
    print(f"    Away ELO Mean: {away_df['elo_value'].mean():.2f} | Std: {away_df['elo_value'].std():.2f}")
    
    # 3. ANALYSIS: "THE HANDICAP TEST"
    # We check if a team's ELO jumps instantly when they switch from Home to Away
    
    # Create lag features to compare "Today's Rating" vs "Last Game's Rating"
    elo_history['prev_elo'] = elo_history.groupby('team_id')['elo_value'].shift(1)
    elo_history['prev_loc'] = elo_history.groupby('team_id')['location'].shift(1)
    
    # Filter for Switchers (Home -> Away OR Away -> Home)
    switchers = elo_history[elo_history['location'] != elo_history['prev_loc']].dropna()
    
    # Calculate the Jump
    switchers['delta'] = switchers['elo_value'] - switchers['prev_elo']
    
    # Group by transition type
    h_to_a = switchers[switchers['location'] == 'AWAY']['delta']
    a_to_h = switchers[switchers['location'] == 'HOME']['delta']
    
    print(f"\n[2] The Handicap Test (What happens when a team travels?)")
    print(f"    Home -> Away Jump: {h_to_a.mean():.2f} points (avg)")
    print(f"    Away -> Home Jump: {a_to_h.mean():.2f} points (avg)")
    
    if abs(h_to_a.mean()) < 5:
        print(f"    👉 VERDICT: SINGLE THREADED ELO.")
        print(f"       Teams carry ONE rating. 'Home ELO' and 'Away ELO' are just the same number.")
        print(f"       The Std Dev difference is likely due to selection bias (better teams play more home games?) OR data skew.")
    else:
        print(f"    👉 VERDICT: SPLIT PERSONALITY ELO.")
        print(f"       The system tracks Home/Away performance separately.")
        print(f"       If Boston plays at Home, it uses Rating A. If they travel, it swaps to Rating B.")

    # 4. CORRELATION CHECK
    # Do Home ELO and Away ELO move together?
    # We can't compare same game, so we compare team averages
    team_avgs = elo_history.groupby(['team_id', 'location'])['elo_value'].mean().unstack()
    correlation = team_avgs['HOME'].corr(team_avgs['AWAY'])
    
    print(f"\n[3] Consistency Check")
    print(f"    Correlation between a team's Home & Away Rating: {correlation:.4f}")
    
    if correlation > 0.90:
        print("    👉 High Correlation: Good teams are good everywhere.")
    else:
        print("    👉 Low Correlation: Some teams are 'Home Tigers, Road Kittens'.")
    
    # 5. SAMPLE SPECIFIC TEAM (e.g., BOS)
    print(f"\n[4] Sample Team Analysis (First team in dataset)")
    sample_team = elo_history['team_id'].iloc[0]
    team_data = elo_history[elo_history['team_id'] == sample_team].head(10).copy()
    
    # Recalculate delta for the subset
    team_data['prev_elo'] = team_data['prev_elo'].fillna(0)
    team_data['delta'] = team_data['elo_value'] - team_data['prev_elo']
    team_data.loc[team_data['prev_elo'] == 0, 'delta'] = np.nan
    
    print(f"    Team: {sample_team}")
    print(f"\n    Date       Location  ELO    Prev ELO  Delta")
    print(f"    {'-'*50}")
    for _, row in team_data.iterrows():
        prev = row['prev_elo']
        delta = row['delta'] if pd.notna(row['delta']) else 0
        prev_str = f"{prev:.1f}" if prev != 0 else "N/A"
        delta_str = f"{delta:+.1f}" if pd.notna(row['delta']) else "N/A"
        print(f"    {row['date']}  {row['location']:5s}  {row['elo_value']:.1f}  {prev_str:>6s}  {delta_str:>6s}")

audit_elo_logic()
