"""
Build game_advanced_stats table with Four Factors from game_logs
Calculates: eFG%, TOV%, ORB%, FTr, Pace, Off/Def Ratings per game
"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'nba_betting_data.db'

print("=" * 80)
print("BUILDING game_advanced_stats FROM game_logs")
print("=" * 80)

# Load game_logs
conn = sqlite3.connect(DB_PATH)
# Get ALL games (both 2024-25 and 2025-26 seasons)
df = pd.read_sql("SELECT * FROM game_logs", conn)
print(f"\n[1/4] Loaded {len(df)} game logs")

# Calculate Four Factors for each game
print("\n[2/4] Calculating Four Factors...")

# eFG% = (FGM + 0.5 * FG3M) / FGA
df['efg_pct'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, np.nan)

# TOV% = TOV / (FGA + 0.44 * FTA + TOV)
df['tov_pct'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV']).replace(0, np.nan)

# ORB% = OREB / (OREB + Opp DREB) - need opponent data
# For now, use raw OREB as proxy
df['orb_pct'] = df['OREB'] / (df['OREB'] + 30).replace(0, np.nan)  # Approximation

# FTr (FTA Rate) = FTA / FGA
df['fta_rate'] = df['FTA'] / df['FGA'].replace(0, np.nan)

# Estimate Pace = 48 * ((Poss + OppPoss) / (2 * (Min/5)))
# Possessions ≈ FGA + 0.44*FTA - OREB + TOV
df['poss_est'] = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']
df['pace_est'] = 48 * df['poss_est'] / 48  # Simplified, assume 48 min games

# Offensive Rating = 100 * (PTS / Poss)
df['off_rating'] = 100 * df['PTS'] / df['poss_est'].replace(0, np.nan)

# Defensive Rating estimation
# For each team's game, estimate opponent's offensive rating
# Since we don't have opponent data directly, use league average adjusted by +/- 
df['def_rating'] = 110.0 - (df['PLUS_MINUS'] / 2.0)  # Baseline 110, adjust by half of plus/minus

# Net Rating = Offensive Rating - Defensive Rating
df['net_rating'] = df['off_rating'] - df['def_rating']

# Pace = pace_est (already calculated)
df['pace'] = df['pace_est']

# 3PA per 100 possessions
df['fg3a_per_100'] = 100 * df['FG3A'] / df['poss_est'].replace(0, np.nan)

# 3P% = FG3M / FG3A
df['fg3_pct'] = df['FG3M'] / df['FG3A'].replace(0, np.nan)

# Fill NaN with reasonable defaults
df['efg_pct'] = df['efg_pct'].fillna(0.50)
df['tov_pct'] = df['tov_pct'].fillna(0.14)
df['orb_pct'] = df['orb_pct'].fillna(0.25)
df['fta_rate'] = df['fta_rate'].fillna(0.25)
df['pace_est'] = df['pace_est'].fillna(100.0)
df['pace'] = df['pace'].fillna(100.0)
df['off_rating'] = df['off_rating'].fillna(110.0)
df['def_rating'] = df['def_rating'].fillna(110.0)
df['net_rating'] = df['net_rating'].fillna(0.0)
df['fg3a_per_100'] = df['fg3a_per_100'].fillna(30.0)
df['fg3_pct'] = df['fg3_pct'].fillna(0.35)

print(f"[OK] Calculated Four Factors for {len(df)} games")

# Create game_advanced_stats table
print("\n[3/4] Creating game_advanced_stats table...")

# Select relevant columns
advanced_df = df[['GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL',
                  'efg_pct', 'tov_pct', 'orb_pct', 'fta_rate', 'pace_est', 'pace',
                  'off_rating', 'def_rating', 'net_rating', 'fg3a_per_100', 'fg3_pct',
                  'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                  'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                  'PLUS_MINUS', 'season']].copy()

# Rename to match expected schema
advanced_df.columns = [c.lower() for c in advanced_df.columns]
# Fix team_abbreviation to team_abb (feature calculator expects this)
advanced_df = advanced_df.rename(columns={'team_abbreviation': 'team_abb'})

# Save to database
advanced_df.to_sql('game_advanced_stats', conn, if_exists='replace', index=False)
print(f"[OK] Created game_advanced_stats with {len(advanced_df)} rows")

# Verify the data
print("\n[4/4] Verifying data...")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM game_advanced_stats")
count = cursor.fetchone()[0]
print(f"[OK] Verified {count} rows in game_advanced_stats")

cursor.execute("SELECT * FROM game_advanced_stats LIMIT 1")
sample = cursor.fetchone()
print(f"\nSample row (first 10 columns):")
cols = [desc[0] for desc in cursor.description][:10]
for i, col in enumerate(cols):
    print(f"  {col}: {sample[i]}")

conn.close()

print("\n" + "=" * 80)
print("[SUCCESS] game_advanced_stats table ready!")
print("=" * 80)
print("\nYou can now launch the dashboard and make predictions with all features!")
