"""Check actual ELO values in database"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

print("=" * 80)
print("ELO RATINGS IN DATABASE")
print("=" * 80)

# OKC ratings
print("\nOKC ELO ratings (2025-26 season):")
okc = pd.read_sql("""
    SELECT team, season, game_date, off_elo, def_elo, composite_elo 
    FROM elo_ratings 
    WHERE team='OKC' AND season='2025-26' 
    ORDER BY game_date DESC 
    LIMIT 10
""", conn)
print(okc.to_string())

# SAS ratings
print("\n\nSAS ELO ratings (2025-26 season):")
sas = pd.read_sql("""
    SELECT team, season, game_date, off_elo, def_elo, composite_elo 
    FROM elo_ratings 
    WHERE team='SAS' AND season='2025-26' 
    ORDER BY game_date DESC 
    LIMIT 10
""", conn)
print(sas.to_string())

# Check if there are ANY 2025-26 ratings
print("\n\nAll teams with 2025-26 ELO ratings:")
all_2025 = pd.read_sql("""
    SELECT DISTINCT team, COUNT(*) as games
    FROM elo_ratings 
    WHERE season='2025-26'
    GROUP BY team
    ORDER BY team
""", conn)
print(all_2025.to_string())

conn.close()

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if len(okc) == 0:
    print("❌ NO ELO RATINGS FOR OKC in 2025-26 season")
    print("   System will default to 1500 for missing data")

if len(sas) == 0:
    print("❌ NO ELO RATINGS FOR SAS in 2025-26 season")
    print("   System will default to 1500 for missing data")

if len(all_2025) == 0:
    print("❌ NO ELO RATINGS FOR ANY TEAM in 2025-26 season!")
    print("   ELO system has not been updated for current season")
    print("   This explains why composite_elo values are wrong")
