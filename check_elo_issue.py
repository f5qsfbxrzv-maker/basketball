"""Check the current ELO ratings for DAL and DET"""
import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path('data/live/nba_betting_data.db')
conn = sqlite3.connect(db_path)

# Check table schema
print("ELO_RATINGS TABLE SCHEMA:")
schema = conn.execute("PRAGMA table_info(elo_ratings)").fetchall()
for col in schema:
    print(f"  {col}")

print("\n" + "="*80)

# Get LATEST ELO for each team
print("\nLATEST ELO RATINGS (highest composite_elo per team):")
query = """
SELECT team, MAX(composite_elo) as max_elo
FROM elo_ratings
WHERE team IN ('DAL', 'DET')
GROUP BY team
ORDER BY max_elo DESC
"""
df = pd.read_sql_query(query, conn)
print(df.to_string(index=False))

print("\n" + "="*80)

# Count entries per team
print("\nNUMBER OF ELO ENTRIES PER TEAM:")
query = """
SELECT team, COUNT(*) as entry_count
FROM elo_ratings
WHERE team IN ('DAL', 'DET')
GROUP BY team
"""
count_df = pd.read_sql_query(query, conn)
print(count_df.to_string(index=False))

conn.close()

print("\n" + "="*80)
print("DIAGNOSIS:")
print("  The elo_ratings table has multiple entries per team!")
print("  This suggests either:")
print("    1. Historical ELO values are being stored (one per game)")
print("    2. Duplicate calculations are being inserted")
print("  The feature calculator likely needs to use MAX(composite_elo) or")
print("  filter by date to get the current ELO rating.")
