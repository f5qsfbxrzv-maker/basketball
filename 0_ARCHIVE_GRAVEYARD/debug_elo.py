"""Debug ELO ratings for OKC and SAS"""
import sqlite3
import pandas as pd
from src.features.off_def_elo_system import OffDefEloSystem

print("=" * 80)
print("ELO SYSTEM DIAGNOSTIC")
print("=" * 80)

# Check what's in the database
conn = sqlite3.connect('data/live/nba_betting_data.db')

# Check if elo_ratings table exists
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%elo%'", conn)
print(f"\nELO-related tables: {tables['name'].tolist()}")

# Check recent ELO ratings for OKC and SAS
if 'elo_ratings' in tables['name'].tolist():
    okc_elo = pd.read_sql("""
        SELECT team_abb, season, as_of_date, off_elo, def_elo, composite_elo
        FROM elo_ratings 
        WHERE team_abb IN ('OKC', 'SAS')
        AND season = '2025-26'
        ORDER BY as_of_date DESC
        LIMIT 10
    """, conn)
    
    print("\nRecent ELO ratings from database:")
    print(okc_elo.to_string())
else:
    print("\n⚠️  No elo_ratings table found!")

conn.close()

# Test the OffDefEloSystem
print("\n" + "=" * 80)
print("TESTING OffDefEloSystem")
print("=" * 80)

elo_system = OffDefEloSystem(db_path='data/live/nba_betting_data.db')

# Get ELO for OKC
okc_rating = elo_system.get_latest('OKC', season='2025-26', before_date='2025-12-13')
print(f"\nOKC rating object: {okc_rating}")
if okc_rating:
    print(f"  off_elo: {okc_rating.off_elo}")
    print(f"  def_elo: {okc_rating.def_elo}")
    print(f"  composite: {okc_rating.composite}")

# Get ELO for SAS
sas_rating = elo_system.get_latest('SAS', season='2025-26', before_date='2025-12-13')
print(f"\nSAS rating object: {sas_rating}")
if sas_rating:
    print(f"  off_elo: {sas_rating.off_elo}")
    print(f"  def_elo: {sas_rating.def_elo}")
    print(f"  composite: {sas_rating.composite}")

# Calculate differentials
if okc_rating and sas_rating:
    print(f"\n" + "=" * 80)
    print("CALCULATED DIFFERENTIALS (Home - Away)")
    print("=" * 80)
    print(f"off_elo_diff = {okc_rating.off_elo} - {sas_rating.off_elo} = {okc_rating.off_elo - sas_rating.off_elo:.2f}")
    print(f"def_elo_diff = {okc_rating.def_elo} - {sas_rating.def_elo} = {okc_rating.def_elo - sas_rating.def_elo:.2f}")
    print(f"composite_diff = {okc_rating.composite} - {sas_rating.composite} = {okc_rating.composite - sas_rating.composite:.2f}")
    
    print(f"\nExpected feature values:")
    print(f"  off_elo_diff: {okc_rating.off_elo - sas_rating.off_elo:.2f}")
    print(f"  def_elo_diff: {okc_rating.def_elo - sas_rating.def_elo:.2f}")
    print(f"  home_composite_elo: {okc_rating.composite:.2f}")

# Check what the feature calculator actually does
print("\n" + "=" * 80)
print("FEATURE CALCULATOR ELO LOGIC")
print("=" * 80)

from src.features.feature_calculator_live import FeatureCalculatorV5
calc = FeatureCalculatorV5()

# Trace through the ELO feature calculation
feats = calc.calculate_game_features('OKC', 'SAS', game_date='2025-12-13')
print(f"\nActual feature values:")
print(f"  off_elo_diff: {feats.get('off_elo_diff', 'MISSING')}")
print(f"  def_elo_diff: {feats.get('def_elo_diff', 'MISSING')}")
print(f"  home_composite_elo: {feats.get('home_composite_elo', 'MISSING')}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
if okc_rating and sas_rating:
    if okc_rating.composite < 1000:
        print("⚠️  OKC composite ELO < 1000 - This is VERY LOW for reigning champs")
        print("    Standard ELO baseline is 1500, good teams should be 1600+")
    
    if abs(okc_rating.def_elo - sas_rating.def_elo) > 200:
        print("⚠️  Defensive ELO differential > 200 points - EXTREME")
        print("    Even Warriors dynasty vs tanking team was ~150 points")
    
    if sas_rating.def_elo > okc_rating.def_elo:
        print(f"⚠️  SAS def_elo ({sas_rating.def_elo:.1f}) > OKC def_elo ({okc_rating.def_elo:.1f})")
        print("    This means SAS is rated as better defense than OKC")
        print("    While Wemby is DPOY, OKC is also elite defensively")
