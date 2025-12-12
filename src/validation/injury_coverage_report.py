"""
Comprehensive injury data coverage report after CSV import
"""

import pandas as pd
import sqlite3
from datetime import datetime

DB_PATH = "data/live/nba_betting_data.db"

def analyze_coverage():
    conn = sqlite3.connect(DB_PATH)
    
    print("="*70)
    print("INJURY DATA COVERAGE REPORT")
    print("="*70)
    
    # Overall stats
    df = pd.read_sql("""
        SELECT 
            MIN(game_date) as min_date,
            MAX(game_date) as max_date,
            COUNT(*) as total_records,
            COUNT(DISTINCT game_date) as unique_dates,
            COUNT(DISTINCT player_name) as unique_players
        FROM historical_inactives
    """, conn)
    
    print("\n📊 OVERALL COVERAGE:")
    print(f"   Date range: {df['min_date'][0]} to {df['max_date'][0]}")
    print(f"   Total records: {df['total_records'][0]:,}")
    print(f"   Unique dates: {df['unique_dates'][0]:,}")
    print(f"   Unique players: {df['unique_players'][0]:,}")
    
    # Coverage by season
    df_season = pd.read_sql("""
        SELECT 
            season,
            COUNT(*) as records,
            MIN(game_date) as start_date,
            MAX(game_date) as end_date,
            COUNT(DISTINCT player_name) as players
        FROM historical_inactives
        WHERE season IS NOT NULL
        GROUP BY season
        ORDER BY season DESC
    """, conn)
    
    print("\n📅 COVERAGE BY SEASON:")
    print(df_season.to_string(index=False))
    
    # Coverage by year
    df_year = pd.read_sql("""
        SELECT 
            strftime('%Y', game_date) as year,
            COUNT(*) as records,
            COUNT(DISTINCT player_name) as unique_players
        FROM historical_inactives
        GROUP BY year
        ORDER BY year DESC
    """, conn)
    
    print("\n📆 COVERAGE BY YEAR:")
    print(df_year.to_string(index=False))
    
    # Training period coverage (2023-2025)
    df_training = pd.read_sql("""
        SELECT 
            strftime('%Y-%m', game_date) as month,
            COUNT(*) as records
        FROM historical_inactives
        WHERE game_date >= '2023-01-01' AND game_date <= '2025-11-01'
        GROUP BY month
        ORDER BY month
    """, conn)
    
    print("\n🎯 TRAINING PERIOD COVERAGE (2023-01 to 2025-11):")
    print(f"   Total months: {len(df_training)}")
    print(f"   Total records: {df_training['records'].sum():,}")
    print(f"   Average records/month: {df_training['records'].mean():.0f}")
    
    # Gap analysis
    print("\n🔍 GAP ANALYSIS:")
    gaps_df = pd.read_sql("""
        SELECT 
            strftime('%Y-%m', game_date) as month,
            COUNT(*) as records
        FROM historical_inactives
        WHERE game_date >= '2023-01-01'
        GROUP BY month
        ORDER BY month
    """, conn)
    
    # Find months with low coverage
    avg_records = gaps_df['records'].mean()
    low_months = gaps_df[gaps_df['records'] < avg_records * 0.3]
    if len(low_months) > 0:
        print(f"   ⚠️ Months with <30% avg coverage:")
        for _, row in low_months.iterrows():
            print(f"      {row['month']}: {row['records']} records")
    else:
        print("   ✅ No significant gaps detected!")
    
    # Most recent data
    recent = pd.read_sql("""
        SELECT game_date, COUNT(*) as count
        FROM historical_inactives
        WHERE game_date >= date('now', '-30 days')
        GROUP BY game_date
        ORDER BY game_date DESC
        LIMIT 10
    """, conn)
    
    print("\n📰 MOST RECENT DATA (Last 30 days):")
    if len(recent) > 0:
        print(recent.to_string(index=False))
    else:
        print("   No data in last 30 days")
    
    # Compare pre-import vs post-import
    old_coverage = pd.read_sql("""
        SELECT COUNT(*) as count
        FROM historical_inactives
        WHERE game_date <= '2023-04-09'
    """, conn)
    
    new_coverage = pd.read_sql("""
        SELECT COUNT(*) as count
        FROM historical_inactives
        WHERE game_date > '2023-04-09'
    """, conn)
    
    print("\n📈 BEFORE vs AFTER CSV IMPORT:")
    print(f"   Before (through 2023-04-09): {old_coverage['count'][0]:,} records")
    print(f"   After (2023-04-10+): {new_coverage['count'][0]:,} records")
    print(f"   Total improvement: +{new_coverage['count'][0]:,} records ({new_coverage['count'][0] / old_coverage['count'][0] * 100:.1f}% increase)")
    
    conn.close()

if __name__ == "__main__":
    analyze_coverage()
