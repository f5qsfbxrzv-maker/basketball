"""
Analyze injury data coverage specifically for the training dataset (2023-2025 games)
"""

import pandas as pd
import sqlite3

DB_PATH = "data/live/nba_betting_data.db"

def analyze_training_coverage():
    conn = sqlite3.connect(DB_PATH)
    
    print("="*70)
    print("INJURY DATA COVERAGE FOR TRAINING DATASET")
    print("="*70)
    
    # Get all games in training period
    games_df = pd.read_sql("""
        SELECT 
            game_date,
            home_team,
            away_team
        FROM game_results
        WHERE game_date >= '2023-01-01' AND game_date < '2025-11-01'
        ORDER BY game_date
    """, conn)
    
    print(f"\n📊 TRAINING GAMES:")
    print(f"   Total games: {len(games_df):,}")
    print(f"   Date range: {games_df['game_date'].min()} to {games_df['game_date'].max()}")
    
    # Get injury data for same period
    injury_df = pd.read_sql("""
        SELECT 
            game_date,
            COUNT(*) as injuries
        FROM historical_inactives
        WHERE game_date >= '2023-01-01' AND game_date < '2025-11-01'
        GROUP BY game_date
    """, conn)
    
    print(f"\n🏥 INJURY DATA:")
    print(f"   Dates with injury data: {len(injury_df):,}")
    print(f"   Total injury records: {injury_df['injuries'].sum():,}")
    print(f"   Avg injuries per date: {injury_df['injuries'].mean():.1f}")
    
    # Calculate coverage percentage
    games_df['has_injury_data'] = games_df['game_date'].isin(injury_df['game_date'])
    coverage_pct = games_df['has_injury_data'].sum() / len(games_df) * 100
    
    print(f"\n📈 COVERAGE ANALYSIS:")
    print(f"   Games with injury data: {games_df['has_injury_data'].sum():,}")
    print(f"   Games without injury data: {(~games_df['has_injury_data']).sum():,}")
    print(f"   Coverage percentage: {coverage_pct:.1f}%")
    
    # Coverage by year
    games_df['year'] = pd.to_datetime(games_df['game_date']).dt.year
    coverage_by_year = games_df.groupby('year').agg({
        'game_date': 'count',
        'has_injury_data': 'sum'
    })
    coverage_by_year['coverage_pct'] = (coverage_by_year['has_injury_data'] / coverage_by_year['game_date'] * 100)
    coverage_by_year.columns = ['total_games', 'games_with_data', 'coverage_pct']
    
    print(f"\n📅 COVERAGE BY YEAR:")
    print(coverage_by_year.to_string())
    
    # Coverage by month
    games_df['month'] = pd.to_datetime(games_df['game_date']).dt.to_period('M')
    coverage_by_month = games_df.groupby('month').agg({
        'game_date': 'count',
        'has_injury_data': 'sum'
    })
    coverage_by_month['coverage_pct'] = (coverage_by_month['has_injury_data'] / coverage_by_month['game_date'] * 100)
    coverage_by_month.columns = ['total_games', 'games_with_data', 'coverage_pct']
    
    print(f"\n📆 COVERAGE BY MONTH (showing gaps <50%):")
    low_coverage = coverage_by_month[coverage_by_month['coverage_pct'] < 50]
    if len(low_coverage) > 0:
        print(low_coverage.to_string())
    else:
        print("   ✅ No months with <50% coverage!")
    
    # Show recent coverage
    print(f"\n📰 MOST RECENT 10 MONTHS:")
    print(coverage_by_month.tail(10).to_string())
    
    # Estimate for 3,224 training games
    print(f"\n🎯 IMPACT ON 3,224 TRAINING GAMES:")
    estimated_coverage = int(3224 * (coverage_pct / 100))
    estimated_missing = 3224 - estimated_coverage
    print(f"   Games with injury data: ~{estimated_coverage:,} ({coverage_pct:.1f}%)")
    print(f"   Games without injury data: ~{estimated_missing:,} ({100-coverage_pct:.1f}%)")
    
    if coverage_pct > 80:
        print(f"\n   ✅ EXCELLENT COVERAGE - Injury features will be meaningful!")
    elif coverage_pct > 50:
        print(f"\n   ⚠️ GOOD COVERAGE - Injury features will have partial signal")
    else:
        print(f"\n   ❌ POOR COVERAGE - Injury features will have limited signal")
    
    conn.close()

if __name__ == "__main__":
    analyze_training_coverage()
