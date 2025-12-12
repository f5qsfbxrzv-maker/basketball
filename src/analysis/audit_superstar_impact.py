"""
Audit how injury impact values correlate with actual game outcomes
to verify VORP/superstar weighting is working correctly
"""

import pandas as pd
import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.features.feature_calculator_v5 import FeatureCalculatorV5

DB_PATH = "data/live/nba_betting_data.db"

def audit_superstar_impacts():
    print("="*80)
    print("🕵️‍♂️ AUDITING SUPERSTAR INJURY IMPACT")
    print("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Load games from training period
    games_df = pd.read_sql("""
        SELECT game_date, home_team, away_team, home_score, away_score
        FROM game_results
        WHERE game_date >= '2023-10-01' AND game_date < '2025-11-01'
        ORDER BY game_date DESC
    """, conn)
    
    conn.close()
    
    print(f"\n📊 Loaded {len(games_df):,} games")
    print("Calculating injury impacts...")
    
    calc = FeatureCalculatorV5()
    
    # Calculate features for all games
    results = []
    for idx, row in games_df.iterrows():
        if idx % 200 == 0:
            print(f"   Progress: {idx}/{len(games_df)}")
        
        features = calc.calculate_game_features(
            row['home_team'], 
            row['away_team'],
            game_date=row['game_date']  # FIXED: Use keyword argument
        )
        
        injury_diff = features.get('injury_impact_diff', 0)
        injury_abs = features.get('injury_impact_abs', 0)
        injury_elo_int = features.get('injury_elo_interaction', 0)
        
        # Calculate actual outcome
        home_win = 1 if row['home_score'] > row['away_score'] else 0
        margin = row['home_score'] - row['away_score']
        
        results.append({
            'date': row['game_date'],
            'matchup': f"{row['away_team']} @ {row['home_team']}",
            'injury_diff': injury_diff,
            'injury_abs': injury_abs,
            'injury_elo_int': injury_elo_int,
            'home_win': home_win,
            'margin': margin,
            'home_score': row['home_score'],
            'away_score': row['away_score']
        })
    
    df = pd.DataFrame(results)
    
    # Analysis
    print("\n" + "="*80)
    print("📈 INJURY IMPACT STATISTICS")
    print("="*80)
    
    print(f"\ninjury_impact_diff:")
    print(f"   Mean: {df['injury_diff'].mean():.4f}")
    print(f"   Std:  {df['injury_diff'].std():.4f}")
    print(f"   Min:  {df['injury_diff'].min():.4f}")
    print(f"   Max:  {df['injury_diff'].max():.4f}")
    print(f"   Non-zero: {(df['injury_diff'] != 0).sum()} games ({(df['injury_diff'] != 0).sum()/len(df)*100:.1f}%)")
    
    print(f"\ninjury_impact_abs:")
    print(f"   Mean: {df['injury_abs'].mean():.4f}")
    print(f"   Std:  {df['injury_abs'].std():.4f}")
    print(f"   Max:  {df['injury_abs'].max():.4f}")
    
    # Top 20 games by absolute injury impact
    print("\n" + "="*80)
    print("🏆 TOP 20 GAMES BY INJURY IMPACT (Absolute)")
    print("="*80)
    print(f"{'Date':<12} {'Matchup':<22} {'Inj Diff':<10} {'Inj Abs':<10} {'Score':<12} {'Margin'}")
    print("-"*80)
    
    top20 = df.nlargest(20, 'injury_abs')
    for _, row in top20.iterrows():
        score = f"{row['away_score']}-{row['home_score']}"
        print(f"{row['date']:<12} {row['matchup']:<22} {row['injury_diff']:>9.2f} {row['injury_abs']:>9.2f} {score:<12} {row['margin']:>6.0f}")
    
    # Correlation analysis
    print("\n" + "="*80)
    print("🔍 PREDICTIVE POWER ANALYSIS")
    print("="*80)
    
    # Does injury_diff predict home wins?
    # Positive injury_diff = home team more injured = home team should lose
    # So we expect negative correlation with home_win
    
    corr_win = df['injury_diff'].corr(df['home_win'])
    corr_margin = df['injury_diff'].corr(df['margin'])
    
    print(f"\nCorrelation with home team winning:")
    print(f"   injury_diff vs home_win:   {corr_win:.4f}")
    print(f"   injury_diff vs home_margin: {corr_margin:.4f}")
    
    if abs(corr_win) < 0.05:
        print(f"   ⚠️ WEAK CORRELATION - Injury features may not be predictive")
    elif abs(corr_win) < 0.10:
        print(f"   📊 MODERATE CORRELATION - Injury features have some signal")
    else:
        print(f"   ✅ STRONG CORRELATION - Injury features are predictive")
    
    # Check if the sign is correct
    if corr_win < 0:
        print(f"   ✅ CORRECT DIRECTION - More home injuries → less likely to win")
    else:
        print(f"   ❌ WRONG DIRECTION - Sign is flipped, need to debug")
    
    # Games with extreme injury differential
    print("\n" + "="*80)
    print("🔥 EXTREME INJURY DIFFERENTIALS")
    print("="*80)
    
    extreme_home_injured = df.nsmallest(10, 'injury_diff')  # Most negative = home team crushed
    extreme_away_injured = df.nlargest(10, 'injury_diff')   # Most positive = away team crushed
    
    print(f"\n📉 Top 10: Home Team Most Injured (injury_diff < 0)")
    print(f"{'Date':<12} {'Matchup':<22} {'Inj Diff':<10} {'Result':<12} {'Expected'}")
    print("-"*70)
    for _, row in extreme_home_injured.iterrows():
        result = "HOME" if row['home_win'] else "AWAY"
        expected = "AWAY" if row['injury_diff'] < 0 else "HOME"
        correct = "✅" if result == expected else "❌"
        score = f"{row['away_score']}-{row['home_score']}"
        print(f"{row['date']:<12} {row['matchup']:<22} {row['injury_diff']:>9.2f} {result:<12} {expected} {correct}")
    
    print(f"\n📈 Top 10: Away Team Most Injured (injury_diff > 0)")
    print(f"{'Date':<12} {'Matchup':<22} {'Inj Diff':<10} {'Result':<12} {'Expected'}")
    print("-"*70)
    for _, row in extreme_away_injured.iterrows():
        result = "HOME" if row['home_win'] else "AWAY"
        expected = "HOME" if row['injury_diff'] > 0 else "AWAY"
        correct = "✅" if result == expected else "❌"
        score = f"{row['away_score']}-{row['home_score']}"
        print(f"{row['date']:<12} {row['matchup']:<22} {row['injury_diff']:>9.2f} {result:<12} {expected} {correct}")
    
    # Calculate accuracy on extreme cases
    home_injured_correct = (extreme_home_injured['home_win'] == 0).sum()
    away_injured_correct = (extreme_away_injured['home_win'] == 1).sum()
    total_correct = home_injured_correct + away_injured_correct
    total_extreme = len(extreme_home_injured) + len(extreme_away_injured)
    
    print(f"\n📊 ACCURACY ON EXTREME INJURY GAMES:")
    print(f"   Predicted correctly: {total_correct}/{total_extreme} ({total_correct/total_extreme*100:.1f}%)")
    
    if total_correct/total_extreme > 0.60:
        print(f"   ✅ GOOD SIGNAL - Injury features are working!")
    elif total_correct/total_extreme > 0.50:
        print(f"   ⚠️ WEAK SIGNAL - Better than random but needs improvement")
    else:
        print(f"   ❌ NO SIGNAL - Injury features not predictive, need to debug VORP")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    audit_superstar_impacts()
