"""
FORENSIC AUDIT: Busted Favorites Investigation
===============================================
Analyzes favorites that looked good (positive edge) but LOST
to identify statistical patterns and improve betting filters
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# CONFIG
DATA_PATH = 'data/training_data_GOLD_ELO_22_features.csv'
CALIBRATOR_PATH = 'models/nba_isotonic_calibrator.joblib'
ODDS_2023 = 'data/closing_odds_2023_24_CLEANED.csv'
ODDS_2024 = 'data/closing_odds_2024_25_CLEANED.csv'
SPLIT_DATE = '2023-10-01'  # Test period

# FEATURES TO AUDIT (Looking for discrepancies)
AUDIT_FEATURES = [
    'off_elo_diff',
    'def_elo_diff', 
    'net_fatigue_score',
    'ewma_pace_diff',
    'home_composite_elo',
    'injury_impact_diff',
    'season_progress',
    'projected_possession_margin',
    'star_power_leverage',
    'ewma_chaos_home'
]

ACTIVE_FEATURES = [
    'home_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'projected_possession_margin', 'ewma_pace_diff', 'net_fatigue_score',
    'ewma_efg_diff', 'ewma_vol_3p_diff', 'three_point_matchup',
    'ewma_chaos_home', 'injury_impact_diff', 'injury_shock_diff',
    'star_power_leverage', 'season_progress', 'league_offensive_context',
    'total_foul_environment', 'net_free_throw_advantage',
    'pace_efficiency_interaction', 'offense_vs_defense_matchup'
]

def american_to_implied_prob(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def audit_busted_favorites():
    print("="*100)
    print("🕵️  FORENSIC AUDIT: BUSTED FAVORITES INVESTIGATION")
    print("="*100)
    print("Objective: Find the statistical fingerprint of favorites that looked good but LOST")
    print("="*100)
    
    # 1. Load Data
    print("\n1️⃣  Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['game_date'] = pd.to_datetime(df['date'])
    
    # Load odds
    odds_2023 = pd.read_csv(ODDS_2023)
    odds_2024 = pd.read_csv(ODDS_2024)
    all_odds = pd.concat([odds_2023, odds_2024], ignore_index=True)
    all_odds['game_date'] = pd.to_datetime(all_odds['game_date'])
    
    # Merge odds
    df = df.merge(
        all_odds[['game_date', 'home_team', 'away_team', 'home_ml_odds', 'away_ml_odds']],
        on=['game_date', 'home_team', 'away_team'],
        how='inner'
    )
    
    print(f"✓ Loaded {len(df):,} games with odds")
    
    # 2. Calculate implied probabilities
    df['home_implied'] = df['home_ml_odds'].apply(american_to_implied_prob)
    df['away_implied'] = df['away_ml_odds'].apply(american_to_implied_prob)
    df['home_is_vegas_fav'] = df['home_implied'] > df['away_implied']
    
    # 3. Load pre-trained model with calibration (from DEFINITIVE_COMPARISON)
    print("\n2️⃣  Loading model and generating predictions...")
    
    # Train on pre-2023 data
    train_df = df[df['game_date'] < SPLIT_DATE].copy()
    X_train = train_df[ACTIVE_FEATURES].copy()
    y_train = train_df['target_moneyline_win'].copy()
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train, y_train = X_train[mask], y_train[mask]
    
    model = xgb.XGBClassifier(
        learning_rate=0.066994, max_depth=2, n_estimators=4529,
        random_state=42, objective='binary:logistic',
        base_score=0.5
    )
    model.fit(X_train, y_train, verbose=False)
    
    # Use the K-fold calibrator (trained on all historical data)
    calibrator = joblib.load(CALIBRATOR_PATH)
    
    # Predict on full dataset (we'll filter to test period after)
    X_all = df[ACTIVE_FEATURES].copy()
    mask = ~X_all.isna().any(axis=1)
    df_clean = df[mask].copy()
    X_all = X_all[mask]
    
    raw_probs = model.predict_proba(X_all)[:, 1]
    df_clean['model_prob'] = calibrator.predict(raw_probs)
    
    print(f"✓ Generated predictions for {len(df_clean):,} games")
    
    # Filter to test period
    recent = df_clean[df_clean['game_date'] >= SPLIT_DATE].copy()
    print(f"✓ Analyzing test period: {len(recent):,} games from {SPLIT_DATE} onwards")
    
    # 5. Calculate edges for BOTH sides
    recent['home_edge'] = recent['model_prob'] - recent['home_implied']
    recent['away_edge'] = (1 - recent['model_prob']) - recent['away_implied']
    
    # Find ALL bets where we had positive edge (regardless of if Vegas agreed)
    recent['bet_side'] = np.where(recent['home_edge'] > recent['away_edge'], 'HOME', 'AWAY')
    recent['bet_edge'] = np.where(recent['home_edge'] > recent['away_edge'], recent['home_edge'], recent['away_edge'])
    recent['bet_won'] = np.where(
        recent['bet_side'] == 'HOME',
        recent['target_moneyline_win'] == 1,
        recent['target_moneyline_win'] == 0
    )
    
    print(f"\n4️⃣  Edge statistics:")
    print(f"   Bet edge range: {recent['bet_edge'].min():.3f} to {recent['bet_edge'].max():.3f}")
    print(f"   Mean bet edge: {recent['bet_edge'].mean():.3f}")
    print(f"   Positive edges: {(recent['bet_edge'] > 0).sum():,}")
    
    # 6. Filter to bets with positive edge above threshold
    EDGE_THRESHOLD = 0.035  # 3.5% - the optimal threshold we found
    candidates = recent[recent['bet_edge'] > EDGE_THRESHOLD].copy()
    
    # Split into SUCCESS vs FAILURE
    winners = candidates[candidates['bet_won'] == True].copy()
    losers = candidates[candidates['bet_won'] == False].copy()  # <--- THE CRIME SCENE
    
    # Tag if bet was on favorite or underdog
    candidates['bet_on_favorite'] = np.where(
        candidates['bet_side'] == 'HOME',
        candidates['home_is_vegas_fav'],
        ~candidates['home_is_vegas_fav']
    )
    
    winners['bet_on_favorite'] = np.where(
        winners['bet_side'] == 'HOME',
        winners['home_is_vegas_fav'],
        ~winners['home_is_vegas_fav']
    )
    
    losers['bet_on_favorite'] = np.where(
        losers['bet_side'] == 'HOME',
        losers['home_is_vegas_fav'],
        ~losers['home_is_vegas_fav']
    )
    
    print("\n" + "="*100)
    print("📊 DATASET BREAKDOWN")
    print("="*100)
    print(f"Total games analyzed: {len(recent):,}")
    print(f"Bets with >{EDGE_THRESHOLD:.1%} edge: {len(candidates):,}")
    
    if len(candidates) == 0:
        print("\n⚠️  NO CANDIDATES FOUND! Try lowering EDGE_THRESHOLD in the script.")
        return
    
    print("\n" + "="*100)
    print("📊 DATASET BREAKDOWN")
    print("="*100)
    print(f"Total 'Positive Edge' Favorites: {len(candidates):,}")
    print(f"   ✅ Winners: {len(winners):,} ({len(winners)/len(candidates)*100:.1f}%)")
    print(f"   ❌ Busted:  {len(losers):,} ({len(losers)/len(candidates)*100:.1f}%) ← INVESTIGATING THESE")
    print(f"   Win Rate: {len(winners)/len(candidates)*100:.1f}%")
    print()
    print(f"Breakdown by Bet Type:")
    print(f"   Winners on Favorites: {(winners['bet_on_favorite']).sum()}")
    print(f"   Winners on Underdogs: {(~winners['bet_on_favorite']).sum()}")
    print(f"   Busts on Favorites: {(losers['bet_on_favorite']).sum()}")
    print(f"   Busts on Underdogs: {(~losers['bet_on_favorite']).sum()}")
    
    if len(losers) == 0:
        print("\n✨ No busted bets found! Model is perfect (or sample too small)")
    # ANALYSIS 1: FEATURE FINGERPRINT (Winners vs. Losers)
    # =================================================================
    print("\n" + "="*100)
    print("🔍 ANALYSIS 1: FEATURE FINGERPRINT (Winners vs. Losers)")
    print("="*100)
    print("Looking for features where BUSTED bets differ significantly from WINNERS")
    print()
    
    # Focus on HOME bets only for clean feature comparison
    # (Otherwise we'd need to flip signs for away bets)
    home_bet_winners = winners[winners['bet_side'] == 'HOME'].copy()
    home_bet_losers = losers[losers['bet_side'] == 'HOME'].copy()
    
    print(f"\nNote: Analyzing {len(home_bet_winners)} home bet WINS vs {len(home_bet_losers)} home bet BUSTS\n")
    
    print(f"{'FEATURE':<35} | {'WIN AVG':<10} | {'LOSS AVG':<10} | {'DIFF %':<10} | {'STATUS'}")
    print("-"*100)
    
    clues_found = []
    
    for feat in AUDIT_FEATURES:
        if feat not in home_bet_winners.columns:
            continue
            
        mean_win = home_bet_winners[feat].mean()
        mean_loss = home_bet_losers[feat].mean()
        diff = mean_loss - mean_win
        diff_pct = (diff / abs(mean_win) * 100) if mean_win != 0 else 0
        
        status = ""
        interpretation = ""
        
        if abs(diff_pct) > 15:
            status = "🔴 MAJOR CLUE"
            clues_found.append((feat, diff_pct, diff))
        elif abs(diff_pct) > 10:
            status = "🟡 CLUE"
            clues_found.append((feat, diff_pct, diff))
        
        print(f"{feat:<35} | {mean_win:>10.3f} | {mean_loss:>10.3f} | {diff_pct:>+9.1f}% | {status}")
    
    # Interpretation Guide
    print("\n" + "-"*100)
    print("💡 INTERPRETATION GUIDE:")
    print("-"*100)
    print("• net_fatigue_score HIGHER in busts → Model ignores fatigue (tired teams losing)")
    print("• off_elo_diff LOWER in busts → Betting weak favorites with small advantages")
    print("• season_progress LOWER in busts → Early season instability (ELO not settled)")
    print("• injury_impact_diff NEGATIVE in busts → Ignoring injury disadvantages")
    print("• ewma_chaos_home HIGHER in busts → Betting volatile/inconsistent home teams")
    
    # =================================================================
    # ANALYSIS 2: ODDS INTEGRITY CHECK
    # =================================================================
    print("\n" + "="*100)
    print("🔍 ANALYSIS 2: ODDS INTEGRITY CHECK")
    print("="*100)
    print("Are we betting on extreme favorites (-500+) or suspicious lines?")
    print()
    
    # Get odds for the bet side
    losers['bet_ml'] = np.where(
        losers['bet_side'] == 'HOME',
        losers['home_ml_odds'],
        losers['away_ml_odds']
    )
    
    winners['bet_ml'] = np.where(
        winners['bet_side'] == 'HOME',
        winners['home_ml_odds'],
        winners['away_ml_odds']
    )
    
    avg_win_odds = winners['bet_ml'].mean()
    avg_loss_odds = losers['bet_ml'].mean()
    
    extreme_wins = len(winners[winners['bet_ml'] < -500])
    extreme_losses = len(losers[losers['bet_ml'] < -500])
    
    print(f"Average Odds:")
    print(f"   Winning Bets: {avg_win_odds:.1f}")
    print(f"   Busted Bets:  {avg_loss_odds:.1f}")
    print(f"   Difference:        {avg_loss_odds - avg_win_odds:+.1f} (negative = busts were heavier favorites)")
    print()
    print(f"Extreme Favorites (-500 or worse):")
    print(f"   Won:   {extreme_wins}/{len(winners)} ({extreme_wins/len(winners)*100:.1f}%)")
    print(f"   Lost:  {extreme_losses}/{len(losers)} ({extreme_losses/len(losers)*100:.1f}%)")
    
    if extreme_losses > extreme_wins:
        print(f"\n⚠️  WARNING: Heavy favorites (-500+) are BUSTING more than winning!")
        print(f"   Recommendation: Add filter to avoid bets with odds < -500")
    
    # =================================================================
    # ANALYSIS 3: TOP OFFENDERS (Teams that burn us)
    # =================================================================
    print("\n" + "="*100)
    print("🔍 ANALYSIS 3: TOP OFFENDERS (Teams That Burn Us)")
    print("="*100)
    
    losers['bet_team'] = np.where(
        losers['bet_side'] == 'HOME',
        losers['home_team'],
        losers['away_team']
    )
    
    burn_counts = losers['bet_team'].value_counts().head(10)
    
    print(f"{'TEAM':<10} | {'BUSTS':<8} | {'% OF TOTAL BUSTS':<20}")
    print("-"*50)
    for team, count in burn_counts.items():
        pct = count / len(losers) * 100
        print(f"{team:<10} | {count:<8} | {pct:>6.1f}%")
    
    # =================================================================
    # ANALYSIS 4: EDGE DISTRIBUTION
    # =================================================================
    print("\n" + "="*100)
    print("🔍 ANALYSIS 4: EDGE DISTRIBUTION")
    print("="*100)
    print("Are we losing on small edges or large edges?")
    print()
    
    print(f"{'Category':<20} | {'Avg Edge (Winners)':<20} | {'Avg Edge (Losers)':<20}")
    print("-"*70)
    print(f"{'All Bets':<20} | {winners['bet_edge'].mean():>18.2%} | {losers['bet_edge'].mean():>18.2%}")
    
    # Edge buckets
    for min_edge, max_edge in [(0.035, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 1.0)]:
        win_bucket = winners[(winners['bet_edge'] >= min_edge) & (winners['bet_edge'] < max_edge)]
        loss_bucket = losers[(losers['bet_edge'] >= min_edge) & (losers['bet_edge'] < max_edge)]
        
        if len(win_bucket) + len(loss_bucket) > 0:
            win_rate = len(win_bucket) / (len(win_bucket) + len(loss_bucket)) * 100
            label = f"{min_edge:.1%}-{max_edge:.1%} edge"
            print(f"{label:<20} | {len(win_bucket):>18} | {len(loss_bucket):>18} (Win Rate: {win_rate:.1f}%)")
    
    # =================================================================
    # SUMMARY & RECOMMENDATIONS
    # =================================================================
    print("\n" + "="*100)
    print("🎯 SUMMARY & RECOMMENDED FILTERS")
    print("="*100)
    
    if len(clues_found) > 0:
        print("\n🔴 MAJOR PATTERNS DETECTED:")
        for feat, pct, diff in clues_found:
            print(f"   • {feat}: {pct:+.1f}% difference (Δ = {diff:+.3f})")
            
            # Specific recommendations
            if 'fatigue' in feat.lower() and pct > 10:
                print(f"      → FILTER: Avoid bets with net_fatigue_score > {home_bet_winners[feat].quantile(0.75):.2f}")
            elif 'elo_diff' in feat.lower() and pct < -10:
                print(f"      → FILTER: Require minimum off_elo_diff > {home_bet_winners[feat].quantile(0.25):.2f}")
            elif 'season_progress' in feat.lower() and pct < -10:
                print(f"      → FILTER: Avoid betting before season_progress > {home_bet_winners[feat].quantile(0.25):.2f}")
    else:
        print("\n✅ No major statistical patterns detected")
        print("   Busted favorites appear to be random variance, not systematic bias")
    
    print("\n" + "="*100)
    print("✅ AUDIT COMPLETE")
    print("="*100)

if __name__ == "__main__":
    audit_busted_favorites()
