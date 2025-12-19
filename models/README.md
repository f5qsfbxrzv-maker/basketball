# NBA Betting Model - Results & Documentation

## Executive Summary

This document records the development, testing, and findings from building an NBA betting prediction model using machine learning and advanced statistical features.

**Date:** December 14, 2025  
**Model Version:** 37-feature XGBoost with calibration  
**Status:** Under development - Do NOT use for live betting

---

## Model Architecture

### Current Configuration
- **Algorithm:** XGBoost with isotonic calibration
- **Features:** 37 (reduced from 44 after feature audit)
- **Training Data:** 10,749 games (2015-2024)
- **Test Period:** 2024-25 & 2025-26 seasons (1,456 games)
- **Cross-Validation:** 5-fold TimeSeriesSplit

### Key Hyperparameters (Trial 306)
```python
learning_rate: 0.001575
max_depth: 3
min_child_weight: 14
subsample: 0.569192
colsample_bytree: 0.877130
gamma: 3.824719
reg_lambda: 0.090566
reg_alpha: 0.002029
```

---

## Feature Evolution

### Phase 1: Initial 43 Features (Dec 12, 2024)
- Missing `away_composite_elo` - only had `home_composite_elo`
- **Problem:** Model could only see one side of matchup strength
- Test AUC: 0.628 (baseline)

### Phase 2: Added away_composite_elo (Dec 13-14, 2024)
- Added missing `away_composite_elo` feature
- **Result:** Feature immediately became #1 importance (10.52% gain)
- **Data Leak Check:** PASSED - ELO uses historical ratings only
- Test AUC: 0.694 (+0.07 improvement)

### Phase 3: Feature Audit & Cleanup (Dec 14, 2024)
Removed redundant/zero-importance features:

**Zero Importance (0.000% gain):**
- `fatigue_mismatch`
- `is_season_opener`
- `endgame_phase`

**Perfect Duplicates (r=1.0):**
- `season_year_normalized` → Kept `season_year`
- `ewma_foul_synergy_away` → Kept `away_ewma_fta_rate`
- `season_progress` → Kept `games_into_season`

**Highly Correlated (r>0.85):**
- Removed `season_month` (redundant with `games_into_season`)

**Final Feature Count:** 37

---

## Top Features by Prediction Power

| Rank | Feature | Gain % | Description |
|------|---------|--------|-------------|
| 1 | away_composite_elo | 10.52% | Away team's offensive/defensive ELO rating |
| 2 | season_year | 4.15% | Era indicator (pace evolution, rule changes) |
| 3 | away_back_to_back | 3.35% | Away team playing 2nd game in 2 days |
| 4 | ewma_efg_diff | 3.27% | Effective field goal % differential |
| 5 | rest_advantage | 2.89% | Days of rest differential |
| 6 | away_rest_days | 2.76% | Away team days since last game |
| 7 | away_3in4 | 2.72% | Away team 3 games in 4 days |
| 8 | injury_shock_diff | 2.69% | Injury impact differential |
| 9 | off_elo_diff | 2.69% | Offensive ELO differential |
| 10 | altitude_game | 2.50% | Game played at high altitude (DEN) |

**Top 10 account for:** 39.7% of total prediction power  
**Top 20 account for:** 62.4% of total prediction power

---

## Data Leakage Audit (CRITICAL)

### Suspicion Trigger
Initial backtest showed +24.86% ROI with 66.9% win rate - **far too good to be true** for sports betting (syndicates achieve 2-4% ROI).

### Leak Detection Process

#### Test 1: ELO Time Travel Check ✅ PASSED
- **Method:** Verified `away_composite_elo` uses ratings BEFORE game, not after
- **Code Check:** `get_latest(before_date=game_date)` ensures no look-ahead
- **Pattern Analysis:** ELO changes lag results (win in Game N → ELO rises in Game N+1)
- **Verdict:** CLEAN - No data leakage in ELO calculation

#### Test 2: Correlation Analysis ✅ PASSED
- `away_composite_elo` vs `target_spread_cover` correlation: 0.12 (weak)
- Expected: Weak correlation (ELO reflects past performance, not current game)
- **Verdict:** CLEAN - ELO does not contain future information

#### Test 3: Odds Assumption Check ❌ FAILED
- **Problem:** Backtest assumed -110 odds for ALL bets
- **Reality:** Favorites have worse odds (-200, -300), underdogs better odds (+150, +200)
- **Impact:** 82% overpayment on favorite wins
- **Example:**
  - Real payout at -200: $47.60 profit on $100
  - Backtest payout: $86.63 profit (using -110)
  - **Difference: +$39 per favorite win = massive artificial inflation**

#### Test 4: Real Odds Availability ⚠️ INCOMPLETE
- **Found:** Moneyline odds for 1,141 games (2024-25 season)
- **Missing:** Historical spread odds (only -110 placeholders)
- **Impact:** Cannot calculate realistic spread betting ROI

### Leak Audit Conclusion
✅ **Model is CLEAN** - No data leakage in features or calculations  
❌ **ROI is FAKE** - Backtest used wrong odds assumptions  
⚠️ **Real Performance: UNKNOWN** - Need actual spread odds to determine true edge

---

## Backtest Results

### Initial Backtest (INVALID - Even Money Bug)
- **Period:** 2024-25 & 2025-26 seasons
- **ROI:** +24.86% ❌ **UNREALISTIC**
- **Win Rate:** 66.9%
- **Problem:** Used -110 odds for all bets (incorrect)
- **Status:** ❌ Results discarded

### Corrected Backtest Attempt (INVALID - Placeholder Odds)
- **ROI:** +17.41% ❌ **STILL UNREALISTIC**
- **Problem:** Database contained -110 for all games (not real market prices)
- **Odds Distribution:** 429 of 429 bets at 1.909 decimal odds (identical)
- **Status:** ❌ Results discarded

### Moneyline Backtest (Real Odds)
**Date:** December 14, 2025  
**Status:** ⏳ In Progress

**Available Data:**
- 1,141 games with real moneyline odds
- Date range: 2024-10-22 to 2025-04-11
- Sources: DraftKings, FanDuel, Caesars, BetMGM, Bovada
- Odds variation: -2500 to +1100 (realistic market prices)

**Methodology:**
- Train on pre-Oct 2024 games
- Test on 2024-25 season with real moneyline odds
- Minimum edge: 3%
- Commission: 4.8% (Kalshi)
- Flat $100 unit sizing

**Results:** See `backtest_moneyline_results.csv`

---

## Known Issues & Limitations

### Critical Blockers
1. **No Historical Spread Odds**
   - Cannot validate spread betting ROI
   - Only have moneyline odds for 2024-25
   - Need to download spread odds from The Odds API

2. **Model-Market Mismatch**
   - Model trained on spread covers
   - Testing on moneyline predictions (approximation)
   - Ideal: Separate models for spread vs moneyline

3. **Limited Test Sample**
   - Only 1,141 games with real odds (1 season)
   - Need multi-season validation
   - Sample size too small for CLV analysis

### Model Limitations
1. **Calibration Uncertainty**
   - Calibrated on historical data
   - May not generalize to live betting environment
   - Brier score improvement: +0.55% (modest)

2. **Feature Staleness**
   - ELO updated through Dec 13, 2025
   - Injury data may lag real-time
   - Rest/fatigue calculated day-of (not predictive for future games)

3. **Market Efficiency**
   - NBA betting markets are highly efficient
   - Closing line value (CLV) is extremely difficult to beat
   - Even +3% ROI would be exceptional

---

## Recommendations

### Before Live Betting
- [ ] Download historical spread odds (2022-2025)
- [ ] Run proper spread backtest with real odds
- [ ] Calculate Closing Line Value (CLV) metrics
- [ ] Train separate moneyline model
- [ ] Implement Kelly criterion with drawdown scaling
- [ ] Paper trade for 500+ games minimum
- [ ] Track calibration drift in live environment

### Data Collection Priorities
1. Historical spread odds from The Odds API
2. Opening line vs closing line differentials
3. Bet tracking database with actual results
4. Market movement data (line shopping)
5. Sharp money indicators

### Model Improvements
1. Separate models for spread, moneyline, totals
2. Ensemble approach (XGBoost + Bayesian + Poisson)
3. Correlated bet analysis (same-game parlays)
4. Injury news impact quantification
5. Lineup uncertainty modeling

---

## File Structure

```
models/
  ├── nba_optuna_37features.db          # Hyperparameter optimization results
  ├── optuna_37features_2000trials_results.json  # Best trial metadata
  ├── feature_importance_44features.csv  # Feature audit results
  ├── all_feature_correlations.csv      # Correlation analysis
  ├── backtest_moneyline_results.csv    # Live odds backtest
  └── README.md                          # This file

data/
  ├── training_data_36features.csv       # Current 37-feature dataset
  ├── training_data_44features.csv       # Original with away_composite_elo
  └── live/
      ├── closing_odds_2024_25.csv       # Real moneyline odds
      └── nba_betting_data.db            # ELO, injuries, predictions

scripts/
  ├── optuna_2000trials_37features.py    # Hyperparameter optimization
  ├── backtest_moneyline.py              # Moneyline backtest with real odds
  ├── audit_payout_bug.py                # Even Money Bug detector
  └── feature_audit_44.py                # Feature importance analysis
```

---

## Optimization History

### Trial 306 (Current Best)
- **Date:** December 14, 2025
- **Trials:** 1,172 completed (8-hour run)
- **Best LogLoss:** 0.6577
- **Best AUC:** 0.6295 (validation)
- **Test AUC:** 0.694 (out-of-sample)
- **Strategy:** Deep learning rate (1e-4 to 0.05) + heavy regularization

### Trial Parameters
```json
{
  "learning_rate": 0.0015749732534824602,
  "max_depth": 3,
  "min_child_weight": 14,
  "lambda": 0.09056638614408571,
  "alpha": 0.002028727741034165,
  "subsample": 0.5691917443013035,
  "colsample_bytree": 0.8771304711104997,
  "gamma": 3.824718716724566
}
```

### Ongoing Optimization
- **Study:** nba_37features_2000trials
- **Target:** 2,000 trials (~10 hours)
- **Status:** Running in background
- **Check progress:** 
  ```python
  import optuna
  study = optuna.load_study(
      study_name='nba_37features_2000trials',
      storage='sqlite:///models/nba_optuna_37features.db'
  )
  print(f"Trials: {len(study.trials)} | Best: {study.best_value:.6f}")
  ```

---

## Theoretical Foundations

### Why This is Hard
1. **Market Efficiency:** Closing lines contain all public information
2. **Vig/Juice:** Books charge 4-10% commission, need to overcome
3. **Sample Size:** 82 games/season per team = limited data
4. **Variance:** High variance sport (upsets common)
5. **Line Shopping:** Can't assume best odds available

### What Success Looks Like
- **Realistic ROI Target:** 2-4% (pre-commission)
- **Win Rate:** 52-54% on -110 lines (breakeven = 52.4%)
- **Closing Line Value:** +0.5 to +1.0 points vs closing line
- **Calibration:** Brier score < 0.20, reliability gap < 5%
- **Consistency:** Positive ROI over 1,000+ bets

### Professional Standards
- Syndicates: 2-4% ROI, $100M+ bankrolls
- Sharp bettors: 1-3% ROI, strict bankroll management
- Casual winners: 0.5-1% ROI (luck vs skill unclear)
- Break-even threshold: Must overcome 4.5% vig + variance

---

## Version History

| Date | Version | Features | Best AUC | Status |
|------|---------|----------|----------|--------|
| Dec 12 | 1.0 | 43 | 0.628 | Missing away_elo |
| Dec 13 | 2.0 | 44 | 0.629 | Added away_elo |
| Dec 14 | 3.0 | 37 | 0.694 | Feature audit, cleaned |

---

## Contact & Support

**Project:** NBA Betting Model  
**Created:** December 2025  
**Last Updated:** December 14, 2025  

⚠️ **DISCLAIMER:** This model is for research and educational purposes only. Sports betting involves substantial risk. Past performance does not guarantee future results. The backtest results presented have known issues with odds assumptions and should not be used as evidence of profitability. Do not bet real money until proper validation with actual market odds is completed.

---

## Next Steps

1. ✅ Complete feature audit
2. ✅ Identify and fix data leakage concerns
3. ⏳ Run moneyline backtest with real odds
4. ⏸️ Download historical spread odds
5. ⏸️ Calculate true spread betting ROI
6. ⏸️ Implement Kelly criterion
7. ⏸️ Paper trade 500+ games
8. ⏸️ Build calibration monitoring system
9. ⏸️ Deploy live prediction pipeline

---

*"The goal is not to predict the winner. The goal is to find mispriced probabilities in the market."* - Unknown Sharp Bettor
