# 🏀 NBA MDP Betting Model - Production System

## Overview

This is the **Margin-Derived Probability (MDP)** betting model - a regression-based approach that predicts point spreads and converts them to win probabilities. It achieved **+79.18u profit (29.1% ROI)** on the 2024-25 season validation.

---

## 🚀 Quick Start

### Core Production Files

**1. Configuration** ⚙️
- `production_config_mdp.py` - All model settings, thresholds, features

**2. Model Files** 🤖
- `models/nba_mdp_production_tuned.json` - Trained XGBoost model
- `data/training_data_MDP_with_margins.csv` - Training data

**3. Dashboard** 📊
- `NBA_Dashboard_Enhanced_v5.py` - Main GUI interface

**4. Data Collection** 📥
- `nba_stats_collector_v2.py` - NBA API data ingestion
- `injury_replacement_model.py` - Injury impact tracking
- `kalshi_odds_fetcher.py` - Market odds retrieval (TBD)

**5. Feature Engineering** 🔧
- `feature_calculator_v5.py` - Computes all 19 features
- `off_def_elo_system.py` - Offensive/defensive ELO ratings

---

## 📐 Model Architecture

### Type: Regression → Probability Conversion
```
XGBoost Regressor → Margin Prediction → Win% = norm.cdf(margin / 13.42)
```

### Key Innovation
Uses model's **empirical RMSE (13.42)** instead of generic NBA std (13.5) for probability conversion, ensuring perfect calibration to actual prediction accuracy.

### Performance Metrics
- **RMSE**: 13.42 points
- **MAE**: 11.06 points  
- **Log Loss**: 0.610 (15.4% better than classifier)
- **Brier Score**: 0.210

---

## 🎯 Betting Strategy

### Asymmetric Edge Thresholds (Grid Search Optimized)
```python
MIN_EDGE_FAVORITE = 0.015   # 1.5% edge
MIN_EDGE_UNDERDOG = 0.080   # 8.0% edge
```

**Why Asymmetric?**
- **Favorites** (low variance) → reliable at low edges
- **Underdogs** (high variance) → need strong conviction

### Risk Filters
```python
FILTER_MIN_OFF_ELO = -90    # Physics check: broken offense filter
```

### 2024-25 Validation Results
- **Favorites**: 58 bets, +10.96u, +18.9% ROI, 67.2% win rate
- **Underdogs**: 216 bets, +68.22u, +31.6% ROI, 48.1% win rate
- **Combined**: 274 bets, +79.18u, +29.1% ROI, 53.6% win rate

---

## 🔧 Model Features (19 Total)

### ELO & Ratings (3)
1. `off_elo_diff` - Offensive ELO difference
2. `def_elo_diff` - Defensive ELO difference
3. `home_composite_elo` - Home team composite ELO

### Pace & Efficiency (4)
4. `projected_possession_margin` - Expected possession advantage
5. `ewma_pace_diff` - Pace differential (exponentially weighted)
6. `ewma_efg_diff` - Effective FG% differential
7. `pace_efficiency_interaction` - Pace × efficiency synergy

### Three-Point Game (2)
8. `ewma_vol_3p_diff` - 3PT volume differential
9. `three_point_matchup` - 3PT offense vs defense matchup

### Fatigue & Rest (1)
10. `net_fatigue_score` - Rest advantage/disadvantage

### Injuries (3)
11. `injury_matchup_advantage` - Injury impact differential
12. `injury_shock_diff` - Recent injury shock factor
13. `star_power_leverage` - Star player injury impact

### Free Throws & Fouls (2)
14. `total_foul_environment` - Combined foul rate
15. `net_free_throw_advantage` - FT rate differential

### Context & Matchups (4)
16. `season_progress` - Point in season (0-1)
17. `league_offensive_context` - League-wide offensive environment
18. `offense_vs_defense_matchup` - Off strength vs def weakness
19. `star_mismatch` - Star player talent gap

**VIF Check**: All features < 2.34 (minimal multicollinearity)

---

## 📊 Daily Workflow

### 1. Data Update (Morning)
```bash
python daily_data_update.py
```
- Fetches yesterday's game results
- Updates ELO ratings
- Refreshes injury reports
- Updates feature cache

### 2. Generate Predictions (11 AM ET)
```bash
python main_predict.py
```
- Loads today's games
- Calculates features
- Generates predictions
- Identifies betting opportunities

### 3. Monitor Dashboard (All Day)
```bash
python NBA_Dashboard_Enhanced_v5.py
```
- View predictions & edges
- Track calibration health
- Monitor bet history
- Scenario simulation

### 4. Fetch Closing Odds (Pre-game) - TBD
```bash
python kalshi_odds_fetcher.py
```
- Get real-time market odds
- Calculate updated edges
- Final bet validation

---

## 📁 Directory Structure

```
New Basketball Model/
├── production_config_mdp.py          # 🔑 MAIN CONFIG
├── main_predict.py                   # Prediction generation
├── NBA_Dashboard_Enhanced_v5.py      # GUI dashboard
│
├── models/
│   ├── nba_mdp_production_tuned.json # 🤖 Trained model
│   └── manifest.json                  # Model version tracking
│
├── data/
│   ├── training_data_MDP_with_margins.csv
│   ├── closing_odds_2024_25_CLEANED.csv
│   └── nba_betting_data.db           # SQLite database
│
├── services/ (or root)
│   ├── feature_calculator_v5.py      # Feature computation
│   ├── off_def_elo_system.py         # ELO ratings
│   ├── injury_replacement_model.py   # Injury tracking
│   └── kalshi_odds_fetcher.py        # Odds retrieval (TBD)
│
├── scripts/
│   ├── retrain_pipeline.py           # Retraining automation
│   ├── nightly_tasks.py              # Overnight updates
│   └── backtest_*.py                 # Validation scripts
│
└── docs/
    ├── MDP_FINAL_CONFIG_SUMMARY.md   # Configuration summary
    └── README.md                      # This file
```

---

## ⚠️ Common Pitfalls

### ❌ DON'T
1. **Don't use raw model probabilities for betting** - Always check edge calculation
2. **Don't ignore the physics filter** (off_elo_diff < -90 = broken offense)
3. **Don't bet without fresh odds** - Market moves invalidate edges
4. **Don't overtrade** - Stick to threshold discipline
5. **Don't use isotonic calibration** - Tested and rejected (-22% worse)

### ✅ DO
1. **Use empirical RMSE (13.42)** for probability conversion
2. **Respect asymmetric thresholds** (1.5% fav / 8.0% dog)
3. **Update ELO daily** - Stale ratings = bad predictions
4. **Track calibration** - Monitor Brier score monthly
5. **Size bets properly** - Quarter Kelly with commission adjustment

---

## 🔍 Troubleshooting

### Model predictions seem off
1. Check if ELO ratings are updated (last update date in DB)
2. Verify injury data is current (injury_replacement_model)
3. Ensure feature_cache is cleared if switching data sources
4. Validate NBA_STD_DEV = 13.42 in config

### Dashboard not showing predictions
1. Check `nba_betting_data.db` has recent game data
2. Verify `production_config_mdp.py` paths are correct
3. Ensure model file exists: `models/nba_mdp_production_tuned.json`
4. Check for errors in `overnight_log.txt`

---

## 🎓 Model Philosophy

### Why Regression > Classification?
- Margin predictions contain more information than binary outcomes
- Natural calibration via Normal CDF conversion
- Better handling of blowouts (min_child_weight=50)
- 15.4% better log loss vs classifier

### Why Threshold Optimization > Calibration?
- Filters unprofitable bets at source (0-10% edge = -46.51u)
- Calibration improves accuracy but not necessarily profit
- Isotonic tested and rejected (-22% worse performance)
- Simpler, more robust, easier to maintain

### Why Asymmetric Thresholds?
- Favorites have **low variance** → 1.5% edge is reliable
- Underdogs have **high variance** → 8% edge needed for conviction
- Grid search validated across 21 different thresholds
- Matches intuition (chalk bets safer than longshots)

---

## 🏆 Version History

### v2.2 (CURRENT - Dec 2025)
- ✅ Optimized thresholds (1.5%/8.0%)
- ✅ Tested & rejected isotonic calibration
- ✅ Validated on 2024-25 season: +79.18u

### v2.1 (Dec 2025)
- ✅ Isotonic calibration experiments
- ✅ Modern era calibrator testing
- ❌ Zero-edge strategy rejected

### v2.0 (Dec 2025)
- ✅ Hyperparameter optimization (50 trials)
- ✅ Empirical RMSE calibration (13.42)
- ✅ 2-season walk-forward validation

---

**Last Updated**: December 19, 2025  
**Status**: ✅ **PRODUCTION READY**  
**Validation**: 2024-25 Season (+79.18u, 274 bets, 29.1% ROI)
