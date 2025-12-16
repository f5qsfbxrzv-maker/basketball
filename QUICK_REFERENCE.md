# TRIAL 1306 - QUICK REFERENCE

## 🎯 Model at a Glance

**File**: `models/xgboost_22features_trial1306_20251215_212306.json`  
**Performance**: 49.7% ROI | 59.1% Win Rate | 0.6222 Log Loss  
**Strategy**: 2% Fav Edge / 10% Dog Edge

---

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| **Backtest ROI (2023-24)** | 30.02% |
| **Backtest ROI (2024-25)** | 9.60% |
| **Combined ROI** | 16.45% |
| **Optimal Strategy ROI** | 49.7% |
| **Total Bets (Combined)** | 1,613 |
| **Win Rate (Combined)** | 71.5% |
| **Features** | 22 |
| **Training Games** | 12,205 |

---

## 🔧 Required Files

### Minimum Files to Run Model
```
models/
├── xgboost_22features_trial1306_20251215_212306.json
├── trial1306_params_20251215_212306.json
└── model_config.json

data/
└── training_data_matchup_with_injury_advantage_FIXED.csv
```

### For Full Analysis
```
+ data/closing_odds_2023_24.csv
+ find_optimal_thresholds.py
+ backtest_2023_24.py
+ backtest_walk_forward.py
+ audit_odds_quality.py
```

---

## 🚀 Usage (Python)

```python
import xgboost as xgb
import json

# Load model
model = xgb.XGBClassifier()
model.load_model('models/xgboost_22features_trial1306_20251215_212306.json')

# Load config
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Get required features
features = config['required_features']  # 22 features

# Predict
home_win_prob = model.predict_proba(X[features])[:, 1]

# Calculate edge
home_edge = home_win_prob - (1 / home_odds_decimal)
away_edge = (1 - home_win_prob) - (1 / away_odds_decimal)

# Apply thresholds
FAV_EDGE = 0.02  # 2%
DOG_EDGE = 0.10  # 10%

if home_odds_decimal < 2.0 and home_edge > FAV_EDGE:
    print("BET HOME (FAVORITE)")
elif home_odds_decimal >= 2.0 and home_edge > DOG_EDGE:
    print("BET HOME (UNDERDOG)")
```

---

## 📋 22 Features (Ranked)

### Top 5
1. **off_elo_diff** (61.3) - Offensive ELO differential
2. **away_composite_elo** (28.7) - Away composite rating
3. **home_composite_elo** (27.6) - Home composite rating ⭐
4. **ewma_efg_diff** (9.4) - Effective FG% differential
5. **net_fatigue_score** (9.1) - Rest/travel impact

### All 22 Features
```python
[
    'home_composite_elo', 'away_composite_elo', 'off_elo_diff', 'def_elo_diff',
    'net_fatigue_score', 'ewma_efg_diff', 'ewma_pace_diff', 'ewma_tov_diff',
    'ewma_orb_diff', 'ewma_vol_3p_diff', 'injury_matchup_advantage',
    'ewma_chaos_home', 'ewma_foul_synergy_home', 'total_foul_environment',
    'league_offensive_context', 'season_progress', 'pace_efficiency_interaction',
    'projected_possession_margin', 'three_point_matchup', 'net_free_throw_advantage',
    'star_power_leverage', 'offense_vs_defense_matchup'
]
```

---

## ⚙️ Hyperparameters

```python
{
    "max_depth": 3,
    "min_child_weight": 25,
    "gamma": 5.1624,
    "learning_rate": 0.0105,
    "n_estimators": 9947,
    "subsample": 0.6278,
    "colsample_bytree": 0.6015,
    "reg_alpha": 6.194
}
```

**Strategy**: Conservative (shallow trees, heavy pruning, slow learning)

---

## 💰 Betting Strategy

### Thresholds
- **Favorites** (odds < 2.0): Edge > 2%
- **Underdogs** (odds ≥ 2.0): Edge > 10%

### Kelly Sizing
```python
b = odds_decimal - 1
q = 1 - win_prob
f_star = (b * win_prob - q) / b
stake = bankroll * f_star * 0.25  # Quarter Kelly
```

### Risk Management
- Max bet: 5% of bankroll
- Quarter Kelly (25% of full Kelly)
- No martingale or chase strategies

---

## 🔍 Critical Fixes Applied

### 1. ELO Corruption (Phase 4)
**Problem**: home_composite_elo had std dev 99.96 (should be ~77)  
**Solution**: Used away_composite_elo as source of truth  
**Result**: home_composite_elo jumped from rank #24 → #3

### 2. Injury Feature Redundancy (Phase 1-2)
**Problem**: 8 injury features with duplicate information  
**Solution**: Consolidated to 1 optimized `injury_matchup_advantage`  
**Result**: Cleaner model, better generalization

### 3. Feature Bloat (Phase 5)
**Problem**: 25 features with 3 redundant components  
**Solution**: Removed shock_diff, impact_diff, star_mismatch  
**Result**: 22-feature model with 5.5% improvement

---

## 📈 Performance by Threshold

| Fav Edge | Dog Edge | Bets | ROI | Win Rate |
|----------|----------|------|-----|----------|
| 0.5% | 10% | 290 | 49.5% | 58.6% |
| 1.0% | 10% | 289 | 49.6% | 58.8% |
| **2.0%** | **10%** | **286** | **49.7%** | **59.1%** |
| 3.0% | 10% | 283 | 49.9% | 59.4% |
| 2.0% | 20% | 229 | 54.3% | 65.5% |

**Selected**: 2% / 10% for balance of volume and efficiency

---

## 🛠️ Maintenance

### Retrain Triggers
- Every 3 months
- After 500 new games
- If win rate drops below 60%
- If ROI drops below 10%

### Data Quality Checks
- ELO std dev should be ~75-80
- Feature values should have no NaN/Inf
- Odds should be within ±2000
- Implied probability ~104-105%

---

## ⚠️ Known Limitations

1. **Small Sample 2024-25**: Only 1,072 bets (partial season)
2. **Odds Coverage**: Not all games have closing odds (44% coverage 2023-24)
3. **Injury Data**: Relies on injury reports (timing lag possible)
4. **Home Court**: Home advantage baked into ELO (not explicit feature)

---

## 📞 Troubleshooting

### Low Win Rate
- Check if using correct thresholds (2% / 10%)
- Verify all 22 features present
- Ensure odds are moneyline (not spread)

### Model Load Error
- Check XGBoost version (>=2.0.0)
- Verify file path is correct
- Ensure JSON not corrupted

### Feature Calculation Errors
- Verify ELO std dev ~75-80 (not corrupted)
- Check for NaN values in EWMA features
- Ensure injury data is up to date

---

## 📚 Reference Files

- **Full README**: `README_TRIAL1306.md`
- **Configuration**: `model_config.json`
- **Commit Script**: `commit_trial1306.ps1`
- **This Guide**: `QUICK_REFERENCE.md`

---

**Last Updated**: December 15, 2024  
**Model Version**: 1.0.0  
**Status**: Production Ready ✅
