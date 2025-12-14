# Dashboard Visual Reference

## 📊 Tab 1: Calibration Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│ Controls                                                         │
│ Model: [Moneyline ▼]  Days: [30]  [Refresh]                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┬──────────────────────────────┐
│ Reliability Curve            │ Brier Score Trend            │
│                              │                              │
│  1.0┌─────────────────┐     │  0.30┌──────────────────┐    │
│     │        ●        │      │      │      ●     ●     │     │
│  0.8│      ●   ●      │     │  0.25│─ ─ ─ ● ─ ─ ● ─ ─│     │
│     │    ●       ●    │      │      │    ●   ●   ●     │     │
│  0.6│  ●           ●  │     │  0.20│  ●               │     │
│     │●               ●│      │      │                  │     │
│  0.4│                 │     │  0.15└──────────────────┘     │
│     │ - - Perfect     │      │       Jan  Feb  Mar  Apr      │
│  0.2│ ● Model         │     │                              │
│     └─────────────────┘      │                              │
│       Predicted Prob         │                              │
└──────────────────────────────┴──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Calibration Statistics                                           │
│ Brier Score: 0.1847    ECE: 0.0234    Samples: 247              │
│ Last Fit: 2024-01-15 14:23                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Reliability curve shows if 70% predictions win ~70% of the time
- Brier score tracks accuracy over time (lower = better)
- Statistics panel summarizes calibration quality


## 📈 Tab 2: Model Metrics

```
┌─────────────────────────────────────────────────────────────────┐
│ Controls                                                         │
│ Model: [Moneyline ▼]  Days: [30]  [Refresh]                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┬──────────────┬──────────────┐
│  Moneyline   │     ATS      │    Total     │
│              │              │              │
│      Win Loss│      Win Loss│      Win Loss│
│ Win  [92][18]│ Win  [78][22]│ Win  [85][15]│
│ Loss [25][85]│ Loss [30][70]│ Loss [20][80]│
│              │              │              │
└──────────────┴──────────────┴──────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ROC Curves                                                       │
│                                                                  │
│  1.0┌────────────────────────────────────┐                     │
│     │           ╱                         │                     │
│  0.8│         ╱  ML (AUC=0.732)          │                     │
│     │       ╱    ATS (AUC=0.698)         │                     │
│  0.6│     ╱      Total (AUC=0.715)       │                     │
│     │   ╱        Random (AUC=0.500)      │                     │
│  0.4│ ╱                                   │                     │
│     └────────────────────────────────────┘                     │
│      False Positive Rate                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Feature Importance (Top 15)                                      │
│                                                                  │
│ elo_diff_rolling_3      ████████████████████████████ 2847       │
│ rest_advantage          ████████████████████ 1923                │
│ home_win_pct_L10        ██████████████ 1456                      │
│ away_off_rating_L5      ████████████ 1234                        │
│ composite_momentum      ██████████ 987                           │
│ home_pace_L3            ████████ 876                             │
│ injury_impact_home      ██████ 654                               │
│ ...                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Performance Statistics                                           │
│ Moneyline: Accuracy: 0.787  AUC: 0.732                          │
│ ATS:       Accuracy: 0.740  AUC: 0.698                          │
│ Total:     Accuracy: 0.760  AUC: 0.715                          │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Confusion matrices show TP/FP/TN/FN for each model
- ROC curves compare discriminative power (AUC scores)
- Feature importance reveals what drives predictions


## ⚠️ Tab 3: Risk Management

```
┌─────────────────────────────────────────────────────────────────┐
│ Controls                                                         │
│ Initial Bankroll: [$10,000]  Days: [90]  Kelly: [25%]  [Refresh]│
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────────────┬──────────────────────────┐
│ Bankroll Equity Curve          │ Drawdown                 │
│                                │                          │
│ $12K┌──────────────────┐      │   0%┌──────────────────┐ │
│     │      ╱───╲        │       │     │                  │ │
│ $11K│    ╱─     ╲──╲   │       │  -5%│    ▓▓            │ │
│     │  ╱─           ╲──│       │     │   ▓▓▓▓  ▓▓       │ │
│ $10K│─────────────────  │       │ -10%│  ▓▓▓▓▓▓▓▓▓      │ │
│     │▓▓Profit           │       │     │ ▓▓▓▓▓▓▓▓▓▓▓     │ │
│ $9K │▓Loss              │       │ -15%│▓▓▓▓▓▓▓▓▓▓▓▓    │ │
│     └──────────────────┘       │     └──────────────────┘ │
│      Oct Nov Dec Jan           │      Oct Nov Dec Jan     │
└────────────────────────────────┴──────────────────────────┘

           ┌──────────────────┐
           │  Risk of Ruin    │
           │                  │
           │    ╭──────╮      │
           │  ╱          ╲    │
           │ │   0.87%    │   │
           │  ╲          ╱    │
           │    ╰──────╯      │
           │   [GREEN]        │
           └──────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Risk Statistics                                                  │
│ Current Bankroll: $11,245    Total P&L: +$1,245    ROI: +12.45% │
│ Max Drawdown: -8.3%  DD Duration: 12 days  Sharpe: 1.87         │
│ Win Rate: 56.2%  Avg Win: $145.23  Avg Loss: $98.45             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Equity curve shows bankroll growth over time
- Drawdown chart identifies worst losing streaks
- Risk-of-ruin gauge warns of bankroll danger (< 1% = safe)
- Statistics quantify risk-adjusted performance


## 🎛️ Interactive Filters (Top of Predictions Tab)

```
┌─────────────────────────────────────────────────────────────────┐
│ Filters & Actions                                                │
│                                                                  │
│ Min Edge: ├────●────────────────┤ 3.5%                          │
│                                                                  │
│ ☑ Hide Negative Edge                                            │
│ ☐ Hide Low Probability (<40%)                                   │
│                                                                  │
│ Sort by: [Kelly Stake] [Expected Value] [Edge]                  │
│                                                                  │
│ [Export to CSV]                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Predictions Table (Filtered)                                     │
│ Team       │ Prob  │ Odds │ Edge │ Kelly │ EV    │              │
│ LAL @ BOS  │ 62%   │ 2.10 │ 5.2% │ $287  │ +$42  │              │
│ MIA @ PHX  │ 58%   │ 1.95 │ 3.8% │ $215  │ +$28  │              │
│ ...        │       │      │      │       │       │              │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Slider filters predictions by minimum edge threshold
- Checkboxes hide negative EV or low-probability bets
- Sort buttons reorder by Kelly stake, EV, or edge
- Export creates CSV with all filtered predictions


## 🎨 Theme Toggle

```
Dark Theme (Default):                Light Theme:
┌─────────────────────┐             ┌─────────────────────┐
│ Background: #1e1e1e │             │ Background: #ffffff │
│ Surface:    #2d2d2d │             │ Surface:    #f8f9fa │
│ Text:       #ffffff │             │ Text:       #212529 │
│ Primary:    #0d6efd │             │ Primary:    #0d6efd │
└─────────────────────┘             └─────────────────────┘

[🌙 Dark Theme]                     [☀️ Light Theme]
```

**Key Features:**
- One-click toggle between dark and light themes
- Consistent color palette across all widgets
- Professional styling for tables, buttons, inputs
- Accessible contrast ratios


## ♿ Tooltips (Hover any label)

```
          ┌─────────────────────────────────┐
          │ Kelly Stake                     │
          │ Optimal bet size based on       │
          │ Kelly Criterion:                │
          │ f = (p×b - q) / b               │
          │ where:                          │
          │ • p = win probability           │
Kelly     │ • q = 1 - p                     │
$287      │ • b = odds (decimal - 1)        │
          │ • f = fraction of bankroll      │
          └─────────────────────────────────┘
          ↑ Hover to see formula
```

**Tooltips Available:**
- Kelly Stake (formula and interpretation)
- Edge (calculation and meaning)
- Expected Value (profit per bet)
- Brier Score (accuracy metric)
- Calibration (reliability explanation)
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown (worst decline)
- Risk of Ruin (probability of bankruptcy)


## 📊 Data Flow Diagram

```
NBA Betting Dashboard v5.1
    │
    ├─── [Predictions Tab]
    │    ├─ Prediction Filter Widget
    │    │  ├─ Edge slider (0-20%)
    │    │  ├─ Checkboxes (hide negative, low prob)
    │    │  ├─ Sort buttons (Kelly/EV/Edge)
    │    │  └─ Export CSV button
    │    └─ Predictions Table (filtered)
    │
    ├─── [📊 Calibration]
    │    ├─ Model selector (ML/ATS/Total)
    │    ├─ Reliability curve chart
    │    ├─ Brier score trend chart
    │    └─ Statistics panel
    │
    ├─── [📈 Model Metrics]
    │    ├─ Confusion matrices (3 models)
    │    ├─ ROC curves (multi-model)
    │    ├─ Feature importance chart
    │    └─ Performance statistics
    │
    └─── [⚠️ Risk Management]
         ├─ Equity curve chart
         ├─ Drawdown chart
         ├─ Risk gauge widget
         └─ Risk statistics panel

Database: nba_betting_data.db
    ├─ predictions (ml_prob, ats_prob, kelly_stake, odds)
    └─ games (home_score, away_score, total_score)

Models: models/
    ├─ model_v5_ml.xgb (Moneyline)
    ├─ model_v5_ats.xgb (ATS)
    └─ model_v5_total.xgb (Total)
```


## 🔧 Integration Workflow

```
1. Install Dependencies
   ├─ matplotlib (charts)
   ├─ scikit-learn (metrics)
   ├─ xgboost (feature importance)
   └─ PyQt6 (GUI framework)
   ✅ ALL VERIFIED INSTALLED

2. Import Modules
   ├─ from dashboard_enhancements import CalibrationTab, ThemeManager, TooltipHelper
   ├─ from dashboard_metrics_tabs import ModelMetricsTab, RiskGauge
   └─ from dashboard_risk_filters import RiskManagementTab, PredictionFilter
   ✅ ALL IMPORTS WORKING

3. Apply Theme
   └─ self.setStyleSheet(ThemeManager.get_stylesheet(is_dark=True))
   ✅ STYLESHEETS GENERATED

4. Add Tabs
   ├─ tabs.addTab(CalibrationTab(db_path), "📊 Calibration")
   ├─ tabs.addTab(ModelMetricsTab(db_path), "📈 Model Metrics")
   └─ tabs.addTab(RiskManagementTab(db_path), "⚠️ Risk Management")
   ⏳ PENDING INTEGRATION

5. Add Filters to Predictions Tab
   └─ prediction_filter = PredictionFilter()
   ⏳ PENDING INTEGRATION

6. Test with Real Data
   └─ python NBA_Dashboard_Enhanced_v5.py
   ⏳ PENDING INTEGRATION
```


## 📝 Quick Reference

### Module Sizes:
- dashboard_enhancements.py: 20,698 bytes (CalibrationTab, ThemeManager)
- dashboard_metrics_tabs.py: 19,848 bytes (ModelMetricsTab, RiskGauge)
- dashboard_risk_filters.py: 21,094 bytes (RiskManagementTab, PredictionFilter)

### Key Classes:
- CalibrationTab(db_path) - Reliability and Brier analysis
- ModelMetricsTab(db_path, models_dir) - Performance metrics
- RiskManagementTab(db_path, initial_bankroll) - Risk tracking
- PredictionFilter() - Interactive filtering
- RiskGauge() - Visual risk indicator
- ThemeManager - Dark/light themes
- TooltipHelper - Formula tooltips

### Database Tables Used:
- predictions: All probability columns, Kelly stakes, odds
- games: Outcomes (scores) for metric calculation

### Refresh Methods:
- calibration_tab.refresh_calibration()
- metrics_tab.refresh_metrics()
- risk_tab.refresh_risk()
- All triggered by "Refresh" buttons or data updates


## ✅ Verification Status

```
Module Imports:           ✅ PASS
ThemeManager:             ✅ PASS
TooltipHelper:            ✅ PASS
RiskGauge:                ✅ PASS
PredictionFilter:         ✅ PASS
Dependencies:             ✅ PASS (matplotlib, sklearn, xgboost, PyQt6)
Documentation:            ✅ PASS (integration guide exists)
File Structure:           ✅ PASS (all files present)

OVERALL STATUS:           ✅ READY FOR INTEGRATION
```

See `DASHBOARD_ENHANCEMENT_INTEGRATION.md` for step-by-step integration instructions.
