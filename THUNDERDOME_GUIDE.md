# 🥊 ADVANCED THUNDERDOME KIT - User Guide

## What You Just Got

Three professional-grade tools for model evaluation that go **beyond simple profit comparison**.

---

## 🎯 The Problem

**Bad Question:** "Which model made more money?"  
**Good Question:** "Which model is safer, more consistent, and wins when they disagree?"

Model A makes $500 betting on crazy +400 underdogs.  
Model B makes $400 with steady, low-risk bets.  
→ **Model B is actually better for long-term wealth.**

The Advanced Thunderdome Kit finds this truth.

---

## 🛠️ The Tools

### 1. **Divergence Analysis** (`divergence_analysis.py`)
**Purpose:** Find games where models disagree (maximum edge opportunities)

**Why it matters:**
- When both models agree → Easy bet, low edge
- When models disagree → High edge potential
- Model that wins disagreements → Trust that one more

**Run it:**
```powershell
cd Sports_Betting_System
python -m src.backtesting.divergence_analysis
```

**Output:**
- Total disagreements between models
- Head-to-head clash count (opposite bets)
- Who was correct in disagreements
- Saved to: `logs/divergence_analysis.csv`

**Example output:**
```
Total Disagreements: 1,247 games (10.2%)
HEAD-TO-HEAD CLASHES: 89 games
Model A Correct: 52 (58.4%)
Model B Correct: 37 (41.6%)
→ WINNER: Model A wins disagreements
```

---

### 2. **Equity Curve Visualizer** (`visualize_equity.py`)
**Purpose:** Visual proof of stability vs volatility

**Why it matters:**
- Numbers lie, charts don't
- Drawdowns reveal psychological pain
- Steady growth > volatile spikes

**Features:**
- Equity curves (cumulative profit over time)
- Drawdown curves (how deep did it fall?)
- Sharpe ratio (risk-adjusted returns)
- Max drawdown (worst losing streak)

**Usage in code:**
```python
from src.backtesting.visualize_equity import plot_thunderdome_results, print_risk_report

# After running backtest
plot_thunderdome_results(bankroll_a, bankroll_b)
print_risk_report(bankroll_a, "Model A")
```

**Output:**
- Chart saved to: `logs/thunderdome_equity_curve.png`
- Risk metrics printed to console

**Interpretation:**
- Sharpe > 0.1 → Excellent
- Max Drawdown < 10% → Safe
- Smooth curve > spiky curve

---

### 3. **Ultimate Thunderdome** (`ultimate_thunderdome.py`)
**Purpose:** Combined master script that runs EVERYTHING

**Features:**
- ✅ Profit comparison
- ✅ ROI calculation
- ✅ Win rate analysis
- ✅ Sharpe ratio (risk-adjusted)
- ✅ Max drawdown
- ✅ Divergence analysis
- ✅ Equity curve visualization
- ✅ Multi-criteria verdict

**Run it:**
```powershell
cd Sports_Betting_System
python -m src.backtesting.ultimate_thunderdome
```

**What it does:**
1. Loads Model A (production) and Model B (challenger)
2. Simulates full season of betting for both
3. Calculates profit, ROI, win rate
4. Computes risk metrics (Sharpe, max drawdown)
5. Finds disagreements between models
6. Determines winner by 3 criteria:
   - Higher profit
   - Higher ROI
   - Better Sharpe ratio (lower risk)
7. Generates equity curve chart
8. Saves results to CSV

**Output:**
```
🏆 FINAL VERDICT
✓ Model A wins on PROFIT
✓ Model B wins on ROI
✓ Model A wins on SHARPE RATIO (lower risk)

👑 WINNER: MODEL A RETAINS THE CROWN
   Victory Score: 2-1
```

---

### 4. **Backtester Verification** (`verify_backtester_logic.py`)
**Purpose:** Test the TOOL before testing the SUBJECT

**Why it matters:**
- If backtester is broken, all results are wrong
- Must verify ruler is straight before measuring

**What it tests:**
1. **Perfect Model Test:** 100% win rate should show ~90% ROI
2. **Random Model Test:** 50% win rate should lose to vig (~-5% ROI)
3. **Realistic Edge Test:** 55% win rate should show modest profit

**Run it:**
```powershell
cd Sports_Betting_System
python tests/verify_backtester_logic.py
```

**Expected result:**
```
✅ ALL TESTS PASSED
   Your backtester logic is SOUND
   You can now trust model comparisons
```

**If tests fail:**
→ Your backtester has bugs  
→ Fix it before comparing models  
→ Current results are UNTRUSTWORTHY

---

## 📋 Recommended Workflow

### Phase 1: Verify the Tool
```powershell
# Test backtester logic FIRST
python tests/verify_backtester_logic.py
# Must see "ALL TESTS PASSED"
```

### Phase 2: Compare Models
```powershell
# Run ultimate thunderdome
python -m src.backtesting.ultimate_thunderdome
# Review equity curves and verdict
```

### Phase 3: Deep Dive on Disagreements
```powershell
# Analyze divergence
python -m src.backtesting.divergence_analysis
# Check logs/divergence_analysis.csv
```

### Phase 4: Make Decision
**Promote Model B to production if:**
- ✅ Wins 2/3 criteria (profit, ROI, Sharpe)
- ✅ Max drawdown < 15%
- ✅ Wins majority of disagreements
- ✅ Sharpe ratio > 0.1

**Keep Model A if:**
- ❌ Model B failed any of above
- ❌ Equity curve too volatile
- ❌ Not enough edge in disagreements

---

## 🎯 Success Metrics

### What to Look For

**Profit:**
- Model B profit > Model A profit? ✅

**Risk:**
- Model B Sharpe > Model A Sharpe? ✅
- Model B max drawdown < 15%? ✅

**Disagreements:**
- Model B wins >55% of disagreements? ✅

**Consistency:**
- Model B equity curve smooth? ✅
- No crazy spikes/crashes? ✅

### Red Flags 🚨

- ❌ 80%+ win rate → Data leak!
- ❌ >20% max drawdown → Too risky
- ❌ Sharpe < 0.05 → No edge
- ❌ Spiky equity curve → Unstable
- ❌ Loses disagreements → No advantage

---

## 📊 Configuration

Edit these values in `ultimate_thunderdome.py`:

```python
# Model paths
MODEL_A_PATH = "models/production/best_model.joblib"
MODEL_B_PATH = "models/staging/candidate_model.joblib"

# Data
TEST_DATA_PATH = "data/processed/training_data_final.csv"

# Betting params
CONFIDENCE_THRESHOLD = 0.55  # Only bet if prob > 0.55 or < 0.45
BET_SIZE = 100  # Units per bet
```

---

## 🎓 Understanding the Metrics

### ROI (Return on Investment)
```
ROI = (Final Profit / Total Wagered) × 100
```
- 5-15% = Excellent for sports betting
- >20% = Likely a leak (too good to be true)

### Sharpe Ratio
```
Sharpe = Average Return / Standard Deviation of Returns
```
- >0.1 = Excellent per-bet
- 0.05-0.1 = Good
- <0.05 = Poor edge

### Max Drawdown
```
Max DD = Largest peak-to-trough decline
```
- <10% = Very safe
- 10-15% = Acceptable
- >20% = High risk

### Win Rate
```
Win Rate = Wins / Total Bets
```
- 52-58% = Realistic with edge
- >60% = Suspicious (possible leak)
- <50% = Losing to vig

---

## 🔧 Troubleshooting

**Problem:** ModuleNotFoundError  
**Solution:**
```powershell
cd Sports_Betting_System
$env:PYTHONPATH = (Get-Location).Path
```

**Problem:** Model files not found  
**Solution:**
```powershell
# Check paths exist
Test-Path "models/production/best_model.joblib"
Test-Path "models/staging/candidate_model.joblib"
```

**Problem:** Feature mismatch error  
**Solution:** Models trained on different features. Retrain with same dataset.

**Problem:** Charts not showing  
**Solution:** Install matplotlib:
```powershell
pip install matplotlib
```

---

## 📁 File Locations

```
Sports_Betting_System/
├── src/backtesting/
│   ├── divergence_analysis.py     # Finds model disagreements
│   ├── visualize_equity.py        # Equity curves & risk metrics
│   └── ultimate_thunderdome.py    # Master comparison script
├── tests/
│   └── verify_backtester_logic.py # Backtester verification
└── logs/
    ├── divergence_analysis.csv    # Disagreement data
    ├── thunderdome_equity_curve.png   # Chart
    └── thunderdome_results.csv    # Summary table
```

---

## 🚀 You Now Have

- ✅ Divergence Engine (find maximum edge)
- ✅ Equity Visualizer (see stability)
- ✅ Risk Calculator (Sharpe, drawdown)
- ✅ Ultimate Thunderdome (all-in-one)
- ✅ Backtester Verification (trust your tools)

**This is professional-grade model evaluation.**

No more guessing. No more "it feels better."  
Numbers, charts, and statistical rigor decide.

---

**Good luck. May the best model win.** 🥊
