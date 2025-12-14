# QUICK COMMAND REFERENCE - Feature & Model Analysis

## 🎯 Dashboard Access
```powershell
# Launch main dashboard
.\.venv\Scripts\python.exe main.py

# Navigate to: "🔬 Feature & Model Analysis" tab
```

## ⚡ Quick Tests (2-5 minutes each)

### Feature Importance (Fast)
```python
python comprehensive_analysis.py features
```
**Output**: Top 10 features with importance scores

### Model Comparison (Fast)
```python
python comprehensive_analysis.py models
```
**Output**: Top 5 models with ROC-AUC scores

### Full Analysis Pipeline (30 minutes)
```python
python comprehensive_analysis.py full
```
**Output**: Complete reports in `analysis_results/`

## 🔍 What Each Analysis Reveals

| Analysis | Time | Key Output | Action Item |
|----------|------|------------|-------------|
| **Correlation** | 2 min | Redundant features | Remove highly correlated pairs |
| **Validation** | 3 min | Noise features | Drop features with <0.52 predictive power |
| **Importance** | 2-5 min | Top contributors | Focus on top 10-20 features |
| **Minimal Test** | 3 min | Top 4 vs All | Simplify model if minimal = full |
| **Model Comparison** | 10 min | Best algorithm | Switch to best performer |
| **Ensemble Test** | 8 min | Voting/Stacking | Use ensemble if >1% improvement |

## 📊 Typical Workflow

### Phase 1: Feature Discovery (10 minutes)
1. Run **Correlation Analysis** → Find redundant features
2. Run **Feature Validation** → Find noise features
3. Create removal list

### Phase 2: Feature Selection (5 minutes)
1. Run **Feature Importance** (XGBoost) → Rank features
2. Run **Minimal Model Test** → Test top 4-10
3. Determine optimal feature set

### Phase 3: Model Selection (15 minutes)
1. Run **Multi-Model Comparison** → Find best algorithm
2. Run **Ensemble Testing** → Test if combining helps
3. Select final model architecture

### Phase 4: Implementation (5 minutes)
1. Update feature list in code
2. Retrain with optimal model
3. Backtest to validate improvements

## 🎯 Common Questions & Quick Answers

### "Which features should I keep?"
```
Dashboard → Analysis Tab → Feature Importance (XGBoost)
Look for: Importance score >0.05
```

### "Can I use only 4 features instead of 100+?"
```
Dashboard → Analysis Tab → Minimal Model Test
If score difference <0.01: Yes, simplify!
```

### "Is XGBoost the best model?"
```
Dashboard → Analysis Tab → Multi-Model Comparison
Compare: XGBoost vs LightGBM vs Neural Net vs others
```

### "Should I use an ensemble?"
```
Dashboard → Analysis Tab → Ensemble Testing
If improvement >0.01 ROC-AUC: Yes
If <0.01: No, single model sufficient
```

### "Are my features creating noise?"
```
Dashboard → Analysis Tab → Feature Validation
Check: Predictive power column
Drop: Features with <0.52 power
```

### "Which features are redundant?"
```
Dashboard → Analysis Tab → Correlation Analysis
Check: Pairs with correlation >0.8
Keep only one from each pair
```

## 📁 Results Locations

```
analysis_results/
├── correlation_heatmap.png          # Visual correlation matrix
├── feature_importance_xgboost.png   # XGBoost importance plot
├── feature_importance_shap.png      # SHAP importance plot
├── feature_analysis_report.txt      # Complete feature report
├── model_comparison_report.txt      # Model performance report
└── model_comparison_results.csv     # Detailed model scores
```

## 🚀 Optimization Pipeline

```
1. Download Data → System Admin → Download Historical Data
2. Feature Analysis → Analysis Tab → Run Complete Analysis
3. Review Reports → analysis_results/ folder
4. Remove Noise → Drop features with low importance/predictive power
5. Retrain Models → System Admin → Train ML Models
6. Validate → System Admin → Run Backtest
7. Deploy → Use improved model for live betting
```

## ⚡ Time Estimates

| Task | Time Required |
|------|---------------|
| Correlation Analysis | 2 minutes |
| Feature Validation | 3 minutes |
| Feature Importance (XGBoost) | 2 minutes |
| Feature Importance (SHAP) | 10 minutes |
| Minimal Model Test | 3 minutes |
| Model Comparison (All) | 10 minutes |
| Ensemble Testing | 8 minutes |
| **Full Analysis Pipeline** | **20-30 minutes** |

## 🎓 Pro Tips

1. **Start with XGBoost importance** (fastest, good enough)
2. **Run SHAP only for final analysis** (slowest but most accurate)
3. **Test minimal model early** (may save hours of computation)
4. **Remove features iteratively** (analyze → remove → reanalyze)
5. **Compare ensembles last** (need baseline scores first)
6. **Save results before retraining** (backup for comparison)

## 🔧 Troubleshooting Quick Fixes

| Error | Fix |
|-------|-----|
| "Training data not found" | Run Download Historical Data first |
| "Module not found" | Install: `pip install shap` |
| "Out of memory" | Reduce dataset or use XGBoost instead of SHAP |
| "Analysis taking too long" | Use quick tests, skip SHAP method |
| "Can't open results folder" | Check `analysis_results/` exists |

## 📞 One-Liners

```python
# Get top 10 features (Python console)
from comprehensive_analysis import quick_feature_test
quick_feature_test(top_n=10)

# Compare top 5 models (Python console)
from comprehensive_analysis import quick_model_test
quick_model_test()

# Full analysis from Python
from comprehensive_analysis import run_complete_analysis
run_complete_analysis(target_column='home_wins')
```

## 🎯 Decision Tree

```
Need to optimize model?
├─ Yes → Start here
│   ├─ Improve accuracy?
│   │   ├─ Run Model Comparison → Switch to best model
│   │   └─ Run Ensemble Testing → Use if improvement >1%
│   │
│   ├─ Reduce overfitting?
│   │   ├─ Run Correlation → Remove redundant features
│   │   └─ Run Validation → Remove noise features
│   │
│   └─ Speed up predictions?
│       └─ Run Minimal Model Test → Use top 4-10 features
│
└─ No → Skip analysis, use current model
```

---

**Version**: v1.0  
**Last Updated**: November 2025
