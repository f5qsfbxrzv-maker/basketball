# NBA BETTING SYSTEM - FEATURE & MODEL ANALYSIS GUIDE

## 🎯 Overview
Complete toolkit for analyzing feature correlations, validating feature quality, testing model performance, and optimizing your betting strategy.

## 🚀 Quick Start

### Launch Dashboard
```powershell
.\.venv\Scripts\python.exe main.py
```

Navigate to **"🔬 Feature & Model Analysis"** tab

## 📊 Analysis Tools

### 1. Feature Correlation Analysis
**Purpose**: Identify redundant features and multicollinearity

**What it does**:
- Calculates correlation matrix for all features
- Identifies highly correlated feature pairs (>threshold)
- Flags redundant features (>0.95 correlation)
- Generates correlation heatmap visualization

**Usage**:
1. Set correlation threshold (default: 0.8)
2. Click "🔍 Run Correlation Analysis"
3. Review results in console
4. View heatmap: `analysis_results/correlation_heatmap.png`

**Interpretation**:
- High correlation (>0.8) = features provide similar information
- Redundant features (>0.95) = consider removing one
- Reduces overfitting and improves model performance

---

### 2. Feature Validation
**Purpose**: Verify features are generating valid, accurate, and useful data

**Tests Performed**:
1. **Data Generation**: Check for missing values and null rates
2. **Accuracy**: Detect constant features, outliers, suspicious patterns
3. **Contribution**: Measure predictive power vs target variable

**Usage**:
1. Click "🧪 Validate All Features"
2. Review feature status:
   - ✅ Good = Passing all checks
   - ⚠️ Warning = Potential issues
   - ❌ Failed = Not usable

**Key Metrics**:
- Null percentage: Should be <10%
- Predictive power: >0.52 ROC-AUC is contributing
- <0.52 ROC-AUC = Noise (consider removing)

---

### 3. Feature Importance Ranking
**Purpose**: Identify which features matter most for predictions

**Methods Available**:
- **XGBoost**: Tree-based importance (default, fast)
- **Random Forest**: Ensemble tree importance
- **Permutation**: Model-agnostic, slower but robust
- **SHAP**: Explainable AI, most accurate but slowest

**Usage**:
1. Select importance method
2. Click "📊 Calculate Feature Importance"
3. Review top 10-20 features
4. View plot: `analysis_results/feature_importance_*.png`

**Best Practice**:
- Use XGBoost for quick insights
- Use SHAP for final analysis and interpretability
- Compare multiple methods to validate findings

---

### 4. Minimal Model Test
**Purpose**: Determine if simplified model (top N features) performs as well as full model

**What it does**:
- Trains model with only top 4 features (or custom N)
- Compares performance vs full feature set
- Recommends whether to use minimal or full model

**Usage**:
1. Set "Top N Features" (default: 4)
2. Click "🎯 Run Minimal Model Test"
3. Review scores:
   - If minimal ≈ full: Use minimal (faster, less overfitting)
   - If full >> minimal: Use full model

**Example Output**:
```
Full model (127 features): 0.5840 ± 0.0120
Minimal model (4 features): 0.5810 ± 0.0115
💡 Use minimal model
```

---

### 5. Multi-Model Comparison
**Purpose**: Test all ML algorithms to find best performer

**Models Tested**:
- XGBoost
- LightGBM
- Random Forest
- Gradient Boosting
- Logistic Regression
- Neural Network (MLP)
- SVM (RBF kernel)
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Bagging Ensemble

**Usage**:
1. Click "🔬 Compare All Models"
2. Wait 5-10 minutes (tests 11 models with cross-validation)
3. Review top 5 performers
4. Check results CSV: `analysis_results/model_comparison.csv`

**Metrics Reported**:
- ROC-AUC (primary metric for classification)
- Accuracy
- Precision, Recall, F1
- Training time

---

### 6. Ensemble Strategy Testing
**Purpose**: Test if combining models improves performance

**Ensemble Types**:
- **Voting**: Average predictions from top 3-5 models
- **Stacking**: Use meta-learner to combine base models

**Usage**:
1. Click "🎭 Test All Ensembles"
2. Compare ensemble scores vs best single model
3. If improvement >0.01: Use ensemble
4. If improvement <0.01: Single model sufficient

**Example Output**:
```
Best Single Model: XGBoost (0.5840)
Voting Ensemble: 0.5865 (+0.0025)
Stacking Ensemble: 0.5892 (+0.0052)
💡 Stacking improves by 0.0052 - RECOMMENDED
```

---

### 7. Complete Analysis Pipeline
**Purpose**: Run ALL analyses above in one comprehensive sweep

**What it runs**:
1. Correlation analysis
2. Feature validation (first 50 features)
3. Feature importance (all 4 methods)
4. Minimal model test
5. Multi-model comparison
6. Ensemble testing

**Usage**:
1. Select target variable:
   - `home_wins` (Moneyline)
   - `covers_spread` (ATS)
   - `goes_over` (Totals)
2. Click "🚀 Run Complete Analysis"
3. **Wait 15-30 minutes**
4. Review comprehensive reports in `analysis_results/`

**Output Files**:
- `feature_analysis_report.txt` - Complete feature analysis
- `model_comparison_report.txt` - Model performance summary
- `model_comparison_results.csv` - Detailed model scores
- `correlation_heatmap.png` - Feature correlation visualization
- `feature_importance_*.png` - Importance plots for each method

---

## 💡 Recommended Workflow

### For Quick Insights (5 minutes)
1. Run **Feature Importance** (XGBoost method)
2. Run **Minimal Model Test** (4 features)
3. Review if top 4 features are sufficient

### For Model Optimization (15 minutes)
1. Run **Multi-Model Comparison**
2. Run **Ensemble Testing**
3. Identify best single model or ensemble

### For Feature Engineering (30 minutes)
1. Run **Correlation Analysis**
2. Run **Feature Validation**
3. Remove redundant and noise features
4. Retrain models with cleaned features

### For Complete System Audit (30 minutes)
1. Run **Complete Analysis Pipeline**
2. Review all generated reports
3. Make data-driven decisions on:
   - Which features to keep/remove
   - Which model architecture to use
   - Whether to use ensemble methods

---

## 🎯 Key Questions This Answers

### "Are my features actually useful?"
→ Run **Feature Validation** and **Feature Importance**
- Look for features with predictive power >0.52
- Remove features marked as "noise"

### "Do I have redundant features?"
→ Run **Correlation Analysis**
- High correlation pairs (>0.8) provide similar info
- Keep one from each highly correlated pair

### "Can I simplify my model?"
→ Run **Minimal Model Test**
- If top 4-10 features perform as well as all features: simplify
- Reduces overfitting, speeds up training/inference

### "Is XGBoost really the best model for my data?"
→ Run **Multi-Model Comparison**
- Tests 11 different algorithms
- May discover LightGBM, Neural Net, or ensemble performs better

### "Should I use an ensemble?"
→ Run **Ensemble Testing**
- If improvement >1%: Yes, use ensemble
- If improvement <1%: No, single model is sufficient

### "What are my top 4 factors, and do I need the rest?"
→ Run **Feature Importance** then **Minimal Model Test**
- Identifies top contributors (e.g., ELO, Rest Days, Recent Form, Home Court)
- Tests if adding more features helps or just adds noise

---

## 📁 Results Location

All analysis results saved to: `analysis_results/`

**Visualizations**:
- `correlation_heatmap.png`
- `feature_importance_xgboost.png`
- `feature_importance_randomforest.png`
- `feature_importance_permutation.png`
- `feature_importance_shap.png`

**Reports**:
- `feature_analysis_report.txt`
- `model_comparison_report.txt`
- `model_comparison_results.csv`

**Access via Dashboard**:
Click "📁 Open Results Folder" button in analysis tab

---

## 🔧 Command Line Usage

Can also run analyses via command line:

```powershell
# Quick feature importance test
.\.venv\Scripts\python.exe comprehensive_analysis.py features

# Quick model comparison
.\.venv\Scripts\python.exe comprehensive_analysis.py models

# Full analysis pipeline
.\.venv\Scripts\python.exe comprehensive_analysis.py full
```

---

## ⚠️ Troubleshooting

### "Training data not found"
→ Run "Download Historical Data" in System Admin tab first

### Analysis takes too long
→ Use quick tests first:
- Feature Importance (XGBoost) ~2 min
- Minimal Model Test ~3 min
- Skip SHAP method (slowest)

### Out of memory errors
→ Reduce dataset size or run analyses on subset of data

### SHAP method fails
→ Use XGBoost, Random Forest, or Permutation instead
- SHAP requires more memory and computation

---

## 📚 Interpretation Guide

### Feature Importance Scores
- **>0.10**: Critical feature, must keep
- **0.05-0.10**: Important feature
- **0.01-0.05**: Moderate contribution
- **<0.01**: Minimal impact, consider removing

### ROC-AUC Scores (Classification)
- **>0.60**: Excellent
- **0.55-0.60**: Good
- **0.52-0.55**: Acceptable
- **<0.52**: Poor (no better than random)

### Correlation Thresholds
- **>0.95**: Highly redundant, remove one
- **0.80-0.95**: Highly correlated, consider removing
- **0.60-0.80**: Moderately correlated, generally OK
- **<0.60**: Low correlation, keep both

---

## 🎓 Best Practices

1. **Start Simple**: Run quick tests before full analysis
2. **Iterate**: Remove noise features, rerun analysis
3. **Cross-Validate**: Use 5-fold CV to avoid overfitting
4. **Document**: Save reports and track improvements
5. **Compare**: Test multiple targets (moneyline, ATS, totals)
6. **Optimize**: Use findings to retrain models with best features
7. **Monitor**: Re-run analysis quarterly as data evolves

---

## 🚀 Next Steps After Analysis

1. **Identify Top Features**: Use importance ranking
2. **Remove Noise**: Drop features with <0.52 predictive power
3. **Retrain Models**: Use cleaned feature set
4. **Update Config**: Save best hyperparameters
5. **Backtest**: Validate improvements on historical data
6. **Deploy**: Use optimized model for live betting

---

## 📞 Support

For questions or issues with feature analysis:
1. Check console output for error messages
2. Verify training data exists in `data/`
3. Review analysis results in `analysis_results/`
4. Check model reports in System Admin tab

---

**Last Updated**: November 2025
**Version**: v1.0 - Feature & Model Analysis Suite
