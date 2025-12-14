"""
Train model on CLEANED data and compare to original dirty data
This will show the impact of fixing:
- Rest days bug (286 days → 3 for season openers)
- ELO inflation (10.7% drift → 0%)
- ELO diff outliers (±500 → ±400)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import json
import time

print("="*70)
print("CLEAN DATA vs DIRTY DATA COMPARISON")
print("="*70)

# Compare lean dataset (cleaned vs original)
datasets = {
    'DIRTY (original)': 'data/training_data_lean.csv',
    'CLEAN (repaired)': 'data/training_data_lean_cleaned.csv'
}

results = {}

for name, filepath in datasets.items():
    print(f"\n{'='*70}")
    print(f"TRAINING: {name}")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Get features
    exclude_cols = ['date', 'game_id', 'home_team', 'away_team', 'season', 
                   'target_spread', 'target_spread_cover', 'target_moneyline_win', 
                   'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target_moneyline_win']
    
    print(f"\n1. Dataset:")
    print(f"   Games: {len(df):,}")
    print(f"   Features: {len(feature_cols)}")
    
    # Use Syndicate-inspired parameters
    params = {
        'learning_rate': 0.02,
        'n_estimators': 2000,
        'max_depth': 9,
        'min_child_weight': 15,
        'gamma': 2.5,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'colsample_bylevel': 0.6,
        'colsample_bynode': 0.6,
        'reg_alpha': 5.0,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1.0,
        'max_delta_step': 5,
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }
    
    # Time series CV
    print("\n2. Time Series Cross-Validation (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        ll = log_loss(y_val, y_pred_proba)
        
        cv_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc, 'logloss': ll})
        print(f"   Fold {fold}: Acc={acc:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}")
    
    cv_df = pd.DataFrame(cv_scores)
    
    print(f"\n3. Mean Performance:")
    print(f"   Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
    print(f"   AUC:      {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
    print(f"   LogLoss:  {cv_df['logloss'].mean():.4f} ± {cv_df['logloss'].std():.4f}")
    
    # Train final model
    print("\n4. Training final model...")
    start = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    train_time = time.time() - start
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n5. Top 10 Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']:<30} {row['importance']:>8.4f}")
    
    # Store results
    results[name] = {
        'mean_auc': cv_df['auc'].mean(),
        'std_auc': cv_df['auc'].std(),
        'mean_accuracy': cv_df['accuracy'].mean(),
        'std_accuracy': cv_df['accuracy'].std(),
        'mean_logloss': cv_df['logloss'].mean(),
        'train_time': train_time,
        'top_feature': importance_df.iloc[0]['feature'],
        'top_importance': importance_df.iloc[0]['importance']
    }

# COMPARISON
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)

dirty = results['DIRTY (original)']
clean = results['CLEAN (repaired)']

print(f"\n{'Metric':<20} {'DIRTY':<15} {'CLEAN':<15} {'Change':<15}")
print("-"*70)

auc_change = (clean['mean_auc'] - dirty['mean_auc']) * 100
acc_change = (clean['mean_accuracy'] - dirty['mean_accuracy']) * 100
ll_change = (clean['mean_logloss'] - dirty['mean_logloss']) * 100

print(f"{'Mean AUC':<20} {dirty['mean_auc']:<15.4f} {clean['mean_auc']:<15.4f} {auc_change:>+6.2f}%")
print(f"{'Mean Accuracy':<20} {dirty['mean_accuracy']:<15.4f} {clean['mean_accuracy']:<15.4f} {acc_change:>+6.2f}%")
print(f"{'Mean LogLoss':<20} {dirty['mean_logloss']:<15.4f} {clean['mean_logloss']:<15.4f} {ll_change:>+6.2f}%")
print(f"{'Training Time':<20} {dirty['train_time']:<15.2f} {clean['train_time']:<15.2f} {clean['train_time']/dirty['train_time']*100-100:>+6.2f}%")

print("\n" + "="*70)
print("VERDICT")
print("="*70)

if clean['mean_auc'] > dirty['mean_auc']:
    improvement = (clean['mean_auc'] - dirty['mean_auc']) * 100
    print(f"✓ CLEANING IMPROVED PERFORMANCE!")
    print(f"  AUC gain: +{improvement:.2f} percentage points")
    print(f"  This validates the data quality fixes:")
    print(f"    - Rest days bug (286 days → 3 for openers)")
    print(f"    - ELO inflation (10.7% drift → 0%)")
    print(f"    - ELO outliers clipped (±500 → ±400)")
    
    if clean['mean_auc'] > 0.56:
        print(f"\n🎯 BROKE THE 0.56 BARRIER!")
        print(f"   AUC: {clean['mean_auc']:.4f}")
    elif clean['mean_auc'] > 0.555:
        print(f"\n📈 CLOSE TO BREAKING 0.56!")
        print(f"   AUC: {clean['mean_auc']:.4f}")
        print(f"   Need: +{(0.56 - clean['mean_auc'])*100:.2f}pp")
else:
    loss = (dirty['mean_auc'] - clean['mean_auc']) * 100
    print(f"⚠️ CLEANING REDUCED PERFORMANCE")
    print(f"  AUC loss: -{loss:.2f} percentage points")
    print(f"  This suggests:")
    print(f"    - Model learned to exploit the bugs")
    print(f"    - Dirty data had accidental signal")
    print(f"    - Need to retune hyperparameters on clean data")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

if clean['mean_auc'] > dirty['mean_auc']:
    print("1. ✓ Use cleaned data going forward")
    print("2. Hyperparameter tune on clean data (Syndicate run)")
    print("3. Add advanced features (schematic matchups, PIE roster share)")
    print("4. Target: 0.58+ AUC with clean data + new features")
else:
    print("1. Hyperparameter tune on clean data (params optimized for bugs)")
    print("2. The model was overfitting to data quality issues")
    print("3. Clean data should generalize better to live betting")
    print("4. Expect clean model to outperform in production")

print("="*70)

# Save comparison
with open('output/clean_vs_dirty_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved: output/clean_vs_dirty_comparison.json")
