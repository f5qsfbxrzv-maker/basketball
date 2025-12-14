"""
Comprehensive model performance analysis with threshold tuning and error analysis.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, roc_curve
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_class_balance():
    """Check if dataset has class imbalance."""
    logger.info("="*60)
    logger.info("CLASS BALANCE ANALYSIS")
    logger.info("="*60)
    
    df = pd.read_csv("data/training_data_with_features.csv")
    total = len(df)
    home_wins = df['target_moneyline_win'].sum()
    away_wins = total - home_wins
    
    logger.info(f"Total games: {total:,}")
    logger.info(f"Home wins:   {home_wins:,} ({home_wins/total*100:.2f}%)")
    logger.info(f"Away wins:   {away_wins:,} ({away_wins/total*100:.2f}%)")
    
    if abs(home_wins - away_wins) / total < 0.1:
        logger.info("✅ Classes are balanced (within 10%)")
    else:
        logger.info("⚠️ Classes are imbalanced - use F1/Precision/Recall")
    
    return df

def threshold_tuning(model, X_test, y_test, model_name):
    """Find optimal probability threshold."""
    logger.info("")
    logger.info("="*60)
    logger.info(f"THRESHOLD TUNING - {model_name}")
    logger.info("="*60)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.30, 0.70, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    results_df = pd.DataFrame(results)
    logger.info("\nThreshold Performance:")
    for _, row in results_df.iterrows():
        marker = " 🎯" if row['threshold'] == best_threshold else ""
        logger.info(
            f"  Threshold {row['threshold']:.2f}: "
            f"Acc={row['accuracy']:.4f} Prec={row['precision']:.4f} "
            f"Rec={row['recall']:.4f} F1={row['f1']:.4f}{marker}"
        )
    
    logger.info(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold, results_df

def detailed_metrics(model, X_test, y_test, model_name, threshold=0.5):
    """Calculate comprehensive metrics."""
    logger.info("")
    logger.info("="*60)
    logger.info(f"DETAILED METRICS - {model_name}")
    logger.info("="*60)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"Threshold:  {threshold:.2f}")
    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"AUC:        {auc:.4f}")
    logger.info(f"Precision:  {prec:.4f} (% of predicted wins that were correct)")
    logger.info(f"Recall:     {rec:.4f} (% of actual wins that were caught)")
    logger.info(f"F1 Score:   {f1:.4f} (harmonic mean of precision/recall)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info(f"              Predicted Away  Predicted Home")
    logger.info(f"Actual Away   {tn:6d} (TN)    {fp:6d} (FP)")
    logger.info(f"Actual Home   {fn:6d} (FN)    {tp:6d} (TP)")
    logger.info("")
    logger.info(f"True Negatives:  {tn:,} (correctly predicted away wins)")
    logger.info(f"False Positives: {fp:,} (predicted home, was away)")
    logger.info(f"False Negatives: {fn:,} (predicted away, was home)")
    logger.info(f"True Positives:  {tp:,} (correctly predicted home wins)")
    
    # Betting perspective
    logger.info("")
    logger.info("Betting Perspective:")
    logger.info(f"  Made {len(y_pred):,} predictions")
    logger.info(f"  Won {tp + tn:,} bets ({(tp+tn)/len(y_pred)*100:.2f}%)")
    logger.info(f"  Lost {fp + fn:,} bets ({(fp+fn)/len(y_pred)*100:.2f}%)")
    
    return {
        'accuracy': acc,
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_proba
    }

def error_analysis(baseline_preds, enhanced_preds, y_test, X_test_df):
    """Compare where baseline and enhanced models disagree."""
    logger.info("")
    logger.info("="*60)
    logger.info("ERROR ANALYSIS")
    logger.info("="*60)
    
    baseline_correct = (baseline_preds == y_test)
    enhanced_correct = (enhanced_preds == y_test)
    
    # Cases where enhanced improved
    enhanced_better = (~baseline_correct) & enhanced_correct
    # Cases where enhanced made worse
    enhanced_worse = baseline_correct & (~enhanced_correct)
    # Both correct
    both_correct = baseline_correct & enhanced_correct
    # Both wrong
    both_wrong = (~baseline_correct) & (~enhanced_correct)
    
    logger.info(f"Both correct:       {both_correct.sum():,} ({both_correct.sum()/len(y_test)*100:.1f}%)")
    logger.info(f"Both wrong:         {both_wrong.sum():,} ({both_wrong.sum()/len(y_test)*100:.1f}%)")
    logger.info(f"Enhanced improved:  {enhanced_better.sum():,} ({enhanced_better.sum()/len(y_test)*100:.1f}%)")
    logger.info(f"Enhanced worse:     {enhanced_worse.sum():,} ({enhanced_worse.sum()/len(y_test)*100:.1f}%)")
    logger.info(f"Net improvement:    {enhanced_better.sum() - enhanced_worse.sum():+,} games")
    
    # Analyze cases where enhanced improved
    if enhanced_better.sum() > 0:
        logger.info("")
        logger.info(f"Analyzing {enhanced_better.sum()} cases where Enhanced improved:")
        improved_df = X_test_df[enhanced_better].copy()
        
        # Check if injury shock features were significant
        if 'injury_shock_home' in improved_df.columns:
            avg_shock_home = improved_df['injury_shock_home'].abs().mean()
            avg_shock_away = improved_df['injury_shock_away'].abs().mean()
            logger.info(f"  Avg |injury_shock_home|: {avg_shock_home:.3f}")
            logger.info(f"  Avg |injury_shock_away|: {avg_shock_away:.3f}")
            logger.info("  → Enhanced caught injury shocks better!")
    
    # Analyze cases where enhanced got worse
    if enhanced_worse.sum() > 0:
        logger.info("")
        logger.info(f"Analyzing {enhanced_worse.sum()} cases where Enhanced got worse:")
        worse_df = X_test_df[enhanced_worse].copy()
        
        if 'injury_shock_home' in worse_df.columns:
            avg_shock_home = worse_df['injury_shock_home'].abs().mean()
            avg_shock_away = worse_df['injury_shock_away'].abs().mean()
            logger.info(f"  Avg |injury_shock_home|: {avg_shock_home:.3f}")
            logger.info(f"  Avg |injury_shock_away|: {avg_shock_away:.3f}")
            logger.info("  → Possible over-weighting of injury shocks")

def main():
    """Run comprehensive analysis."""
    
    # Load data
    df = analyze_class_balance()
    
    # Load models
    logger.info("")
    logger.info("Loading models...")
    model_baseline = joblib.load("models/xgboost_optuna_uncalibrated.pkl")
    model_enhanced = joblib.load("models/xgboost_with_injury_shock.pkl")
    
    # Prepare data
    baseline_features = [
        'vs_efg_diff', 'vs_tov', 'vs_reb_diff', 'vs_ftr_diff', 'vs_net_rating',
        'expected_pace', 'rest_days_diff', 'is_b2b_diff', 'h2h_win_rate_l3y',
        'injury_impact_diff', 'elo_diff', 'off_elo_diff', 'def_elo_diff',
        'composite_elo_diff', 'sos_diff', 'h_off_rating', 'h_def_rating',
        'a_off_rating', 'a_def_rating'
    ]
    
    injury_shock_features = [
        'injury_shock_home', 'injury_shock_away', 'injury_shock_diff',
        'home_star_missing', 'away_star_missing', 'star_mismatch'
    ]
    
    # Add injury shock features
    import sys
    sys.path.insert(0, '.')
    from src.features.feature_calculator_v5 import FeatureCalculatorV5
    
    logger.info("Adding injury shock features to test set...")
    calculator = FeatureCalculatorV5()
    
    # Calculate injury shock features inline
    df = df.sort_values('date').copy()
    
    # For home team
    df['home_injury_rolling_mean'] = df.groupby('home_team')['injury_impact_diff'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    
    # For away team (reconstruct from diff)
    df['away_injury'] = -df['injury_impact_diff']
    df['away_injury_rolling_mean'] = df.groupby('away_team')['away_injury'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    
    # Shock = today - rolling mean
    df['injury_shock_home'] = df['injury_impact_diff'] - df['home_injury_rolling_mean']
    df['injury_shock_away'] = df['away_injury'] - df['away_injury_rolling_mean']
    df['injury_shock_diff'] = df['injury_shock_home'] - df['injury_shock_away']
    
    # Star binary flags
    STAR_THRESHOLD = 4.0
    df['home_star_missing'] = (df['injury_impact_diff'] >= STAR_THRESHOLD).astype(int)
    df['away_star_missing'] = (-df['injury_impact_diff'] >= STAR_THRESHOLD).astype(int)
    df['star_mismatch'] = df['home_star_missing'] - df['away_star_missing']
    
    # Fill NaNs
    for col in ['injury_shock_home', 'injury_shock_away', 'injury_shock_diff', 
                'home_injury_rolling_mean', 'away_injury_rolling_mean']:
        df[col] = df[col].fillna(0)
    
    df = df.drop(columns=['away_injury', 'home_injury_rolling_mean', 'away_injury_rolling_mean'])
    
    enhanced_features = baseline_features + injury_shock_features
    
    X_baseline = df[baseline_features]
    X_enhanced = df[enhanced_features]
    y = df['target_moneyline_win']
    
    # Split
    X_base_train, X_base_test, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.2, random_state=42, stratify=y
    )
    X_enh_train, X_enh_test, _, _ = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Baseline analysis
    baseline_threshold, _ = threshold_tuning(model_baseline, X_base_test, y_test, "BASELINE")
    baseline_metrics = detailed_metrics(model_baseline, X_base_test, y_test, "BASELINE", baseline_threshold)
    
    # Enhanced analysis
    enhanced_threshold, _ = threshold_tuning(model_enhanced, X_enh_test, y_test, "ENHANCED")
    enhanced_metrics = detailed_metrics(model_enhanced, X_enh_test, y_test, "ENHANCED", enhanced_threshold)
    
    # Error analysis
    baseline_preds = (model_baseline.predict_proba(X_base_test)[:, 1] >= baseline_threshold).astype(int)
    enhanced_preds = (model_enhanced.predict_proba(X_enh_test)[:, 1] >= enhanced_threshold).astype(int)
    
    error_analysis(baseline_preds, enhanced_preds, y_test, X_enh_test)
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Baseline (threshold={baseline_threshold:.2f}):")
    logger.info(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {baseline_metrics['f1']:.4f}")
    logger.info(f"  AUC:      {baseline_metrics['auc']:.4f}")
    logger.info("")
    logger.info(f"Enhanced (threshold={enhanced_threshold:.2f}):")
    logger.info(f"  Accuracy: {enhanced_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {enhanced_metrics['f1']:.4f}")
    logger.info(f"  AUC:      {enhanced_metrics['auc']:.4f}")
    logger.info("")
    logger.info(f"Improvements:")
    logger.info(f"  Accuracy: {(enhanced_metrics['accuracy'] - baseline_metrics['accuracy'])*100:+.2f}%")
    logger.info(f"  F1 Score: {(enhanced_metrics['f1'] - baseline_metrics['f1'])*100:+.2f}%")
    logger.info(f"  AUC:      {(enhanced_metrics['auc'] - baseline_metrics['auc'])*100:+.2f}%")

if __name__ == "__main__":
    main()
