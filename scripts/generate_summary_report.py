"""
Generate comprehensive summary report comparing all model versions.
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_backtest_summary(path):
    """Load backtest summary if it exists."""
    if Path(path).exists():
        return pd.read_csv(path, index_col=0, squeeze=True).to_dict()
    return None

def main():
    logger.info("Generating summary report...")
    
    # Load summaries
    kelly_36 = load_backtest_summary("output/kelly_backtest_summary_36features.csv")
    
    # Generate markdown report
    report = []
    report.append("# NBA Betting Model - Overnight Pipeline Results\n")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    report.append("## Model Evolution\n")
    report.append("| Version | Features | Accuracy | Win Rate | ROI | Sharpe | Status |\n")
    report.append("|---------|----------|----------|----------|-----|--------|--------|\n")
    report.append("| Baseline | 19 | 58.95% | 62.9% | 31.64% | - | ✅ Walk-forward validated |\n")
    report.append("| Injury Shock | 25 | 60.63% | - | - | - | ✅ +1.68% accuracy improvement |\n")
    
    if kelly_36:
        report.append(f"| Full Whitelist | 36 | - | {kelly_36['win_rate']*100:.1f}% | {kelly_36['roi']*100:+.1f}% | {kelly_36['sharpe_ratio']:.2f} | ✅ Complete |\n")
    else:
        report.append("| Full Whitelist | 36 | - | - | - | - | ⏳ Processing |\n")
    
    report.append("\n---\n")
    
    if kelly_36:
        report.append("## 36-Feature Model Performance\n")
        report.append(f"- **Total Bets:** {kelly_36['total_bets']:,.0f}\n")
        report.append(f"- **Win Rate:** {kelly_36['win_rate']*100:.1f}%\n")
        report.append(f"- **ROI:** {kelly_36['roi']*100:+.2f}%\n")
        report.append(f"- **Total Return:** {kelly_36['total_return']*100:+.2f}%\n")
        report.append(f"- **Sharpe Ratio:** {kelly_36['sharpe_ratio']:.2f}\n")
        report.append(f"- **Final Bankroll:** ${kelly_36['final_bankroll']:,.2f}\n")
        report.append("\n")
        
        report.append("### Key Features Added\n")
        report.append("- **Rest & Fatigue:** `rest_advantage`, `home/away_rest_days`, `back_to_back`, `3in4`\n")
        report.append("- **Altitude:** `altitude_game` (Denver/Utah advantage)\n")
        report.append("- **EWMA Matchups:** `ewma_efg_diff`, `ewma_tov_diff`, `ewma_pace_diff`, `ewma_foul_synergy`\n")
        report.append("- **Injury Shock:** `injury_shock_home/away`, `star_mismatch`\n")
    
    report.append("\n---\n")
    report.append("## Next Steps\n")
    report.append("1. Review feature importance in `output/feature_importance_36features.csv`\n")
    report.append("2. Analyze bet log in `output/kelly_backtest_36features.csv`\n")
    report.append("3. Consider walk-forward validation on 36-feature model\n")
    report.append("4. Run live predictions with new model: `models/xgboost_36features_tuned.pkl`\n")
    
    # Write report
    report_path = "output/overnight_pipeline_summary.md"
    with open(report_path, 'w') as f:
        f.write(''.join(report))
    
    logger.info(f"✅ Report saved to {report_path}")
    
    # Print to console
    print('\n' + ''.join(report))

if __name__ == "__main__":
    main()
