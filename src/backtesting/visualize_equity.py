"""
EQUITY CURVE VISUALIZER
Visual proof of stability vs volatility
Lines don't lie - drawdowns reveal the truth
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_thunderdome_results(results_a, results_b, labels=None):
    """
    Plot equity curves for two models
    
    Args:
        results_a: List/array of cumulative bankroll over time for Model A
        results_b: List/array of cumulative bankroll over time for Model B
        labels: Tuple of (label_a, label_b) for legend
    """
    if labels is None:
        labels = ('Model A (Production)', 'Model B (Challenger)')
    
    plt.figure(figsize=(14, 8))
    
    # Main equity curves
    plt.subplot(2, 1, 1)
    plt.plot(results_a, label=labels[0], color='green', linewidth=2)
    plt.plot(results_b, label=labels[1], color='red', linestyle='--', linewidth=2)
    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.title('The Thunderdome: Cumulative Profit (Equity Curve)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Bets', fontsize=12)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Drawdown curves
    plt.subplot(2, 1, 2)
    
    # Calculate drawdowns
    running_max_a = pd.Series(results_a).cummax()
    drawdown_a = pd.Series(results_a) - running_max_a
    
    running_max_b = pd.Series(results_b).cummax()
    drawdown_b = pd.Series(results_b) - running_max_b
    
    plt.plot(drawdown_a, label=f'{labels[0]} Drawdown', color='darkgreen', linewidth=2)
    plt.plot(drawdown_b, label=f'{labels[1]} Drawdown', color='darkred', linestyle='--', linewidth=2)
    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.title('Drawdown Analysis (Lower is Better)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Bets', fontsize=12)
    plt.ylabel('Drawdown ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add metrics annotations
    max_dd_a = drawdown_a.min()
    max_dd_b = drawdown_b.min()
    
    plt.text(0.02, 0.98, 
             f'Max Drawdown A: ${max_dd_a:.2f}\nMax Drawdown B: ${max_dd_b:.2f}',
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the chart
    Path("logs").mkdir(exist_ok=True)
    output_path = "logs/thunderdome_equity_curve.png"
    plt.savefig(output_path, dpi=150)
    print(f"📸 Equity curve saved to: {output_path}")
    
    plt.show()

def calculate_risk_metrics(bankroll_history):
    """
    Calculate Sharpe ratio and max drawdown
    
    Args:
        bankroll_history: List/array of cumulative bankroll values
        
    Returns:
        tuple: (sharpe_ratio, max_drawdown, max_drawdown_pct)
    """
    if len(bankroll_history) < 2:
        return 0, 0, 0
    
    # Convert cumulative bankroll to per-bet returns
    returns = pd.Series(bankroll_history).diff().dropna()
    
    if len(returns) < 2:
        return 0, 0, 0
    
    avg_return = returns.mean()
    std_dev = returns.std()
    
    # Sharpe Ratio (Simplified for betting)
    # > 0.1 per bet is excellent
    sharpe = avg_return / std_dev if std_dev != 0 else 0
    
    # Max Drawdown (The "Pain" Metric)
    running_max = pd.Series(bankroll_history).cummax()
    drawdown = pd.Series(bankroll_history) - running_max
    max_drawdown = drawdown.min()
    
    # Max Drawdown Percentage
    max_dd_pct = (max_drawdown / running_max.max() * 100) if running_max.max() != 0 else 0
    
    return sharpe, max_drawdown, max_dd_pct

def print_risk_report(bankroll_history, model_name="Model"):
    """Print detailed risk metrics for a bankroll history"""
    
    sharpe, max_dd, max_dd_pct = calculate_risk_metrics(bankroll_history)
    
    final_profit = bankroll_history[-1] if bankroll_history else 0
    peak = max(bankroll_history) if bankroll_history else 0
    
    print(f"\n📊 RISK METRICS: {model_name}")
    print("=" * 60)
    print(f"Final Profit:        ${final_profit:,.2f}")
    print(f"Peak Profit:         ${peak:,.2f}")
    print(f"Sharpe Ratio:        {sharpe:.3f} {'✅ Excellent' if sharpe > 0.1 else '⚠️  Poor' if sharpe < 0.05 else '➖ Fair'}")
    print(f"Max Drawdown:        ${max_dd:,.2f}")
    print(f"Max Drawdown %:      {max_dd_pct:.1f}%")
    print("=" * 60)
    
    # Risk assessment
    if max_dd_pct < -10:
        print("⚠️  WARNING: High drawdown risk (>10%)")
    elif max_dd_pct < -20:
        print("🚨 DANGER: Severe drawdown risk (>20%)")
    else:
        print("✅ Acceptable drawdown risk")
    
    return sharpe, max_dd, max_dd_pct

# Example usage
if __name__ == "__main__":
    # Simulate example data
    np.random.seed(42)
    
    # Model A: Steady growth with small drawdowns
    results_a = np.cumsum(np.random.normal(5, 20, 500))
    
    # Model B: Volatile with larger swings
    results_b = np.cumsum(np.random.normal(4, 35, 500))
    
    print("=" * 80)
    print("EQUITY CURVE VISUALIZER - DEMO")
    print("=" * 80)
    
    print_risk_report(results_a, "Model A (Steady)")
    print_risk_report(results_b, "Model B (Volatile)")
    
    plot_thunderdome_results(results_a, results_b)
