"""
FORENSIC AUDIT: Deep dive on +36.7% ROI moneyline backtest
Tests for: underdog luck, team bias, time decay, calibration quality
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print("FORENSIC AUDIT: MONEYLINE BETTING PERFORMANCE")
print("="*70)

# Load the bet log from successful backtest
with open('models/backtest_moneyline_filtered.json', 'r') as f:
    backtest_data = json.load(f)

bet_log = pd.DataFrame(backtest_data['bet_log'])
bet_log['date'] = pd.to_datetime(bet_log['date'])

print(f"\nLoaded {len(bet_log)} bets from backtest")
print(f"Date range: {bet_log['date'].min().date()} to {bet_log['date'].max().date()}")
print(f"Overall ROI: {backtest_data['best_results']['roi']:.1f}%")
print(f"Overall Win Rate: {backtest_data['best_results']['win_rate']:.1f}%")

# Calculate cumulative profit
bet_log['cumulative'] = bet_log['profit'].cumsum()
bet_log['month'] = bet_log['date'].dt.to_period('M')

# =============================================================================
# TEST 1: THE "UNDERDOG LUCK" TEST
# Are we just getting lucky on longshots?
# =============================================================================
print(f"\n{'='*70}")
print("TEST 1: UNDERDOG LUCK ANALYSIS")
print("="*70)
print("\nQuestion: Is profit driven by lucky longshot wins?")

# Categorize by odds range
def categorize_odds(odds):
    if odds < -250:
        return 'Heavy Fav (<-250)'
    elif odds < -150:
        return 'Favorite (-250 to -150)'
    elif odds < -110:
        return 'Light Fav (-150 to -110)'
    elif odds <= 110:
        return 'Pick-em (-110 to +110)'
    elif odds <= 150:
        return 'Light Dog (+110 to +150)'
    elif odds <= 250:
        return 'Dog (+150 to +250)'
    else:
        return 'Longshot (>+250)'

bet_log['odds_bucket'] = bet_log['odds'].apply(categorize_odds)

# Order for display
bucket_order = [
    'Heavy Fav (<-250)',
    'Favorite (-250 to -150)',
    'Light Fav (-150 to -110)',
    'Pick-em (-110 to +110)',
    'Light Dog (+110 to +150)',
    'Dog (+150 to +250)',
    'Longshot (>+250)'
]

print(f"\n{'ODDS BUCKET':<25} | {'BETS':>5} | {'W-L':>8} | {'WIN%':>6} | {'PROFIT':>9} | {'ROI':>7} | {'% TOTAL'}")
print("-" * 95)

total_profit = bet_log['profit'].sum()
results_by_bucket = []

for bucket in bucket_order:
    group = bet_log[bet_log['odds_bucket'] == bucket]
    if len(group) == 0:
        continue
    
    wins = (group['result'] == 1).sum()
    losses = len(group) - wins
    win_pct = (wins / len(group)) * 100
    profit = group['profit'].sum()
    roi = (profit / len(group)) * 100
    pct_total = (profit / total_profit) * 100
    
    results_by_bucket.append({
        'bucket': bucket,
        'bets': len(group),
        'profit': profit,
        'roi': roi
    })
    
    status = "✓" if profit > 0 else "✗"
    print(f"{status} {bucket:<23} | {len(group):>5} | {wins:>3}-{losses:<3} | {win_pct:>5.1f}% | {profit:>+8.1f}u | {roi:>+6.1f}% | {pct_total:>6.1f}%")

# Verdict
longshot_profit = bet_log[bet_log['odds'] > 250]['profit'].sum()
longshot_pct = (longshot_profit / total_profit) * 100 if total_profit > 0 else 0

print(f"\n{'='*70}")
print("VERDICT: UNDERDOG LUCK TEST")
print("="*70)
if longshot_pct > 60:
    print(f"🚨 RED FLAG: {longshot_pct:.1f}% of profit from longshots (>+250)")
    print("   This is high variance and likely to regress.")
elif longshot_pct > 40:
    print(f"⚠️  CAUTION: {longshot_pct:.1f}% of profit from longshots (>+250)")
    print("   Moderate variance. Monitor closely.")
else:
    print(f"✅ PASS: Only {longshot_pct:.1f}% of profit from longshots (>+250)")
    print("   Profit is from solid favorites/pick-ems. Low variance.")

# Check if favorites are profitable
fav_profit = bet_log[bet_log['odds'] < -110]['profit'].sum()
fav_roi = (fav_profit / len(bet_log[bet_log['odds'] < -110])) * 100 if len(bet_log[bet_log['odds'] < -110]) > 0 else 0
print(f"\n   Favorites (<-110): {fav_roi:+.1f}% ROI")
if fav_roi > 10:
    print(f"   ✅ Strong edge on favorites - model is fundamentally sound")

# =============================================================================
# TEST 2: THE "TEAM BIAS" TEST
# Are we just exploiting one mispriced team?
# =============================================================================
print(f"\n{'='*70}")
print("TEST 2: TEAM BIAS ANALYSIS")
print("="*70)
print("\nQuestion: Is profit concentrated in a few teams?")

# Analyze by team (both when betting on them home/away)
team_stats = []

# Get all unique teams
all_teams = set(bet_log['home'].unique()) | set(bet_log['away'].unique())

for team in all_teams:
    # When we bet on this team
    team_bets = bet_log[
        ((bet_log['pick'] == 'HOME') & (bet_log['home'] == team)) |
        ((bet_log['pick'] == 'AWAY') & (bet_log['away'] == team))
    ]
    
    if len(team_bets) > 0:
        team_stats.append({
            'team': team,
            'bets': len(team_bets),
            'wins': (team_bets['result'] == 1).sum(),
            'profit': team_bets['profit'].sum(),
            'roi': (team_bets['profit'].sum() / len(team_bets)) * 100
        })

team_df = pd.DataFrame(team_stats).sort_values('profit', ascending=False)

print(f"\nTOP 10 MOST PROFITABLE TEAMS (Betting ON them):")
print(f"{'TEAM':>4} | {'BETS':>5} | {'W-L':>8} | {'PROFIT':>9} | {'ROI':>7}")
print("-" * 50)
for _, row in team_df.head(10).iterrows():
    losses = row['bets'] - row['wins']
    print(f"{row['team']:>4} | {row['bets']:>5} | {int(row['wins']):>3}-{int(losses):<3} | {row['profit']:>+8.1f}u | {row['roi']:>+6.1f}%")

print(f"\nBOTTOM 5 TEAMS (Worst performers when betting ON them):")
print(f"{'TEAM':>4} | {'BETS':>5} | {'W-L':>8} | {'PROFIT':>9} | {'ROI':>7}")
print("-" * 50)
for _, row in team_df.tail(5).iterrows():
    losses = row['bets'] - row['wins']
    print(f"{row['team']:>4} | {row['bets']:>5} | {int(row['wins']):>3}-{int(losses):<3} | {row['profit']:>+8.1f}u | {row['roi']:>+6.1f}%")

# Calculate concentration
top5_profit = team_df.head(5)['profit'].sum()
concentration_pct = (top5_profit / total_profit) * 100 if total_profit > 0 else 0

print(f"\n{'='*70}")
print("VERDICT: TEAM BIAS TEST")
print("="*70)
if concentration_pct > 70:
    print(f"🚨 RED FLAG: Top 5 teams account for {concentration_pct:.1f}% of profit")
    print("   Edge may be temporary or team-specific.")
elif concentration_pct > 50:
    print(f"⚠️  CAUTION: Top 5 teams account for {concentration_pct:.1f}% of profit")
    print("   Moderate concentration. Diversify bet selection.")
else:
    print(f"✅ PASS: Top 5 teams account for {concentration_pct:.1f}% of profit")
    print("   Profit is well-distributed across teams.")

# =============================================================================
# TEST 3: THE "TIME DECAY" TEST
# Did the edge disappear over time?
# =============================================================================
print(f"\n{'='*70}")
print("TEST 3: TIME DECAY ANALYSIS")
print("="*70)
print("\nQuestion: Is the edge consistent or degrading?")

print(f"\n{'MONTH':<10} | {'BETS':>5} | {'W-L':>8} | {'WIN%':>6} | {'PROFIT':>9} | {'ROI':>7}")
print("-" * 65)

monthly_stats = []
for month, group in bet_log.groupby('month'):
    wins = (group['result'] == 1).sum()
    losses = len(group) - wins
    win_pct = (wins / len(group)) * 100
    profit = group['profit'].sum()
    roi = (profit / len(group)) * 100
    
    monthly_stats.append({
        'month': str(month),
        'bets': len(group),
        'profit': profit,
        'roi': roi
    })
    
    status = "✓" if profit > 0 else "✗"
    print(f"{status} {str(month):<8} | {len(group):>5} | {int(wins):>3}-{int(losses):<3} | {win_pct:>5.1f}% | {profit:>+8.1f}u | {roi:>+6.1f}%")

monthly_df = pd.DataFrame(monthly_stats)

# Check for degradation
positive_months = (monthly_df['profit'] > 0).sum()
total_months = len(monthly_df)
positive_pct = (positive_months / total_months) * 100

# Calculate trend (simple regression slope)
monthly_df['month_num'] = range(len(monthly_df))
if len(monthly_df) > 2:
    slope = np.polyfit(monthly_df['month_num'], monthly_df['roi'], 1)[0]
else:
    slope = 0

print(f"\n{'='*70}")
print("VERDICT: TIME DECAY TEST")
print("="*70)
if positive_months < total_months * 0.5:
    print(f"🚨 RED FLAG: Only {positive_months}/{total_months} positive months ({positive_pct:.0f}%)")
    print("   Edge is unstable.")
elif slope < -5:
    print(f"⚠️  CAUTION: ROI declining over time (slope: {slope:.2f})")
    print("   Market may be adjusting. Monitor closely.")
else:
    print(f"✅ PASS: {positive_months}/{total_months} positive months ({positive_pct:.0f}%)")
    if slope > 0:
        print(f"   ✅ Edge is stable or improving (slope: {slope:.2f})")
    else:
        print(f"   Edge is stable (slope: {slope:.2f})")

# =============================================================================
# TEST 4: THE "CALIBRATION" TEST
# When we say 10% edge, do we actually have 10% edge?
# =============================================================================
print(f"\n{'='*70}")
print("TEST 4: CALIBRATION QUALITY")
print("="*70)
print("\nQuestion: Are our edge estimates accurate?")

# Bin by predicted edge
bet_log['edge_bin'] = pd.cut(bet_log['edge'], bins=[0, 0.08, 0.12, 0.16, 0.20, 1.0],
                              labels=['5-8%', '8-12%', '12-16%', '16-20%', '>20%'])

print(f"\n{'EDGE BIN':<10} | {'BETS':>5} | {'WIN%':>6} | {'PROFIT':>9} | {'ROI':>7} | {'EXPECTED'}")
print("-" * 70)

for edge_bin, group in bet_log.groupby('edge_bin', observed=True):
    if len(group) == 0:
        continue
    
    wins = (group['result'] == 1).sum()
    win_pct = (wins / len(group)) * 100
    profit = group['profit'].sum()
    roi = (profit / len(group)) * 100
    avg_edge = group['edge'].mean() * 100
    
    # Check if actual ROI matches expected edge
    diff = roi - avg_edge
    match = "✓" if abs(diff) < 10 else "✗"
    
    print(f"{match} {edge_bin:<8} | {len(group):>5} | {win_pct:>5.1f}% | {profit:>+8.1f}u | {roi:>+6.1f}% | {avg_edge:>+6.1f}%")

print(f"\n{'='*70}")
print("VERDICT: CALIBRATION TEST")
print("="*70)
# Simple check: does higher edge = higher ROI?
edge_bins = bet_log.groupby('edge_bin', observed=True)['profit'].apply(lambda x: (x.sum() / len(x)) * 100)
is_monotonic = edge_bins.is_monotonic_increasing

if is_monotonic:
    print("✅ PASS: Higher predicted edges → higher actual ROI")
    print("   Model calibration is sound.")
else:
    print("⚠️  CAUTION: Edge estimates not monotonic")
    print("   Model may need recalibration.")

# =============================================================================
# VISUALIZATION: EQUITY CURVE & DRAWDOWNS
# =============================================================================
print(f"\n{'='*70}")
print("GENERATING VISUALIZATIONS")
print("="*70)

# Create output directory
Path('output').mkdir(exist_ok=True)

# Equity curve
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Main equity curve
ax1.plot(range(len(bet_log)), bet_log['cumulative'], linewidth=2, color='blue')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_title('Equity Curve: Filtered Moneyline Backtest', fontsize=14, fontweight='bold')
ax1.set_xlabel('Bet Number')
ax1.set_ylabel('Cumulative Profit (units)')
ax1.grid(True, alpha=0.3)

# Calculate drawdown
running_max = bet_log['cumulative'].expanding().max()
drawdown = bet_log['cumulative'] - running_max

ax2.fill_between(range(len(bet_log)), drawdown, 0, alpha=0.3, color='red')
ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
ax2.set_xlabel('Bet Number')
ax2.set_ylabel('Drawdown (units)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/forensic_audit_equity.png', dpi=150, bbox_inches='tight')
print("✓ Saved: output/forensic_audit_equity.png")

# Monthly performance
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if x > 0 else 'red' for x in monthly_df['profit']]
ax.bar(range(len(monthly_df)), monthly_df['profit'], color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xticks(range(len(monthly_df)))
ax.set_xticklabels(monthly_df['month'], rotation=45)
ax.set_title('Monthly Profit/Loss', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Profit (units)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('output/forensic_audit_monthly.png', dpi=150, bbox_inches='tight')
print("✓ Saved: output/forensic_audit_monthly.png")

# Odds bucket distribution
fig, ax = plt.subplots(figsize=(10, 6))
bucket_results = []
bucket_labels = []
for bucket in bucket_order:
    group = bet_log[bet_log['odds_bucket'] == bucket]
    if len(group) > 0:
        roi = (group['profit'].sum() / len(group)) * 100
        bucket_results.append(roi)
        bucket_labels.append(bucket)

colors = ['green' if x > 0 else 'red' for x in bucket_results]
ax.bar(range(len(bucket_labels)), bucket_results, color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xticks(range(len(bucket_labels)))
ax.set_xticklabels(bucket_labels, rotation=45, ha='right')
ax.set_title('ROI by Odds Range', fontsize=14, fontweight='bold')
ax.set_xlabel('Odds Bucket')
ax.set_ylabel('ROI (%)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('output/forensic_audit_odds_buckets.png', dpi=150, bbox_inches='tight')
print("✓ Saved: output/forensic_audit_odds_buckets.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*70}")
print("FORENSIC AUDIT SUMMARY")
print("="*70)

max_drawdown = drawdown.min()
max_drawdown_pct = (max_drawdown / bet_log['cumulative'].max()) * 100 if bet_log['cumulative'].max() > 0 else 0

print(f"\nOverall Statistics:")
print(f"  Total Bets:        {len(bet_log)}")
print(f"  Win Rate:          {backtest_data['best_results']['win_rate']:.1f}%")
print(f"  Total Profit:      {backtest_data['best_results']['profit']:.1f} units")
print(f"  ROI:               {backtest_data['best_results']['roi']:.1f}%")
print(f"  Max Drawdown:      {max_drawdown:.1f} units ({max_drawdown_pct:.1f}%)")

print(f"\nTest Results:")
print(f"  1. Underdog Luck:  {'✅ PASS' if longshot_pct < 40 else '🚨 FAIL'}")
print(f"  2. Team Bias:      {'✅ PASS' if concentration_pct < 50 else '🚨 FAIL'}")
print(f"  3. Time Decay:     {'✅ PASS' if positive_pct > 65 else '🚨 FAIL'}")
print(f"  4. Calibration:    {'✅ PASS' if is_monotonic else '⚠️  CAUTION'}")

# Overall verdict
tests_passed = sum([
    longshot_pct < 40,
    concentration_pct < 50,
    positive_pct > 65,
    is_monotonic
])

print(f"\n{'='*70}")
print(f"FINAL VERDICT: {tests_passed}/4 TESTS PASSED")
print("="*70)

if tests_passed >= 3:
    print("✅ PRODUCTION READY")
    print("   Model demonstrates genuine edge with sound fundamentals.")
    print("   Proceed to paper trading with confidence.")
elif tests_passed >= 2:
    print("⚠️  CONDITIONAL APPROVAL")
    print("   Model shows promise but has some concerns.")
    print("   Start with micro stakes and monitor closely.")
else:
    print("🚨 NOT READY")
    print("   Model has significant red flags.")
    print("   Further development required before deployment.")

print(f"\n{'='*70}")
print("FORENSIC AUDIT COMPLETE")
print("="*70)
