"""
Backtest betting simulator using the tuned moneyline model.
Saves betting log and performance plot to `V2/reporting/`.

Usage: python V2/scripts/backtest_betting.py
"""
import os
import json
import math
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Config
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'training_data_ready_for_ats.vegas.csv')
MODEL_PATHS = [
    os.path.join(ROOT, 'models', 'trained', 'moneyline_xgb_tuned_v1.joblib'),
    os.path.join(ROOT, 'models', 'trained', 'moneyline_xgb_tuned.joblib')
]
OUTPUT_DIR = os.path.join(ROOT, 'reporting')
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_CSV = os.path.join(OUTPUT_DIR, 'betting_log.csv')
PLOT_PATH = os.path.join(OUTPUT_DIR, 'betting_performance.png')

INITIAL_BANKROLL = 10000.0
KELLY_FRACTION = 0.125
MAX_BET_PCT = 0.05
MIN_EDGE = 0.04  # updated to calibrated sweet-spot (4%)

FEATURE_BLACKLIST = [
    'covers_spread', 'home_cover_spread', 'canonical_target',
    'home_wins', 'home_won', 'away_won', 'goes_over',
    'home_score', 'away_score', 'total_points',
    'margin', 'home_margin', 'actual_spread',
    'vegas_spread', 'closing_spread', 'opening_spread',
    'raw_spread', 'sharp_spread', 'closing_total', 'opening_total',
    'spread_movement', 'spread_abs_movement', 'is_steam_spread',
    'home_odds', 'away_odds', 'moneyline'
]

# Helpers for odds conversion
def implied_prob_from_american(od):
    if pd.isna(od) or od == 0:
        return np.nan
    od = float(od)
    if od < 0:
        return (-od) / (-od + 100)
    else:
        return 100.0 / (od + 100.0)

def decimal_from_american(od):
    if pd.isna(od) or od == 0:
        return np.nan
    od = float(od)
    if od < 0:
        return 1.0 + (100.0 / -od)
    else:
        return 1.0 + (od / 100.0)

# Find model path
model_path = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        model_path = p
        break
if model_path is None:
    raise FileNotFoundError(f"No tuned model found in {MODEL_PATHS}; train moneyline model first")

print('Using model:', model_path)
model = joblib.load(model_path)

# Load data
print('Loading data:', DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)
df['game_date'] = pd.to_datetime(df['game_date'])
df = df.sort_values('game_date').reset_index(drop=True)

# Determine test holdout (last 20%) — same logic used earlier
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].copy()
print('Simulating on holdout games:', len(test_df))

# Determine odds column: prefer 'moneyline' then 'home_odds' then others
od_cols = ['moneyline','home_odds','home_moneyline','home_ml','home_ml_odds']
od_col = None
for c in od_cols:
    if c in test_df.columns:
        od_col = c
        break
if od_col is None:
    # fallback try some likely names
    for c in ['home_odds','home_moneyline','home_ml']:
        if c in test_df.columns:
            od_col = c
            break
if od_col is None:
    raise RuntimeError('No moneyline/home-odds column found in CSV; cannot simulate')
print('Using odds column:', od_col)

# Feature selection: numeric columns minus blacklist and excluding target
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in FEATURE_BLACKLIST and c != 'home_wins' and c != 'season_start']
print('Feature columns count:', len(feature_cols))

# Prepare X_test for model. Fill missing cols with training medians computed on full training portion
train_df = df.iloc[:split_idx]
medians = train_df[feature_cols].median()
X_test = test_df.reindex(columns=feature_cols)
# If any feature missing entirely, add column with zeros
for c in feature_cols:
    if c not in X_test.columns:
        X_test[c] = 0.0
X_test = X_test.fillna(medians)

# Predict
print('Predicting probabilities...')
probs = model.predict_proba(X_test)[:,1]
test_df = test_df.copy()
test_df['model_prob'] = probs

# Simulate bets
bankroll = INITIAL_BANKROLL
history = []
bets_placed = 0
wins = 0

for idx, row in test_df.iterrows():
    odds = row.get(od_col)
    if pd.isna(odds) or odds == 0:
        continue
    implied = implied_prob_from_american(odds)
    model_p = float(row['model_prob'])
    edge = model_p - implied

    bet_size = 0.0
    profit = 0.0
    outcome = 'Pass'

    if edge > MIN_EDGE:
        decimal = decimal_from_american(odds)
        b = decimal - 1.0
        q = 1.0 - model_p
        # Kelly fraction
        if b == 0:
            kelly_pct = 0.0
        else:
            kelly_pct = (b * model_p - q) / b
        kelly_pct = max(0.0, kelly_pct)
        bet_pct = min(kelly_pct * KELLY_FRACTION, MAX_BET_PCT)
        bet_size = bankroll * bet_pct

        if bet_size > 0 and bankroll > 0:
            bets_placed += 1
            # Determine real outcome
            did_win = False
            if 'home_wins' in row.index:
                did_win = (row['home_wins'] == 1)
            elif 'home_score' in row.index and 'away_score' in row.index and not (pd.isna(row['home_score']) or pd.isna(row['away_score'])):
                did_win = (row['home_score'] > row['away_score'])

            if did_win:
                profit = bet_size * (decimal - 1.0)
                bankroll += profit
                outcome = 'Win'
                wins += 1
            else:
                profit = -bet_size
                bankroll += profit
                outcome = 'Loss'

    history.append({
        'date': row['game_date'],
        'home_team': row.get('home_team') or row.get('home_team_id') or '',
        'odds': odds,
        'model_prob': round(model_p, 4),
        'implied_prob': round(float(implied) if not pd.isna(implied) else math.nan, 4),
        'edge': round(float(edge), 4),
        'bet_size': round(float(bet_size), 2),
        'result': outcome,
        'profit': round(float(profit), 2),
        'bankroll': round(float(bankroll), 2)
    })

# Save results
res_df = pd.DataFrame(history)
res_df.to_csv(REPORT_CSV, index=False)

# Summary
final_bankroll = bankroll
total_roi = (final_bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL if INITIAL_BANKROLL else 0.0
win_rate = wins / bets_placed if bets_placed > 0 else 0.0

print('\n=== SIMULATION RESULTS ===')
print('Final Bankroll: $', round(final_bankroll,2))
print('Total Profit: $', round(final_bankroll - INITIAL_BANKROLL,2))
print('ROI:', f"{total_roi:.2%}")
print('Bets Placed:', bets_placed)
print('Win Rate:', f"{win_rate:.2%}")
print('Detailed log:', REPORT_CSV)

# Plot
if not res_df.empty:
    plt.figure(figsize=(10,6))
    plt.plot(pd.to_datetime(res_df['date']), res_df['bankroll'], marker='o', linewidth=1)
    plt.title(f'Bankroll Simulation (Kelly {int(KELLY_FRACTION*100)}%, Max Bet {int(MAX_BET_PCT*100)}%)')
    plt.xlabel('Date')
    plt.ylabel('Bankroll ($)')
    plt.axhline(y=INITIAL_BANKROLL, color='r', linestyle='--', label='Starting Balance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print('Performance chart saved to', PLOT_PATH)
else:
    print('No bets placed; adjust MIN_EDGE or verify odds column')

# Also save a small summary JSON
summary = {
    'final_bankroll': final_bankroll,
    'profit': final_bankroll - INITIAL_BANKROLL,
    'roi': total_roi,
    'bets_placed': bets_placed,
    'win_rate': win_rate
}
with open(os.path.join(OUTPUT_DIR, 'betting_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('Summary saved to', os.path.join(OUTPUT_DIR, 'betting_summary.json'))
