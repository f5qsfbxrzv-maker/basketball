import pandas as pd

df = pd.read_csv('output/feature_importance_pruned.csv')
df['rank'] = range(1, len(df) + 1)

print('\n📊 ALL 31 FEATURES (Ranked by Importance)')
print('='*80)
print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Category'}")
print('-'*80)

categories = {
    'ewma_efg': 'EWMA/Shooting',
    'ewma_tov': 'EWMA/Turnovers', 
    'ewma_orb': 'EWMA/Rebounding',
    'ewma_pace': 'EWMA/Pace',
    'ewma_vol': 'EWMA/Volume',
    'ewma_3p': 'EWMA/Shooting',
    'ewma_foul': 'Foul Synergy',
    'ewma_chaos': 'Chaos/Volatility',
    'ewma_net': 'Chaos/Volatility',
    'ewma_fta': 'EWMA/FreeThrows',
    'def_elo': 'ELO Engine',
    'off_elo': 'ELO Engine',
    'composite_elo': 'ELO Engine',
    'rest': 'Rest/Fatigue',
    'fatigue': 'Rest/Fatigue',
    'back_to_back': 'Rest/Fatigue',
    '3in4': 'Rest/Fatigue',
    'injury': 'Injury Context',
    'total_foul': 'Foul Synergy',
    'altitude': 'Altitude Effect',
    'orb': 'Rebounding',
}

for _, row in df.iterrows():
    feature = row['feature']
    category = 'Other'
    for key, cat in categories.items():
        if key in feature:
            category = cat
            break
    print(f"{row['rank']:<6} {feature:<30} {row['importance']:<12.4f} {category}")

print('='*80)

# Summary by category
print('\n📈 CATEGORY SUMMARY:')
print('='*80)

feature_categories = []
for _, row in df.iterrows():
    feature = row['feature']
    category = 'Other'
    for key, cat in categories.items():
        if key in feature:
            category = cat
            break
    feature_categories.append({'category': category, 'importance': row['importance']})

cat_df = pd.DataFrame(feature_categories)
summary = cat_df.groupby('category')['importance'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
summary.columns = ['Total Importance', 'Count', 'Avg Importance']

print(f"{'Category':<25} {'Total %':<12} {'Count':<8} {'Avg %'}")
print('-'*80)
for cat, row in summary.iterrows():
    print(f"{cat:<25} {row['Total Importance']*100:<11.2f}% {int(row['Count']):<8} {row['Avg Importance']*100:.2f}%")

print('='*80)
