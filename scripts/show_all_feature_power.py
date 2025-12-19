"""
Show all features ranked by prediction power
"""

import pandas as pd

df = pd.read_csv('models/feature_importance_44features.csv')
df_sorted = df.sort_values('gain', ascending=False).reset_index(drop=True)

print('='*80)
print('ALL 44 FEATURES - PREDICTION POWER (GAIN %)')
print('='*80)
print(f'\n{"Rank":<6} {"Feature":<40} {"Gain%":<10} {"Weight%":<10}')
print('-'*80)

for i, row in df_sorted.iterrows():
    print(f'{i+1:<6} {row["feature"]:<40} {row["gain_pct"]:<10.3f} {row["weight_pct"]:<10.3f}')

print('='*80)
print(f'\nTop 10 account for: {df_sorted.head(10)["gain_pct"].sum():.1f}%')
print(f'Top 20 account for: {df_sorted.head(20)["gain_pct"].sum():.1f}%')
print(f'Top 30 account for: {df_sorted.head(30)["gain_pct"].sum():.1f}%')
print('='*80)
