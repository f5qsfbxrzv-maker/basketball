import pandas as pd

df = pd.read_csv('data/training_data_with_temporal_features.csv', nrows=10)

spread_cols = [c for c in df.columns if 'spread' in c.lower() or 'line' in c.lower()]

print('Columns with spread/line:')
for col in spread_cols:
    print(f'  {col}')

print('\nSample values:')
for col in spread_cols[:5]:
    print(f'\n{col}:')
    print(df[col].head())
