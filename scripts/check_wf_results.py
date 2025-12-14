import pandas as pd

df = pd.read_csv('output/walk_forward_results.csv')
print(f'Total games: {len(df)}')
print(f'Accuracy: {(df["model_pred"] == df["target_moneyline_win"]).mean():.4f}')
print(f'Date range: {df["game_date"].min()} to {df["game_date"].max()}')
