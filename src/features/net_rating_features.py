import sqlite3
import pandas as pd
import numpy as np

class NetRatingFeatureCalculator:
    def __init__(self, db_path):
        self.db_path = db_path

    def compute_net_rating_diffs(self, home_team, away_team, game_date):
        """
        Computes rolling and EWMA Net Rating differentials strictly using PRE-GAME data.
        Returns a dictionary of features.
        """
        # Default return if calculation fails or no history
        default_res = {
            'net_rating_l5_diff': 0.0,
            'net_rating_l10_diff': 0.0,
            'net_rating_ewma_diff': 0.0,
            'home_net_rating': 0.0,
            'away_net_rating': 0.0
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # --- THE FIX: STRICTLY PRE-GAME DATA (< ?) ---
            # We fetch the raw Game Logs strictly BEFORE the target date.
            # Net Rating = (PTS - OPP_PTS) / POSS * 100
            
            query = """
                SELECT 
                    game_date, 
                    (PTS - OPP_PTS) as margin,
                    POSS_EST as poss
                FROM game_logs
                WHERE team_abbreviation = ? 
                AND game_date < ? 
                ORDER BY game_date DESC
            """
            
            # Fetch Home History
            h_df = pd.read_sql(query, conn, params=(home_team, game_date))
            
            # Fetch Away History
            a_df = pd.read_sql(query, conn, params=(away_team, game_date))
            
            conn.close()
            
            if h_df.empty or a_df.empty:
                return default_res

            # Calculate Net Rating for each game in history
            # Avoid divide by zero: if poss is 0, assume league avg ~98
            h_df['poss'] = h_df['poss'].replace(0, 98.0) 
            a_df['poss'] = a_df['poss'].replace(0, 98.0)
            
            h_df['net_rtg'] = (h_df['margin'] / h_df['poss']) * 100
            a_df['net_rtg'] = (a_df['margin'] / a_df['poss']) * 100
            
            # --- CALCULATE FEATURES ---
            
            # 1. Rolling L5
            h_l5 = h_df['net_rtg'].head(5).mean()
            a_l5 = a_df['net_rtg'].head(5).mean()
            
            # 2. Rolling L10
            h_l10 = h_df['net_rtg'].head(10).mean()
            a_l10 = a_df['net_rtg'].head(10).mean()
            
            # 3. EWMA (Exponential Weighted Moving Average)
            # Span=10 matches the pace of NBA form shifts.
            # Pandas ewm requires ascending data (oldest -> newest).
            # We reverse [::-1], calc ewm, then take the last value.
            if len(h_df) > 0:
                h_ewma = h_df['net_rtg'].iloc[::-1].ewm(span=10, adjust=False).mean().iloc[-1]
            else:
                h_ewma = 0.0
                
            if len(a_df) > 0:
                a_ewma = a_df['net_rtg'].iloc[::-1].ewm(span=10, adjust=False).mean().iloc[-1]
            else:
                a_ewma = 0.0
            
            # Handle potential NaNs from short history
            if pd.isna(h_l5): h_l5 = 0.0
            if pd.isna(a_l5): a_l5 = 0.0
            if pd.isna(h_l10): h_l10 = 0.0
            if pd.isna(a_l10): a_l10 = 0.0
            if pd.isna(h_ewma): h_ewma = 0.0
            if pd.isna(a_ewma): a_ewma = 0.0

            return {
                'net_rating_l5_diff': h_l5 - a_l5,
                'net_rating_l10_diff': h_l10 - a_l10,
                'net_rating_ewma_diff': h_ewma - a_ewma,
                'home_net_rating': h_ewma,
                'away_net_rating': a_ewma
            }

        except Exception as e:
            # print(f"Error computing net rating: {e}")
            return default_res

# Direct helper for integration with extractor
def compute_net_rating_diffs(home_team, away_team, game_date, db_path):
    calc = NetRatingFeatureCalculator(db_path)
    return calc.compute_net_rating_diffs(home_team, away_team, game_date)
