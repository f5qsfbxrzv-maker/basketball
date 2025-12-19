"""
FEATURE IMPORTANCE RANKING & KALSHI INTEGRATION PLAN
=====================================================

## TOP 20 FEATURES (BY IMPORTANCE - GAIN)

1.  season_year                    15.8  (Time context - currently defaulted to 0)
2.  season_year_normalized         15.2  (Time context - currently defaulted to 0)
3.  ewma_efg_diff                  11.9  ✅ Four Factors (Shooting efficiency)
4.  off_elo_diff                    9.3  ✅ ELO ratings
5.  away_back_to_back               9.2  ✅ Rest/fatigue
6.  def_elo_diff                    8.6  ✅ ELO ratings
7.  away_rest_days                  8.3  ✅ Rest/fatigue
8.  ewma_tov_diff                   8.2  ✅ Four Factors (Turnovers)
9.  altitude_game                   8.2  ✅ Venue/location
10. away_3in4                       7.9  ✅ Rest/fatigue
11. home_back_to_back               7.9  ✅ Rest/fatigue
12. away_star_missing               7.7  ✅ Injury impact
13. away_ewma_tov_pct               7.5  ✅ Four Factors
14. ewma_pace_diff                  7.5  ✅ Pace differential
15. home_rest_days                  7.2  ✅ Rest/fatigue
16. fatigue_mismatch                7.1  ✅ Rest/fatigue
17. injury_shock_diff               7.0  ✅ Injury impact
18. home_composite_elo              7.0  ✅ ELO ratings
19. home_orb                        6.7  ✅ Four Factors (Rebounding)
20. ewma_chaos_home                 6.7  ✅ Game volatility

## FEATURES BY CATEGORY

### Four Factors (7 features) - Core statistical edge
- ewma_efg_diff (11.9)
- ewma_tov_diff (8.2)
- away_ewma_tov_pct (7.5)
- home_orb (6.7)
- ewma_orb_diff (6.6)
- away_orb (6.4)
- away_ewma_3p_pct (6.0)

### ELO Ratings (4 features) - Team strength
- off_elo_diff (9.3)
- def_elo_diff (8.6)
- home_composite_elo (7.0)

### Rest & Fatigue (8 features) - Scheduling edge
- away_back_to_back (9.2)
- away_rest_days (8.3)
- away_3in4 (7.9)
- home_back_to_back (7.9)
- home_rest_days (7.2)
- fatigue_mismatch (7.1)
- rest_advantage (6.7)
- home_3in4 (5.8)

### Injury Impact (7 features) - Roster changes
- away_star_missing (7.7)
- injury_shock_diff (7.0)
- injury_shock_home (6.7)
- injury_shock_away (6.6)
- injury_impact_abs (6.4)
- star_mismatch (6.2)
- home_star_missing (6.1)
- injury_impact_diff (6.0)

### Season Context (7 features) - CURRENTLY DEFAULTED TO 0 ⚠️
- season_year (15.8) 🔴 HIGHEST IMPORTANCE
- season_year_normalized (15.2) 🔴 2ND HIGHEST
- endgame_phase (6.5)
- season_progress (6.3)
- games_into_season (6.2)
- season_month (6.1)
- is_season_opener (missing from importance - likely unused)

### Pace & Style (6 features) - Game flow
- ewma_pace_diff (7.5)
- ewma_vol_3p_diff (6.5)
- ewma_net_chaos (6.4)
- home_ewma_3p_pct (6.6)

### Venue & Other (3 features)
- altitude_game (8.2)
- ewma_foul_synergy_home (6.5)
- total_foul_environment (6.3)

## KALSHI INTEGRATION PLAN

### Current State
- Using default -110 odds (50% probability after vig)
- No real market odds being fetched
- Multi-source odds service not available

### Required Integration

1. **Use Kalshi API Contract Prices**
   - Contract prices are YES/NO probabilities (0-100 cents)
   - Example: YES at 60 cents = 60% implied probability
   - Need to fetch live NBA moneyline markets

2. **API Endpoints Needed**
   - Get active NBA markets: /trade-api/v2/markets?series_ticker=KXNBA*
   - Get market orderbook: /trade-api/v2/markets/{market_ticker}/orderbook
   - Extract yes_bid, yes_ask, no_bid, no_ask prices

3. **Integration Points**
   - Replace multi_source_odds_service with KalshiClient
   - Convert contract prices to implied probabilities
   - Calculate edge: model_prob - market_prob
   - Apply split thresholds (1.0% fav / 15.0% dog)

4. **Key Code Changes**
   ```python
   # In nba_gui_dashboard_v2.py, lines 240-250
   from src.services._OLD_kalshi_client import KalshiClient
   
   kalshi = KalshiClient(api_key, api_secret, environment='prod')
   markets = kalshi.get_nba_markets(home_team, away_team, game_date)
   
   # Get YES contract price (home team win)
   yes_price = markets['yes_ask']  # What we pay to buy YES
   no_price = markets['no_bid']    # What we get for NO
   
   # Convert to probabilities
   market_home_prob = yes_price / 100  # Cents to decimal
   market_away_prob = 1 - market_home_prob
   ```

5. **Authentication Setup**
   - API key and secret stored in config.json
   - RSA-PSS signature authentication (already implemented)
   - Rate limiting: 10 requests/second

## ACTION ITEMS

⚠️ **CRITICAL MISSING FEATURES**
- Season context features (15.8 + 15.2 importance) are currently set to 0
- This significantly impacts prediction accuracy
- Need to add season_year, season_year_normalized to feature_calculator_v5.py

✅ **Kalshi Integration**
- Activate _OLD_kalshi_client.py (rename/move to active services)
- Add credentials to config.json
- Replace odds_data logic in predict_game() method
- Test with live markets before December 15, 2025 games

📊 **Recommendation**
Focus on activating Kalshi API first - it will provide real market odds
for edge calculation. The season features can be added after to improve
model accuracy further.
"""

# Save to file
with open('FEATURES_AND_KALSHI_PLAN.txt', 'w', encoding='utf-8') as f:
    f.write(__doc__)

print(__doc__)
