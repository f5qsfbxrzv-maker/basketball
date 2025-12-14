# Live Monitor Implementation - Our Engine's Win Probability

## Summary
Successfully integrated our live win probability engine into the ESPN monitor, enabling real-time display of our engine's odds alongside ESPN's during live games.

## Changes Made

### 1. Dashboard Table Enhancement
**File:** `NBA_Dashboard_Enhanced_v5.py`

**New Columns Added:**
- Column 4: **Our WP (Home)** - Win probability from our engine (color-coded)
- Column 5: **Our Odds** - Implied moneyline odds from our probability
- Columns 6-7: ESPN WP and ESPN Odds (shifted from 4-5)
- Column 12: Last Play (shifted from 10)

**Table Header (13 columns):**
```
Game | Status | Score | Clock | Our WP (Home) | Our Odds | ESPN WP (Home) | ESPN Odds | Total Line | Yes¢/No¢ | Over % | Under % | Last Play
```

### 2. Helper Function Added
**Function:** `_parse_game_clock_to_seconds(game_clock, period)`

**Purpose:** Convert game clock (MM:SS format) and period to total seconds remaining

**Algorithm:**
- Parse clock string to extract minutes and seconds
- For regulation (Q1-Q4): `remaining = (4 - period) * 720 + period_seconds`
- For overtime (Q5+): `remaining = period_seconds`
- Returns 1440 seconds (half game) on parse failure

**Test Results:**
```
PASS Q1 12:00: 2880s (expected 2880s)
PASS Q2 6:00: 1800s (expected 1800s)
PASS Q4 0:30: 30s (expected 30s)
PASS Q5 5:00: 300s (expected 300s)
PASS Q4 0:00: 0s (expected 0s)
```

### 3. Live Win Probability Integration
**Method:** `_refresh_live_games()` (Lines 4073-4178)

**Integration Flow:**
1. Load `LiveWinProbabilityModel` on first refresh
2. For each live game:
   - Extract period and game_clock from NBA API
   - Convert clock to seconds remaining using helper function
   - Calculate score differential (home - away)
   - Build features dict: `{score_diff, time_remaining_seconds, possession}`
   - Call `live_wp_model.predict_probability(features)` → returns 0.0-1.0
   - Store `our_win_prob` in game data
3. Display in table with color coding:
   - Green background: >70% probability
   - Yellow background: 30-70% probability
   - Red background: <30% probability

**Odds Calculation:**
- Home favored (WP > 50%): `odds = -100 / (1/prob - 1)`
- Away favored (WP < 50%): `odds = +100 * (1/prob_away - 1)`
- Display format: "+150" or "-200"

### 4. Clock Display Enhancement
**Behavior:**
- Uses ESPN clock if available (more reliable during live games)
- Falls back to NBA API clock if ESPN data unavailable
- Format: "Q2 6:34"

## How to Use

### Starting the Monitor
1. Open the NBA Dashboard
2. Navigate to the "Live Odds" tab
3. Click **"Start ESPN Monitor"** button
4. Table refreshes every 15 seconds during live games

### Reading the Display
**Our Engine vs ESPN:**
- **Our WP**: Calculated using our Z-score model based on score differential and time remaining
- **Our Odds**: Implied moneyline from our probability (fair odds before vig)
- **ESPN WP**: ESPN's proprietary win probability model
- **ESPN Odds**: Implied odds from ESPN's probability

**Interpretation:**
- Compare our WP vs ESPN WP to identify edge opportunities
- If our WP > ESPN WP significantly, our engine favors home team more
- Look for discrepancies with market odds (Kalshi columns)

### Example Display
```
Game              | Status  | Score  | Clock    | Our WP | Our Odds | ESPN WP | ESPN Odds | Last Play
BOS @ LAL         | Live    | 98-95  | Q4 3:24  | 68.2%  | -214     | 72.5%   | -262      | Davis makes layup
PHX @ DEN         | Live    | 102-102| Q4 0:45  | 51.3%  | +103     | 49.8%   | +100      | Murray miss 3PT
```

## Technical Details

### Live Win Probability Model
**File:** `live_win_probability_model.py`

**Algorithm:** Z-Score Approach
```python
effective_lead = score_diff + (possession * POSSESSION_VALUE)
remaining_possessions = time_remaining_seconds / SECONDS_PER_POSSESSION
remaining_stdev = math.sqrt(remaining_possessions) * STDEV_PER_POSSESSION
z_score = effective_lead / remaining_stdev
win_prob = norm.cdf(z_score)
```

**Hyperparameters:**
- `POSSESSION_VALUE = 0.8` (expected points from possession)
- `STDEV_PER_POSSESSION = 1.2` (scoring variance)
- `SECONDS_PER_POSSESSION = 14.4` (average possession length)

**Assumptions:**
- Possession value set to 0 (not available from basic NBA API)
- Could be enhanced with ESPN possession data if available

### Data Sources
1. **NBA Stats API:** Live scores, period, game clock
2. **ESPN API:** Win probability, last play, possession (if available)
3. **Our Engine:** Live win probability from features

### Refresh Behavior
- **Interval:** 15 seconds (configurable via timer)
- **Auto-start:** Monitor starts automatically when live games detected
- **Background:** Non-blocking refresh, UI remains responsive

## Future Enhancements

### Short-Term
1. **Possession Data:** Integrate ESPN possession info to improve accuracy
2. **Historical Tracking:** Store our WP predictions for calibration analysis
3. **Alert System:** Notify when our WP diverges >10% from ESPN
4. **Edge Calculator:** Show edge% = (our_fair_prob - implied_prob_from_market)

### Medium-Term
1. **Situational Context:** Factor in timeouts remaining, fouls, bonus
2. **Momentum Indicators:** Recent scoring runs (last 2 min differential)
3. **Player Impact:** Adjust for who's on court (starters vs bench)
4. **Bayesian Updates:** Blend pre-game prediction with in-game model

### Long-Term
1. **ML-Based Live Model:** Train on play-by-play sequences
2. **Deep Learning:** LSTM/Transformer for sequence modeling
3. **Multi-Sport:** Extend to NFL, college basketball
4. **Betting Integration:** Direct connection to sportsbooks for line comparison

## Testing

### Unit Tests
- [x] Clock parser: All test cases pass (Q1-Q4, OT, edge cases)
- [ ] Win probability: Verify against historical game outcomes
- [ ] Odds conversion: Test edge cases (high/low probabilities)

### Integration Tests
- [x] Table display: 13 columns render correctly
- [x] Color coding: Green/yellow/red backgrounds apply properly
- [ ] Live refresh: Test during actual live games (pending game availability)
- [ ] ESPN data: Verify last play and WP extraction

### System Tests
- [ ] Full game monitoring: Track entire game from tipoff to final
- [ ] Calibration: Compare our WP vs actual outcomes over 50+ games
- [ ] Performance: Ensure 15s refresh doesn't cause lag
- [ ] Error handling: Gracefully handle API failures

## Known Issues

### Current Limitations
1. **No Possession Data:** Currently set to 0 (neutral), could improve with ESPN data
2. **No Player Context:** Doesn't account for who's on the court
3. **No Timeout Tracking:** Doesn't factor in timeouts remaining
4. **Basic Model:** Z-score approach is simple, not ML-based

### Potential Bugs
1. **ESPN Data Extraction:** May fail if API response structure changes
2. **Clock Format Variations:** Some games may use different clock formats
3. **Overtime Handling:** Only tested for 5-minute OT (not double/triple OT)

### Monitoring Required
- Check if ESPN win probability is actually populating
- Verify our engine's probabilities are reasonable (40-60% range for close games)
- Watch for any crashes during live refresh

## Validation Plan

### When Live Games Available
1. **Visual Inspection:**
   - Verify our WP updates every 15 seconds
   - Check color coding matches probability ranges
   - Confirm last play displays correctly

2. **Comparison Analysis:**
   - Record our WP and ESPN WP for 10+ games
   - Calculate correlation between the two models
   - Identify systematic differences (our model more/less optimistic?)

3. **Calibration Check:**
   - For games where our WP = 70%, home should win ~70% of the time
   - Track predictions vs outcomes over 50+ games
   - Compute Brier score for our live model

## Documentation

### User Guide
See: **QUICK_START.md** (section on Live Monitoring)

### Developer Notes
- Helper function is instance method (has access to self.live_wp_model)
- Error handling catches clock parse failures gracefully
- Table column indices shifted: Kalshi now 8-11, last play is 12

### Code References
- **Live refresh:** Lines 4073-4243 in NBA_Dashboard_Enhanced_v5.py
- **Table setup:** Lines 4030-4050 in NBA_Dashboard_Enhanced_v5.py
- **Helper function:** Lines 4238-4273 in NBA_Dashboard_Enhanced_v5.py
- **WP model:** live_win_probability_model.py

---

**Status:** ✅ Implementation Complete, Pending Live Game Testing  
**Last Updated:** November 19, 2024  
**Next Steps:** Test during tomorrow's games, track calibration metrics
