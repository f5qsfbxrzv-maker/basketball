# Manual Bet Entry - Dashboard Guide

## What Was Added

### New Features in Predictions Tab

The dashboard now includes a **Manual Bet Entry** section at the bottom of the Predictions tab with complete bet tracking functionality.

### Input Fields

1. **Game Selector** - Dropdown populated with available games from predictions table
2. **Bet Type** - Select from:
   - Moneyline
   - Spread
   - Total (Over/Under)

3. **Side** - Choose:
   - Home
   - Away
   - Over
   - Under

4. **Price (American Odds)** - Spin box for entering odds
   - Example: -110, +150, -200, +300
   - Range: -10000 to +10000
   - Default: -110

5. **Line/Total** - For spread and totals bets
   - Example: -5.5 (spread) or 215.5 (total points)
   - Range: -50 to 300
   - Increments: 0.5

6. **Stake Amount** - How much you're betting
   - Format: $10.00
   - Minimum: $0.01
   - Maximum: $100,000

7. **Bookmaker** - Name of the sportsbook
   - Text field (optional)
   - Examples: "DraftKings", "FanDuel", "BetMGM"

### Action Buttons

#### Calculate EV
- Calculates Expected Value based on:
  - Your entered price (American odds)
  - Implied probability from the odds
  - Estimated model probability (currently 10% edge for demo)
  - Shows EV in dollars and percentage
  
- Also calculates Kelly Criterion recommendation:
  - Optimal stake based on edge
  - Percentage of bankroll
  - Only recommends bet if positive edge

#### Place Bet
- Records the bet to the database (`placed_bets` table)
- Generates a unique bet ID
- Deducts stake from current bankroll
- Logs transaction to bankroll_history
- Updates bankroll display
- Clears form after successful placement
- Displays bet confirmation in console

#### Clear
- Resets all form fields to defaults
- Clears calculation results

### Output Display

**Expected Value Result**
- Shows calculated EV: `$2.50 (+25.0%)`
- Green text for positive EV
- Red text for negative EV

**Kelly Recommendation**
- Shows recommended stake: `$15.75 (1.58% of bankroll)`
- Only displays if positive edge exists
- Follows your Kelly fraction setting from System Admin tab

### Database Integration

All placed bets are saved to the SQLite database in the `placed_bets` table with:

| Column | Description |
|--------|-------------|
| `id` | Auto-incrementing bet ID |
| `timestamp` | When bet was placed |
| `game` | Game matchup (e.g., "Lakers @ Warriors") |
| `bet_type` | Moneyline, Spread, or Total |
| `side` | Home/Away/Over/Under |
| `price` | American odds |
| `line` | Spread or total line |
| `stake` | Amount wagered |
| `bookmaker` | Sportsbook name |
| `result` | Win/Loss/Push (default: 'pending') |
| `profit` | Profit/loss amount (default: 0) |

### Bankroll Integration

- **Automatic Updates**: Placing a bet deducts stake from current bankroll
- **Persistent Storage**: All bankroll changes saved to `bankroll_history` table
- **Transaction Log**: Every bet placement creates a bankroll transaction record
- **Live Display**: Bankroll display updates immediately after bet placement

### Console Logging

When you place a bet, the console shows:
```
✅ BET PLACED (ID: 42)
   Game: Lakers @ Warriors
   Type: Moneyline - Home
   Price: -110 | Line: 0.0
   Stake: $25.00
   Bookmaker: DraftKings
   Saved to database
   Bankroll updated: $1,000.00 → $975.00
```

## Usage Example

### Placing a Moneyline Bet

1. **Select Game**: "Lakers @ Warriors"
2. **Bet Type**: Moneyline
3. **Side**: Home (Warriors)
4. **Price**: -150
5. **Stake**: $30.00
6. **Bookmaker**: "FanDuel"
7. Click **Calculate EV** to see expected value
8. Review Kelly recommendation
9. Click **Place Bet** to record

### Placing a Spread Bet

1. **Select Game**: "Celtics @ Heat"
2. **Bet Type**: Spread
3. **Side**: Away (Celtics)
4. **Price**: -110
5. **Line**: -5.5
6. **Stake**: $20.00
7. **Bookmaker**: "DraftKings"
8. Click **Calculate EV**
9. Click **Place Bet**

### Placing a Total Bet

1. **Select Game**: "Nuggets @ Suns"
2. **Bet Type**: Total (Over/Under)
3. **Side**: Over
4. **Price**: -105
5. **Line**: 225.5
6. **Stake**: $50.00
7. **Bookmaker**: "BetMGM"
8. Click **Calculate EV**
9. Click **Place Bet**

## Integration Points

### With Predictions Table
- Game selector automatically populated when predictions refresh
- Can quickly select a game from the table above

### With Kelly Optimizer
- Uses your configured Kelly fraction from System Admin tab
- Respects your minimum bet size settings
- Uses current bankroll for Kelly calculations
- Updates bankroll after bet placement

### With Database
- All bets stored for analysis
- Bankroll history maintains complete transaction log
- Can query `placed_bets` table for bet history
- Can update bet results manually in database

## Testing

Run the dashboard to test:
```bash
python main.py
```

Navigate to the **Predictions** tab and scroll down to see the **Manual Bet Entry** section.
