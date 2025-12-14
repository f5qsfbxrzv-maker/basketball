# Predictions Page Enhancements

## Overview
The predictions page has been completely redesigned with professional styling, team branding, and comprehensive statistics.

## New Features

### 1. Team Symbols & Branding
- **Team Emojis**: Each team displays a unique symbol (e.g., 👑 Lakers, ☘️ Celtics, 🔥 Heat)
- **Team Colors**: Primary team colors used for headers matching official branding
- **Consistent Styling**: Professional color scheme throughout

### 2. Predicted Scores
- **Home Score Prediction**: Model-predicted final score for home team
- **Away Score Prediction**: Model-predicted final score for away team
- **Total Prediction**: Combined predicted total displayed
- **Spread Display**: Shows point differential and favored team

### 3. Enhanced Edge Display
- **Win Percentage**: Shows model's win probability for each team
- **Point Differential**: Displays predicted margin of victory
- **Edge Percentage**: Color-coded edge display:
  - 🟢 Green: Edge > 3% (strong betting opportunity)
  - 🔴 Red: Edge < -3% (avoid)
  - ⚪ Gray: Edge between -3% and +3% (neutral)

### 4. Relevant Statistics

#### Last 10 Record (L10)
- Shows each team's record from their last 10 games
- Format: "7-3" (7 wins, 3 losses)
- Helps identify teams on hot/cold streaks

#### Head-to-Head Record (H2H)
- Shows season record between these two teams
- Format: "2-1" (away team perspective)
- Identifies matchup advantages

#### Injuries
- **Injury Count Badge**: Shows total injured players
- **Player Names**: Expandable details show who's out
- **Status**: "Out" or "Questionable" designation
- **Impact**: Each injury reduces win probability ~2%

### 5. Professional Synopsis Layout

```
┌─────────────────────────────────────────────────────────┐
│  🔥 MIA               @              👑 LAL             │
│  Predicted: 108               H2H: 1-1    Predicted: 112 │
│  L10: 6-4                                    L10: 8-2    │
├─────────────────────────────────────────────────────────┤
│  47.3%            Spread: 4.0 pt LAL            52.7%   │
├─────────────────────────────────────────────────────────┤
│  Edge: -2.1%      🏥 2 out      ▼ Details    Edge: +2.1% │
└─────────────────────────────────────────────────────────┘
```

## Team Symbol Reference

| Team | Symbol | Color | Full Name |
|------|--------|-------|-----------|
| ATL  | 🦅     | #E03A3E | Atlanta Hawks |
| BOS  | ☘️     | #007A33 | Boston Celtics |
| BKN  | 🕸️     | #000000 | Brooklyn Nets |
| CHA  | 🐝     | #1D1160 | Charlotte Hornets |
| CHI  | 🐂     | #CE1141 | Chicago Bulls |
| CLE  | ⚔️     | #860038 | Cleveland Cavaliers |
| DAL  | 🐴     | #00538C | Dallas Mavericks |
| DEN  | ⛰️     | #0E2240 | Denver Nuggets |
| DET  | 🚗     | #C8102E | Detroit Pistons |
| GSW  | 🌉     | #1D428A | Golden State Warriors |
| HOU  | 🚀     | #CE1141 | Houston Rockets |
| IND  | 🏎️     | #002D62 | Indiana Pacers |
| LAC  | ⛵     | #C8102E | LA Clippers |
| LAL  | 👑     | #552583 | Los Angeles Lakers |
| MEM  | 🐻     | #5D76A9 | Memphis Grizzlies |
| MIA  | 🔥     | #98002E | Miami Heat |
| MIL  | 🦌     | #00471B | Milwaukee Bucks |
| MIN  | 🐺     | #0C2340 | Minnesota Timberwolves |
| NOP  | ⚜️     | #0C2340 | New Orleans Pelicans |
| NYK  | 🗽     | #006BB6 | New York Knicks |
| OKC  | ⚡     | #007AC1 | Oklahoma City Thunder |
| ORL  | 🪄     | #0077C0 | Orlando Magic |
| PHI  | 🔔     | #006BB6 | Philadelphia 76ers |
| PHX  | ☀️     | #1D1160 | Phoenix Suns |
| POR  | 🌲     | #E03A3E | Portland Trail Blazers |
| SAC  | 👑     | #5A2D81 | Sacramento Kings |
| SAS  | ⭐     | #C4CED4 | San Antonio Spurs |
| TOR  | 🦖     | #CE1141 | Toronto Raptors |
| UTA  | 🎵     | #002B5C | Utah Jazz |
| WAS  | 🎩     | #002B5C | Washington Wizards |

## Model Calculations

### Win Probability (Multi-Factor Model)
1. **ELO Ratings (40%)**: Composite + Off/Def ELO with home court advantage
2. **Net Rating (30%)**: Point differential with +3 home advantage
3. **Win Percentage (20%)**: Season record with home boost
4. **Injuries (10%)**: Deduction for each injured player (~2% per player)

### Predicted Scores
- **Home Score**: `(home_off_rating * 0.6 + league_avg_adj * 0.4) * pace / 100`
- **Away Score**: `(away_off_rating * 0.6 + league_avg_adj * 0.4) * pace / 100`
- **Total**: Sum of predicted scores
- **Spread**: Home score - Away score

### Edge Calculation
- **Fair Probability**: Remove vig from market prices
- **Edge**: Model probability - Fair market probability
- **Color Coding**:
  - Green (>3%): Strong betting opportunity
  - Red (<-3%): Market heavily favors opposite side
  - Gray (-3% to +3%): Neutral, no clear edge

## Files Modified

1. **nba_team_data.py** (NEW)
   - Team symbols dictionary
   - Team colors dictionary
   - Team full names mapping
   - Helper functions for lookups

2. **NBA_Dashboard_Enhanced_v5.py**
   - Imported team data module
   - Added `_get_last_n_record()` method
   - Added `_get_head_to_head()` method
   - Completely redesigned synopsis section
   - Enhanced visual styling with gradients

## Usage

The enhanced predictions page automatically displays when you:
1. Launch the dashboard
2. Navigate to "🎯 Predictions" tab
3. Select a date with games

All statistics are computed automatically from the database. No manual configuration needed.

## Benefits

✅ **Quick Visual Assessment**: Instantly identify betting opportunities with color-coded edges
✅ **Context**: Recent form (L10) and matchup history (H2H) inform decisions
✅ **Transparency**: See exactly what the model predicts for scores and outcomes
✅ **Professional**: Clean, branded interface matches sports betting standards
✅ **Information Dense**: All key metrics visible without scrolling
