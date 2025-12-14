# NBA Betting System Schema Documentation

## Core Tables

### bets
Unified bet tracking table (replaces logged_bets and bet_history)

| Column | Type | Description |
|--------|------|-------------|
| bet_id | INTEGER PK | Auto-increment primary key |
| timestamp | TEXT | ISO datetime of bet placement |
| game_id | TEXT | Unique game identifier (YYYY-MM-DD_AWAY_HOME) |
| game_date | TEXT | Game date (YYYY-MM-DD) |
| matchup | TEXT | Display string "AWAY @ HOME" |
| market_type | TEXT | Type: TOTAL, SPREAD, MONEYLINE, PROP |
| selection | TEXT | Bet selection (Over/Under, team, etc.) |
| market_price | REAL | Kalshi contract price 0.0-1.0 (converted from cents) |
| model_probability | REAL | Model's predicted probability 0.0-1.0 |
| edge | REAL | Model edge after commission (model_prob - effective_price) |
| kelly_fraction | REAL | Kelly criterion fraction |
| stake_amount | REAL | Dollar amount wagered |
| bankroll_before | REAL | Bankroll before bet |
| bankroll_after | REAL | Bankroll after bet settles |
| bookmaker | TEXT | Default 'Kalshi' |
| outcome | TEXT | PENDING, WON, LOST, PUSH, CANCELLED |
| profit_loss | REAL | Net P/L after commission |
| confidence | REAL | Optional confidence score |
| notes | TEXT | Additional metadata |
| created_at | TEXT | Row creation timestamp |
| updated_at | TEXT | Last update timestamp |

**Indexes:**
- `idx_bets_timestamp` on timestamp
- `idx_bets_outcome` on outcome
- `idx_bets_game_date` on game_date

---

### odds_snapshots
Historical odds price movements

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| timestamp | TEXT | Snapshot datetime |
| game_id | TEXT | Game identifier |
| line | REAL | Total line or spread value |
| yes_price | INTEGER | Kalshi YES price (cents 0-100) |
| no_price | INTEGER | Kalshi NO price (cents 0-100) |
| home_ml | INTEGER | Home moneyline (American odds) |
| away_ml | INTEGER | Away moneyline (American odds) |

---

### calibration_outcomes
Prediction tracking for calibration

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| game_id | TEXT | Game identifier |
| predicted_prob | REAL | Model probability (pre-calibration) |
| outcome | INTEGER | Actual result (0=Under/Loss, 1=Over/Win, NULL=pending) |
| features_snapshot | TEXT | JSON of raw features used |
| prediction_timestamp | TEXT | When prediction was made |
| outcome_timestamp | TEXT | When outcome was logged |

---

### calibration_models
Stored calibration parameters

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| model_type | TEXT | 'isotonic' or 'platt' |
| parameters | TEXT | JSON serialized model state |
| created_at | TEXT | Model fit timestamp |
| sample_count | INTEGER | Number of samples used |

---

### bankroll_history
Bankroll tracking ledger

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| timestamp | TEXT | Transaction datetime |
| bankroll | REAL | New bankroll amount |
| change | REAL | Delta from previous |
| reason | TEXT | Description (deposit, withdrawal, bet_settle) |
| bet_id | INTEGER | FK to bets.bet_id if applicable |

---

### active_injuries
Current injury status (refreshed periodically)

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| player_name | TEXT | Player full name |
| team | TEXT | Team abbreviation |
| status | TEXT | Out, Questionable, Doubtful, GTD |
| injury_detail | TEXT | Body part / description |
| last_updated | TEXT | Scrape timestamp |

---

### player_stats
Player performance metrics (including PIE)

| Column | Type | Description |
|--------|------|-------------|
| player_id | TEXT PK | NBA player ID |
| player_name | TEXT | Full name |
| team | TEXT | Current team |
| season | TEXT | Season year |
| pie | REAL | Player Impact Estimate |
| games_played | INTEGER | GP count |
| minutes_avg | REAL | MPG |
| ... | ... | (Additional stats as needed) |

---

## Naming Conventions

**Standardized Terms:**
- `market_price` (not `odds`) - REAL 0.0-1.0
- `stake_amount` (not `stake`) - REAL dollar amount
- `outcome` (not `result` or `status`) - TEXT enum
- `game_id` format: `YYYY-MM-DD_AWAY_HOME`
- Timestamps: ISO 8601 TEXT format
- Kalshi prices: stored as INTEGER cents (0-100) in odds_snapshots, converted to REAL (0.0-1.0) in bets table

**Commission Handling:**
- Buy commission: 0.02 (2%)
- Sell commission: 0.02 (2%)
- Expiry commission: 0.0 (0%)
- Effective price = market_price * (1 + buy_commission)
- Edge = model_probability - effective_price

---

## Migration Notes

Legacy tables renamed to `*_backup`:
- `logged_bets_backup` - original dashboard bet log
- `bet_history_backup` - original kelly_optimizer log

After verifying data integrity, drop backups with:
```sql
DROP TABLE logged_bets_backup;
DROP TABLE bet_history_backup;
```
