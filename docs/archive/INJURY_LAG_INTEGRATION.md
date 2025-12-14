# Injury Lag Integration with Off/Def Elo System

## Overview
Injury impacts now feed into Elo expected point calculations, ensuring rating updates account for roster quality changes.

## Design

### Constants Added (constants.py)
```python
INJURY_OFF_SHARE: float = 0.6      # 60% of injury impact affects offensive output
INJURY_DEF_SHARE: float = 0.4      # 40% affects defensive quality
INJURY_ELO_SCALER: float = 0.35    # Damping factor to avoid Elo volatility
```

### Modified: OffDefEloSystem.update_game()

**New Parameters:**
- `home_injury_impact: float` - Total replacement-level point impact from home injuries
- `away_injury_impact: float` - Total replacement-level point impact from away injuries

**Sign Convention:**
- Negative values = team weakened by injuries
- Positive values = team strengthened (rare, rotation upgrades)

**Mechanism:**
1. Convert injury point impacts into Elo deltas:
   - `off_delta = injury_impact × INJURY_OFF_SHARE × INJURY_ELO_SCALER`
   - `def_delta = injury_impact × INJURY_DEF_SHARE × INJURY_ELO_SCALER`

2. Compute effective Elo for expectation:
   - `eff_off_elo = off_elo + off_delta`
   - `eff_def_elo = def_elo + def_delta`

3. Calculate expected points using effective Elos:
   - `exp_home_pts = LEAGUE_AVG + (home_eff_off - away_eff_def) / SCALE`
   - `exp_away_pts = LEAGUE_AVG + (away_eff_off - home_eff_def) / SCALE`

4. Update base Elos using actual vs expected errors (injury context baked into expectation)

## Integration Points

### feature_calculator_v5.py
Already surfaces `injury_home_total`, `injury_away_total` from `injury_replacement_model`.

**Next Step:** Pass these to `OffDefEloSystem.update_game()` when processing historical results.

### Example Usage
```python
from off_def_elo_system import OffDefEloSystem
from injury_replacement_model import calculate_injury_impact_differential_advanced

elo_sys = OffDefEloSystem('nba_betting.db')

# Get injury impacts
diff, home_inj, away_inj = calculate_injury_impact_differential_advanced(
    'LAL', 'BOS', game_date, breakdown=True
)

# Update Elo with injury context
elo_sys.update_game(
    season='2024-25',
    game_date='2024-11-19',
    home_team='LAL',
    away_team='BOS',
    home_points=108,
    away_points=105,
    is_playoffs=False,
    home_injury_impact=home_inj,  # e.g., -8.5 (LeBron out)
    away_injury_impact=away_inj   # e.g., -2.1 (minor absence)
)
```

## Benefits
1. **Contextualized Updates**: Elo adjusts expectations for diminished rosters, preventing over/under-reaction
2. **Chemistry Lag**: Already captured via `CHEMISTRY_LAG_FACTOR` in injury model, flows through impact values
3. **Positional Scarcity**: Scarcity multipliers from injury model amplify impact appropriately
4. **Volatility Control**: `INJURY_ELO_SCALER` prevents wild swings from single-game injury variance

## Testing
Run `off_def_elo_system.py` standalone to verify injury parameters accepted without errors:
```python
python off_def_elo_system.py
```

## Future Enhancements
- Automatic fetching of injury impacts during batch Elo recalculation
- Diagnostics dashboard overlay showing injury-adjusted vs raw Elo trajectories
- Backtest comparison: Elo calibration with/without injury context
