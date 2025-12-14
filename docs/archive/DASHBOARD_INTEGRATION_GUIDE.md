# Dashboard Integration Guide

## Overview

Step-by-step guide to integrate new infrastructure modules into `NBA_Dashboard_Enhanced_v5.py`.

---

## Phase 1: Feature Cache Integration

### Step 1: Add Imports
```python
# Add to imports section (around line 20)
from feature_cache import get_feature_cache
```

### Step 2: Initialize Cache in __init__
```python
# In NBA_Dashboard_Enhanced_v5.__init__ (around line 150)
def __init__(self, master):
    self.master = master
    # ... existing initialization ...
    
    # Initialize feature cache
    self.feature_cache = get_feature_cache()
    logger.info("Feature cache initialized")
```

### Step 3: Update _load_predictions_for_date
```python
def _load_predictions_for_date(self, date_str):
    """Load predictions with feature caching"""
    season = self._get_season_for_date(date_str)
    
    # Get games for date
    games = self._get_games_for_date(date_str)
    
    if not games:
        return []
    
    # Try batch cache retrieval
    game_tuples = [(date_str, g['home_team'], g['away_team']) for g in games]
    cached_features = self.feature_cache.batch_get(season, game_tuples)
    
    predictions = []
    features_to_cache = []
    
    for game in games:
        cache_key = (date_str, game['home_team'], game['away_team'])
        
        # Use cached features if available
        if cache_key in cached_features:
            features = cached_features[cache_key]
            logger.debug(f"Cache hit for {game['home_team']} vs {game['away_team']}")
        else:
            # Compute features
            features = self.feature_calculator.calculate_game_features(
                game['home_team'],
                game['away_team'],
                date_str
            )
            features_to_cache.append((cache_key, features))
            logger.debug(f"Cache miss for {game['home_team']} vs {game['away_team']}")
        
        # Generate prediction using features
        pred = self._make_prediction(game, features)
        predictions.append(pred)
    
    # Batch cache storage for misses
    if features_to_cache:
        cache_tuples = [ct[0] for ct in features_to_cache]
        cache_features = [ct[1] for ct in features_to_cache]
        self.feature_cache.batch_put(season, cache_tuples, cache_features)
    
    # Log cache statistics
    stats = self.feature_cache.get_stats()
    logger.info(f"Cache stats: {stats['hit_rate']:.1%} hit rate, "
                f"{stats['total_entries']} entries")
    
    return predictions
```

### Step 4: Add Cache Refresh Button
```python
# In create_control_panel (around line 400)
def create_control_panel(self):
    # ... existing controls ...
    
    # Cache control
    cache_frame = ttk.LabelFrame(self.control_panel, text="Cache")
    cache_frame.pack(fill='x', padx=5, pady=5)
    
    ttk.Button(
        cache_frame,
        text="Refresh Cache",
        command=self._refresh_cache
    ).pack(side='left', padx=5)
    
    self.cache_stats_label = ttk.Label(cache_frame, text="Cache: --")
    self.cache_stats_label.pack(side='left', padx=5)

def _refresh_cache(self):
    """Clear cache for current date"""
    current_date = self.date_var.get()
    self.feature_cache.invalidate_date(current_date)
    logger.info(f"Cache cleared for {current_date}")
    
    # Refresh display
    self._update_cache_stats()
    self._load_predictions_for_date(current_date)

def _update_cache_stats(self):
    """Update cache statistics display"""
    stats = self.feature_cache.get_stats()
    self.cache_stats_label.config(
        text=f"Cache: {stats['hit_rate']:.0%} hit rate, "
             f"{stats['total_entries']} entries"
    )
```

---

## Phase 2: Async Data Fetcher Integration

### Step 1: Add Imports
```python
from async_data_fetcher import get_async_fetcher
from tkinter import ttk
import threading
```

### Step 2: Initialize Async Fetcher
```python
def __init__(self, master):
    # ... existing initialization ...
    
    # Initialize async fetcher
    self.async_fetcher = get_async_fetcher()
    self.active_tasks = {}  # Track background tasks
    logger.info("Async data fetcher initialized")
```

### Step 3: Add Background Task UI
```python
def create_data_panel(self):
    """Create panel for data operations"""
    data_frame = ttk.LabelFrame(self.control_panel, text="Data Updates")
    data_frame.pack(fill='x', padx=5, pady=5)
    
    # Injury data fetch
    ttk.Button(
        data_frame,
        text="Fetch Injuries",
        command=self._fetch_injuries_async
    ).pack(side='left', padx=5)
    
    self.injury_progress = ttk.Progressbar(
        data_frame,
        mode='indeterminate',
        length=100
    )
    self.injury_progress.pack(side='left', padx=5)
    
    self.injury_status = ttk.Label(data_frame, text="Ready")
    self.injury_status.pack(side='left', padx=5)
    
    # Odds data fetch
    ttk.Button(
        data_frame,
        text="Fetch Odds",
        command=self._fetch_odds_async
    ).pack(side='left', padx=5)
    
    self.odds_progress = ttk.Progressbar(
        data_frame,
        mode='indeterminate',
        length=100
    )
    self.odds_progress.pack(side='left', padx=5)
    
    self.odds_status = ttk.Label(data_frame, text="Ready")
    self.odds_status.pack(side='left', padx=5)

def _fetch_injuries_async(self):
    """Fetch injury data in background"""
    current_date = self.date_var.get()
    
    # Start progress indicator
    self.injury_progress.start()
    self.injury_status.config(text="Fetching...")
    
    # Submit async task
    task_id = self.async_fetcher.submit_injury_fetch(
        game_date=current_date,
        progress_callback=self._on_injury_progress
    )
    
    self.active_tasks['injury'] = task_id
    
    # Poll for completion
    self._poll_task('injury', self._on_injury_complete)

def _on_injury_progress(self, progress):
    """Update injury fetch progress"""
    self.injury_status.config(text=f"{progress}%")

def _on_injury_complete(self, result):
    """Handle injury fetch completion"""
    self.injury_progress.stop()
    
    if result.get('error'):
        self.injury_status.config(text=f"Error: {result['error']}")
        logger.error(f"Injury fetch failed: {result['error']}")
    else:
        count = result.get('result', 0)
        self.injury_status.config(text=f"✓ {count} injuries")
        logger.info(f"Fetched {count} injuries")
        
        # Refresh predictions with new data
        self._refresh_cache()

def _poll_task(self, task_name, callback, interval=500):
    """Poll async task for completion"""
    task_id = self.active_tasks.get(task_name)
    
    if not task_id:
        return
    
    status = self.async_fetcher.get_task_status(task_id)
    
    if status['status'] in ['COMPLETED', 'FAILED']:
        callback(status)
        del self.active_tasks[task_name]
    else:
        # Schedule next poll
        self.master.after(interval, lambda: self._poll_task(task_name, callback))
```

---

## Phase 3: Security Integration

### Step 1: Add Imports
```python
from security import SafeDBConnection, InputValidator
```

### Step 2: Initialize Security Components
```python
def __init__(self, master):
    # ... existing initialization ...
    
    # Initialize input validator
    self.validator = InputValidator()
    logger.info("Input validator initialized")
```

### Step 3: Replace Database Connections
```python
# OLD:
def _get_games_for_date(self, date_str):
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM games WHERE game_date = ?", (date_str,))
    games = cursor.fetchall()
    conn.close()
    return games

# NEW:
def _get_games_for_date(self, date_str):
    # Validate input
    try:
        self.validator.validate_date(date_str)
    except ValueError as e:
        logger.error(f"Invalid date: {e}")
        return []
    
    # Use safe DB connection
    db = SafeDBConnection(self.db_path)
    games = db.execute_safe(
        "SELECT * FROM games WHERE game_date = ?",
        (date_str,)
    )
    return games
```

### Step 4: Validate User Inputs
```python
def _on_team_selected(self, event):
    """Handle team selection with validation"""
    team = self.team_var.get()
    
    try:
        self.validator.validate_team(team)
        self._load_team_data(team)
    except ValueError as e:
        messagebox.showerror("Invalid Team", str(e))
        logger.warning(f"Invalid team selected: {team}")
```

---

## Phase 4: Config Manager Integration

### Step 1: Add Imports
```python
from config_manager import get_config_manager
```

### Step 2: Load Configuration
```python
def __init__(self, master):
    # ... existing initialization ...
    
    # Load configuration via ConfigManager
    self.config_mgr = get_config_manager()
    self.config = self.config_mgr.config
    
    # Extract settings
    self.kelly_fraction = self.config.get('kelly_fraction', 0.02)
    self.max_bet_size = self.config.get('max_bet_size', 500)
    
    logger.info("Configuration loaded via ConfigManager")
```

---

## Phase 5: Resilience Integration

### Step 1: Add Imports
```python
from resilience import get_circuit_breaker, get_health_checker
```

### Step 2: Initialize Health Monitoring
```python
def __init__(self, master):
    # ... existing initialization ...
    
    # Initialize health checker
    self.health_checker = get_health_checker()
    
    # Register components
    self.health_checker.register_component(
        'database',
        self._check_database_health,
        critical=True
    )
    
    self.health_checker.register_component(
        'model_files',
        self._check_model_health,
        critical=True
    )
    
    logger.info("Health monitoring initialized")

def _check_database_health(self):
    """Check if database is accessible"""
    try:
        db = SafeDBConnection(self.db_path)
        db.execute_safe("SELECT 1", ())
        return True
    except Exception:
        return False

def _check_model_health(self):
    """Check if model files exist"""
    return all([
        Path(f'models/model_v5_{t}.xgb').exists()
        for t in ['ats', 'ml', 'total']
    ])
```

### Step 3: Add Health Status Display
```python
def create_status_bar(self):
    """Create status bar with health indicator"""
    status_frame = ttk.Frame(self.master)
    status_frame.pack(side='bottom', fill='x')
    
    self.health_indicator = ttk.Label(
        status_frame,
        text="● System Health: Checking...",
        foreground='orange'
    )
    self.health_indicator.pack(side='right', padx=10)
    
    # Periodic health check
    self._update_health_status()

def _update_health_status(self):
    """Update health status display"""
    self.health_checker.check_all()
    status = self.health_checker.get_status()
    
    if status['overall'] == 'healthy':
        self.health_indicator.config(
            text="● System Health: Healthy",
            foreground='green'
        )
    else:
        self.health_indicator.config(
            text="● System Health: Degraded",
            foreground='red'
        )
    
    # Schedule next check (every 60 seconds)
    self.master.after(60000, self._update_health_status)
```

---

## Testing Integration

### Test Checklist

**Feature Cache:**
- [ ] Cache initializes on startup
- [ ] First load computes features (cache miss)
- [ ] Second load uses cache (cache hit)
- [ ] Refresh button clears cache
- [ ] Statistics display updates

**Async Fetcher:**
- [ ] Injury fetch runs in background
- [ ] Progress bar animates
- [ ] UI remains responsive
- [ ] Completion callback triggers
- [ ] Error handling works

**Security:**
- [ ] Invalid team rejected
- [ ] Invalid date rejected
- [ ] SQL injection attempts blocked
- [ ] All queries parameterized

**Health Monitoring:**
- [ ] Health status displays
- [ ] Critical failure shows red
- [ ] All healthy shows green
- [ ] Updates every 60 seconds

### Manual Test Script
```python
# test_dashboard_integration.py
import tkinter as tk
from NBA_Dashboard_Enhanced_v5 import NBA_Dashboard_Enhanced_v5

# Create test window
root = tk.Tk()
dashboard = NBA_Dashboard_Enhanced_v5(root)

# Test feature cache
print("Testing feature cache...")
cache_stats = dashboard.feature_cache.get_stats()
print(f"Cache entries: {cache_stats['total_entries']}")

# Test async fetcher
print("Testing async fetcher...")
task_id = dashboard.async_fetcher.submit_injury_fetch(
    game_date='2024-01-15',
    progress_callback=lambda p: print(f"Progress: {p}%")
)
print(f"Task submitted: {task_id}")

# Test health checker
print("Testing health checker...")
status = dashboard.health_checker.get_status()
print(f"System health: {status['overall']}")

print("Integration tests complete!")
```

---

## Performance Benchmarks

### Before Integration
- Feature calculation: 150-200ms per game (cold)
- UI freeze during scraping: 5-10 seconds
- Full prediction load: 2-3 seconds (10 games)

### After Integration (Expected)
- Feature calculation: 20-30ms per game (cached)
- UI freeze during scraping: 0 seconds (async)
- Full prediction load: 0.5-1 second (cached)

**Target Improvement: 70-80% reduction in load time**

---

## Rollback Plan

If integration causes issues:

1. **Disable feature cache:**
   ```python
   # Comment out cache initialization
   # self.feature_cache = get_feature_cache()
   ```

2. **Disable async fetcher:**
   ```python
   # Revert to synchronous fetching
   # self._fetch_injuries_sync()
   ```

3. **Revert to old DB connection:**
   ```python
   # Use sqlite3.connect() directly
   ```

4. **Git revert:**
   ```bash
   git checkout NBA_Dashboard_Enhanced_v5.py
   ```

---

## Next Steps After Integration

1. **Monitor Performance:**
   - Track cache hit rates (target >80%)
   - Measure UI responsiveness
   - Log async task completion times

2. **Optimize:**
   - Tune cache TTL based on data update frequency
   - Adjust async worker pool size
   - Fine-tune rate limits

3. **Extend:**
   - Add more async operations (model training)
   - Implement dashboard health page
   - Add cache prewarming on startup

---

## Support

**Issues?** Check:
- Logs in `logs/` directory
- Module test scripts (run individually)
- `SYSTEM_ENHANCEMENTS_SUMMARY.md` for detailed docs

**Questions?** See:
- Individual module docstrings
- `QUICK_REFERENCE_ENHANCEMENTS.md` for API examples
