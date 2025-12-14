# Quick Reference: System Enhancements

## New Modules Overview

### 1. ConfigManager - Configuration with .env Support
```python
from config_manager import get_config_manager

config = get_config_manager()
kelly = config.get('kelly_fraction')
api_keys = config.get_api_keys()
is_valid = config.validate_api_key_format('odds_api_key', key_value)
```

**Features:** .env loading, API key validation, single warning, fallback chain

---

### 2. RobustScraper - Enhanced Web Scraping
```python
from scraping_utils import RobustScraper, ScraperConfig

config = ScraperConfig(max_retries=3, timeout=15)
scraper = RobustScraper(config)
soup = scraper.fetch_and_parse(url, validate_structure={'table': 1})
```

**Features:** User agent rotation (10 agents), exponential backoff, HTML structure validation

---

### 3. Resilience - Circuit Breakers & Health Checks
```python
from resilience import get_circuit_breaker, with_retry, get_health_checker

# Circuit breaker
cb = get_circuit_breaker('odds_api', failure_threshold=5)
result = cb.call(api_function, *args)

# Retry decorator
@with_retry(max_attempts=3, exponential_base=2.0)
def fetch_data():
    pass

# Health monitoring
health = get_health_checker()
health.register_component('database', check_func, critical=True)
status = health.get_status()
```

**Features:** Circuit breakers, retry with backoff, health checks, graceful degradation

---

### 4. Security - Encryption & Validation
```python
from security import get_key_manager, get_rate_limiter, InputValidator

# API key encryption
key_mgr = get_key_manager()
key_mgr.store_api_key('odds_api', 'secret_key')
api_key = key_mgr.get_api_key('odds_api')

# Rate limiting
limiter = get_rate_limiter()
limiter.wait_if_needed('odds_api')

# Input validation
validator = InputValidator()
validator.validate_team('ATL')
validator.validate_date('2024-01-15')
```

**Features:** Fernet encryption, token bucket rate limiting, SQL injection prevention

---

### 5. FeatureCache - Performance Optimization
```python
from feature_cache import get_feature_cache

cache = get_feature_cache()

# Single game
cache.put('2024-25', '2024-01-15', 'LAL', 'BOS', features)
features = cache.get('2024-25', '2024-01-15', 'LAL', 'BOS')

# Batch operations
games = [('2024-01-15', 'LAL', 'BOS'), ('2024-01-15', 'GSW', 'PHX')]
results = cache.batch_get('2024-25', games)
cache.batch_put('2024-25', games, features_list)

# Statistics
stats = cache.get_stats()
```

**Features:** In-memory + disk cache, TTL (24h), LRU eviction, batch ops

---

### 6. AsyncDataFetcher - Background Tasks
```python
from async_data_fetcher import get_async_fetcher

fetcher = get_async_fetcher()

# Submit task
task_id = fetcher.submit_injury_fetch(
    game_date='2024-01-15',
    progress_callback=lambda p: print(f"{p}%")
)

# Poll status
status = fetcher.get_task_status(task_id)
if status['status'] == 'COMPLETED':
    result = status['result']

# Or wait
result = fetcher.wait_for_task(task_id, timeout=30.0)
```

**Features:** Thread pool (4 workers), progress tracking, task cancellation

---

### 7. WarmStartTrainer - Incremental Training
```python
from warm_start_trainer import WarmStartTrainer

trainer = WarmStartTrainer()

# Incremental update
trainer.partial_fit('ats', X_new, y_new)

# Rolling window (last 365 days)
trainer.rolling_window_retrain('ats', window_days=365)

# Scheduled retrain
trainer.schedule_retrain('ats', frequency_days=7, min_new_games=100)

# Rollback
trainer.rollback_model('ats', version='20241115_123045')
```

**Features:** XGBoost warm start, versioning, rollback, scheduled retrain

---

## Integration Checklist

### Dashboard Integration
- [ ] Initialize `get_feature_cache()` in `__init__`
- [ ] Use `batch_get`/`batch_put` in `_load_predictions_for_date`
- [ ] Initialize `get_async_fetcher()` for background tasks
- [ ] Add progress bars for injury/odds fetching
- [ ] Replace `sqlite3.connect()` with `SafeDBConnection`
- [ ] Validate inputs with `InputValidator`

### API Client Integration
- [ ] Add `get_rate_limiter().wait_if_needed()` before API calls
- [ ] Wrap API calls in circuit breakers
- [ ] Migrate API keys to encrypted storage
- [ ] Update clients to use `get_key_manager().get_api_key()`

### Model Training Integration
- [ ] Add scheduled warm start retraining
- [ ] Track model metrics per version
- [ ] Implement auto-rollback on degradation

---

## Configuration Files

### .env (create in project root)
```env
ODDS_API_KEY=your_32_to_64_char_key_here
KALSHI_API_KEY=your_16_to_128_char_key
KALSHI_API_SECRET=your_32_to_256_char_secret
```

### config.json (existing, now supports .env override)
```json
{
  "kelly_fraction": 0.02,
  "max_bet_size": 500,
  "model_retrain_frequency": 7,
  "odds_api_key": "",
  "kalshi_api_key": "",
  "kalshi_api_secret": ""
}
```

**Priority:** .env > environment variables > config.json > defaults

---

## Testing

### Quick Module Tests
```bash
python config_manager.py       # Config validation
python scraping_utils.py       # Scraping & retry
python resilience.py           # Circuit breakers
python security.py             # Encryption & validation
python feature_cache.py        # Cache operations
python async_data_fetcher.py   # Async tasks
python warm_start_trainer.py   # Training
```

### Integration Test
```bash
python enhanced_system_demo.py
```

---

## Performance Metrics

### Cache Hit Rates (Target >80%)
```python
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

### Rate Limit Violations (Target: 0)
```python
limiter = get_rate_limiter()
# Check logs for violations
```

### Circuit Breaker Health
```python
cb = get_circuit_breaker('service')
print(f"State: {cb.state}, Failures: {cb.failure_count}")
```

---

## Troubleshooting

**"python-dotenv not installed"**
```bash
pip install python-dotenv
```

**"API key too short/invalid format"**
- Check key meets minimum length requirements
- Verify no extra whitespace or newlines
- See API_KEY_PATTERNS in config_manager.py

**"Circuit breaker is open"**
```python
cb = get_circuit_breaker('service')
cb.reset()  # Manual reset
```

**"Scraping structure validation failed"**
- Website structure changed
- Update validate_structure selectors
- Check CBS/ESPN URLs still valid

**Cache not persisting**
- Ensure `cache/` directory exists
- Check disk space
- Verify write permissions

---

## Security Best Practices

1. **Use .env for secrets** (never commit to git)
2. **Add to .gitignore:** `.env`, `secrets/`, `cache/`, `*.enc`
3. **Rotate cipher key** quarterly
4. **Monitor rate limits** for abuse
5. **Regular security audits** of validation rules
6. **HTTPS only** for external APIs

---

## Next Steps

### Immediate (Phase 1)
1. Integrate feature cache into dashboard
2. Add async fetcher with progress bars
3. Replace direct DB queries with SafeDBConnection

### Short-term (Phase 2)
1. Migrate API keys to encrypted storage
2. Add rate limiting to all API clients
3. Implement circuit breakers around external calls

### Medium-term (Phase 3)
1. Schedule warm start retraining (nightly)
2. Add model performance tracking
3. Implement health monitoring dashboard

---

## Files Summary

| Module | Lines | Purpose |
|--------|-------|---------|
| config_manager.py | 280 | Configuration with .env |
| scraping_utils.py | 390 | Robust web scraping |
| resilience.py | 400 | Circuit breakers, health |
| security.py | 470 | Encryption, validation |
| feature_cache.py | 340 | Performance caching |
| async_data_fetcher.py | 370 | Background tasks |
| warm_start_trainer.py | 330 | Incremental training |
| **TOTAL** | **~2,580** | **Infrastructure code** |

**Modified Files:**
- injury_data_collector_v2.py (integrated RobustScraper)
- main.py (integrated ConfigManager)

---

## Support

For detailed documentation, see:
- `SYSTEM_ENHANCEMENTS_SUMMARY.md` - Complete feature documentation
- `enhanced_system_demo.py` - Integration examples
- Individual module docstrings - API documentation

**All modules include test code in `if __name__ == "__main__"` blocks.**
