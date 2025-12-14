# Critical Fixes Applied - Crash Prevention & Data Download

## Issues Fixed

### 1. ✅ Injury Scraper Crash - FIXED
**Problem:** "Scrape injuries" button crashes the entire app  
**Cause:** Unhandled exceptions when CBS Sports/ESPN change their HTML structure or network fails

**Fix Applied:**
- Wrapped `_task_scrape_injuries()` with comprehensive try-except blocks
- Handles AttributeError (missing methods)
- Handles network errors (timeout, connection refused, DNS)
- Handles parsing errors (BeautifulSoup failures)
- Shows user-friendly error message instead of crashing
- Logs full traceback to console for debugging

**Result:** App will no longer crash - shows error message and continues running

---

### 2. ✅ PBP Data Download - FIXED  
**Problem:** Only 2,441 PBP events (need 50,000+ for backtesting)  
**Cause:** Download logic only ran when `pbp_count == 0`, but you already had 2,441 events from 5 games

**Fixes Applied:**

#### A. Updated Download Logic
Changed threshold from `pbp_count == 0` to `pbp_count < 50000`:
```python
# OLD: need_pbp = pbp_count == 0  
# NEW: need_pbp = pbp_count < 50000
```

Now downloads will continue until you have ~50,000 events (300+ games across 3 seasons)

#### B. Created Emergency Download Script
**File:** `download_pbp_emergency.py`

**Features:**
- Standalone script - runs outside dashboard (safer)
- Shows current database status
- Downloads 3 seasons: 2023-24, 2022-23, 2021-22
- Resume capability - skips already-downloaded games
- Saves progress every 50 games
- Estimated time: 30-60 minutes total
- Can be interrupted and resumed

**Usage:**
```powershell
python download_pbp_emergency.py
```

**Expected Output:**
```
Current Database:
   PBP Events: 2,441
   Unique Games: 5

⚠️  Need more data for backtesting:
   Target: 50,000 events, 300 games
   Need: 47,559 more events

DOWNLOADING PBP DATA FOR 3 SEASONS
This will take 30-60 minutes...

SEASON 1/3: 2023-24
   📥 Downloading PBP data for 2023-24...
   ✅ Season 2023-24 complete!
   Time: 18.3 minutes
   Gained: 18,234 events from 98 games
...
```

---

### 3. ✅ Backtest Crash - IMPROVED DIAGNOSTICS
**Problem:** Backtest crashes app without clear error messages  
**Cause:** Subprocess errors not captured properly, insufficient data not detected early

**Fixes Applied:**

#### A. Better Error Diagnostics
- Captures both stdout and stderr from subprocess
- Shows first 500 chars of error for quick diagnosis
- Logs full output to console for detailed debugging
- Added common fixes checklist:
  1. Download more PBP data (need 50,000+ events)
  2. Check console for full traceback
  3. Verify dependencies installed

#### B. Enhanced Subprocess Execution
- Added working directory (`cwd=os.getcwd()`)
- Pass environment variables (`env=os.environ.copy()`)
- Better timeout handling (5 minutes)
- Shows Python executable path for debugging

#### C. Improved Data Validation
Changed PBP requirement check:
```python
# Now warns and continues download if insufficient data
if pbp_count > 0 and pbp_count < 50000:
    logging.warning(f"Only {pbp_count:,} PBP events (need 50,000+)")
    logging.warning("Continuing download to complete dataset...")
```

---

## How to Use

### Immediate Action: Download PBP Data

**Option 1: Emergency Script (Recommended)**
```powershell
# Activate virtual environment
& "C:/Users/d76do/OneDrive/Documents/New Basketball Model/.venv/Scripts/Activate.ps1"

# Run download script
python download_pbp_emergency.py
```

This will:
- Check current status (2,441 events)
- Download ~47,000 more events
- Take 30-60 minutes
- Save progress every 50 games
- Can be interrupted and resumed

**Option 2: Dashboard Download**
1. Open NBA Dashboard
2. Settings tab
3. Click "1️⃣ Download Historical Data"
4. Wait for completion (same 30-60 minutes)

### After PBP Download Complete

**Step 1: Verify Data**
```powershell
python -c "import sqlite3; conn = sqlite3.connect('nba_betting_data.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM pbp_logs'); print(f'PBP events: {cursor.fetchone()[0]:,}'); cursor.execute('SELECT COUNT(DISTINCT game_id) FROM pbp_logs'); print(f'Unique games: {cursor.fetchone()[0]}'); conn.close()"
```

Expected output:
```
PBP events: 50,000+ (or more)
Unique games: 300+
```

**Step 2: Test Injury Scraper**
1. Open dashboard
2. Settings tab
3. Click "2️⃣ Scrape Current Injuries"
4. Should show success message (not crash!)
5. Check console for "✅ Scraped X injury reports"

**Step 3: Run Backtest**
1. Click "3️⃣ Train ML Models (with Backtest)"
2. Wait 2-3 minutes
3. Should complete with Brier score (0.10-0.20 expected)
4. If fails, check console for detailed error message

---

## Kalshi API Credentials

Your Kalshi credentials should be saved in Settings. To verify:

### Check Settings Tab
1. Open dashboard → Settings
2. Look for "Kalshi API Key" and "Kalshi API Secret"
3. Should show masked values if saved

### Check Config File
```powershell
cat config.json | Select-String kalshi
```

Should show:
```json
"kalshi_api_key": "your-key-here",
"kalshi_api_secret": "your-secret-here"
```

### Re-enter if Needed
If credentials are missing:
1. Settings tab
2. Enter API Key (16-128 alphanumeric)
3. Enter API Secret (PEM private key content)
4. Click "Save Settings"
5. Restart dashboard

### Test Kalshi Connection
```powershell
python test_kalshi_tomorrow.py
```

Expected output:
```
✅ Kalshi authentication successful
📊 Found 4 games with odds:
LAC @ ORL
  Total 213.0: Over 52¢ | Under 48¢
...
```

---

## Error Handling Summary

### Before These Fixes
- **Injury Scraper**: 💥 CRASH → App freezes, must force quit
- **PBP Download**: ⚠️ Silently skips (thinks it's done at 2,441 events)
- **Backtest**: 💥 CRASH → Vague "Unknown error" message

### After These Fixes
- **Injury Scraper**: ✅ Shows error message, app continues running
- **PBP Download**: ✅ Downloads until 50,000+ events, shows progress
- **Backtest**: ✅ Shows detailed error with fixes checklist, no crash

---

## Testing Checklist

### 1. Test Injury Scraper (Won't Crash Now)
```
✅ Click "Scrape Current Injuries"
✅ Wait 10-15 seconds
✅ Should show success or error message (not crash)
✅ Check console for "[INJURY] ✅ Scraped X injury reports"
```

### 2. Download PBP Data
```
✅ Run emergency script: python download_pbp_emergency.py
✅ Wait 30-60 minutes (can check progress)
✅ Verify final count: 50,000+ events
```

### 3. Test Backtest (Better Errors Now)
```
✅ After PBP download complete
✅ Click "Train ML Models (with Backtest)"
✅ Wait 2-3 minutes
✅ Should complete with Brier score
✅ If fails, console shows detailed error
```

### 4. Test Kalshi Odds
```
✅ Open Predictions tab
✅ Select tomorrow's date (Nov 20)
✅ Game cards should show Kalshi totals and moneylines
✅ Critical Info box shows Kelly wager and predicted score
```

---

## Troubleshooting

### Injury Scraper Still Fails
**Check Console Output:**
```
[INJURY] ❌ Scraper crashed: ...
[INJURY] Traceback: ...
```

**Common Causes:**
- Network timeout → Check internet connection
- Site structure changed → Wait for update or use backup source
- Missing dependencies → `pip install beautifulsoup4 lxml`

**Workaround:** Injuries are optional - predictions work without them

### PBP Download Stalls
**Symptoms:** Hangs at "Downloading game X/Y..."

**Solution:**
1. Press Ctrl+C to interrupt
2. Check database: `SELECT COUNT(*) FROM pbp_logs`
3. Run script again - it will resume from where it stopped
4. Progress is saved every 50 games

### Backtest Still Crashes
**Get Full Error:**
1. Check console for "[BACKTEST] === FULL SUBPROCESS OUTPUT ==="
2. Look for Python traceback
3. Common issues:
   - ImportError → Missing dependency
   - KeyError → Database schema mismatch
   - MemoryError → Close other apps

**Post Error to Console**

---

## Files Modified

1. **NBA_Dashboard_Enhanced_v5.py**:
   - `_task_scrape_injuries()`: Added comprehensive error handling (lines 4906-4930)
   - `_task_download_data()`: Changed PBP threshold to 50,000 (line 4823)
   - `_task_train_models()`: Enhanced subprocess error capture (lines 4960-5020)

2. **download_pbp_emergency.py**: NEW standalone script for safe PBP downloading

3. **test_kalshi_tomorrow.py**: Already created for Kalshi testing

---

**Next Step:** Run `python download_pbp_emergency.py` to get the 50,000+ PBP events needed for backtesting!
