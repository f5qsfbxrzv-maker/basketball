# The Odds API - Complete Setup Guide

## 🏀 What is The Odds API?

The Odds API provides real-time sports betting odds from multiple bookmakers worldwide. For NBA betting, it gives you:

- **Live Odds**: Moneyline, point spreads, totals (over/under)
- **Multiple Bookmakers**: DraftKings, FanDuel, BetMGM, Caesars, and more
- **Real-time Updates**: Odds changes as they happen
- **Historical Data**: Line movement tracking
- **Arbitrage Detection**: Compare odds across books

## 💰 Pricing & Plans

### Free Tier
- **500 requests/month** 
- **Perfect for testing** your NBA system
- **No credit card required**
- **Access to all markets**

### Paid Plans
- **$10/month**: 10,000 requests
- **$25/month**: 50,000 requests  
- **$100/month**: 250,000 requests
- **Enterprise**: Custom pricing

## 🚀 How to Get Your API Key

### Step 1: Sign Up
1. Go to: **https://the-odds-api.com/**
2. Click **"Get API Key"**
3. Sign up with email/password
4. Verify your email

### Step 2: Get Your Key
1. Log into your dashboard
2. Copy your API key (starts with a long string of letters/numbers)
3. Keep it secure - treat it like a password

### Step 3: Test Your Key
```bash
# Test your key works (replace YOUR_KEY with actual key)
curl "https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey=YOUR_KEY&regions=us&markets=h2h"
```

## ⚙️ Integration with Your NBA System

### Add to Configuration
Edit your `config.json`:
```json
{
  "odds_api_key": "your_actual_api_key_here",
  "environment": "demo",
  "paper_trading": true
}
```

### Test Integration
```bash
python api_test.py
```

## 📊 What Data You'll Get

### NBA Game Example
```json
{
  "id": "nba_game_12345",
  "sport_title": "NBA",
  "commence_time": "2025-11-18T00:30:00Z",
  "home_team": "Los Angeles Lakers",
  "away_team": "Boston Celtics",
  "bookmakers": [
    {
      "key": "draftkings",
      "title": "DraftKings",
      "markets": [
        {
          "key": "h2h",
          "outcomes": [
            {"name": "Los Angeles Lakers", "price": -110},
            {"name": "Boston Celtics", "price": -110}
          ]
        },
        {
          "key": "spreads", 
          "outcomes": [
            {"name": "Los Angeles Lakers", "price": -110, "point": -2.5},
            {"name": "Boston Celtics", "price": -110, "point": 2.5}
          ]
        },
        {
          "key": "totals",
          "outcomes": [
            {"name": "Over", "price": -110, "point": 218.5},
            {"name": "Under", "price": -110, "point": 218.5}
          ]
        }
      ]
    }
  ]
}
```

## 🎯 Available NBA Markets

### Supported Bet Types
- **h2h**: Moneyline (who wins)
- **spreads**: Point spreads 
- **totals**: Over/under total points
- **player_props**: Player statistics (premium)

### Supported Regions
- **us**: US bookmakers (DraftKings, FanDuel, etc.)
- **uk**: UK bookmakers  
- **au**: Australian bookmakers
- **eu**: European bookmakers

## 🔧 API Usage in Your System

### Your System Will Use The Odds API For:

1. **Live Odds Collection**
   ```python
   # odds_api_client.py automatically handles this
   odds_data = client.get_all_odds()
   ```

2. **Best Odds Finding**
   ```python
   # Finds best prices across all bookmakers
   best_odds = client.find_best_odds(odds_data)
   ```

3. **Arbitrage Detection**
   ```python
   # Identifies profit opportunities
   arb_opps = client.detect_arbitrage_opportunities(odds_data)
   ```

4. **Line Movement Tracking**
   ```python
   # Monitors how odds change over time
   movement = client.get_line_movement(game_id, hours_back=24)
   ```

## 📈 Rate Limiting & Best Practices

### Free Tier Limits
- **500 requests/month** = ~16 requests/day
- **1 request/second** maximum
- **Your system automatically handles rate limiting**

### Optimization Tips
- **Cache responses**: Your system stores data for 5 minutes
- **Batch requests**: Get multiple games at once
- **Smart polling**: Only update when needed

### Request Examples
```bash
# Get all NBA games today
https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey=YOUR_KEY

# Get specific markets only
https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey=YOUR_KEY&markets=h2h,spreads

# US bookmakers only
https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey=YOUR_KEY&regions=us
```

## 🚨 Important Notes

### Security
- **Never share your API key**
- **Don't commit it to GitHub**
- **Use environment variables in production**

### Usage Monitoring
- **Check your dashboard**: See requests used/remaining
- **Set up alerts**: Get notified near your limit
- **Upgrade if needed**: More requests for heavy usage

### Data Freshness
- **Updates every 30 seconds** during games
- **Less frequent** when games aren't active  
- **Your system caches** to minimize requests

## 🎯 Expected Results After Setup

Once you add your Odds API key:

### ✅ What Will Work
- **Live NBA odds** from multiple bookmakers
- **Real-time line movement** tracking
- **Arbitrage opportunity** detection
- **Best odds comparison** across books
- **Automated odds collection** every 5 minutes

### 📊 Dashboard Features Unlocked
- **Live Odds Tab**: Current market prices
- **Line Movement Charts**: How odds change
- **Arbitrage Alerts**: Profit opportunities
- **Bookmaker Comparison**: Best available prices

### 🔄 System Integration
- **Automatic data updates** every 5 minutes
- **Kelly criterion calculations** with live odds
- **Expected value computations** for each bet
- **Risk-free arbitrage** opportunity alerts

## 🚀 Quick Setup Commands

### 1. Get your API key from https://the-odds-api.com/

### 2. Update config.json:
```bash
notepad config.json  # Windows
# Add your key: "odds_api_key": "your_actual_key_here"
```

### 3. Test the connection:
```bash
python api_test.py
```

### 4. Start the system:
```bash
python main.py
```

## 💡 Pro Tips

### Free Tier Strategy
- **Use for development**: Perfect for building/testing
- **Monitor usage**: Check dashboard regularly
- **Cache aggressively**: Your system does this automatically
- **Focus on high-value games**: Don't waste requests

### Upgrade Timing
- **When approaching 500/month**: Consider $10 plan
- **For live trading**: Higher tiers for real-time data
- **Multiple strategies**: More requests needed

---

**Ready to get started? Visit https://the-odds-api.com/ and get your free API key to unlock live NBA odds data in your betting system!**