"""
CLV Integration Demo - How to Hook Up Closing Line Value Tracking

This script demonstrates the complete CLV workflow:
1. Initialize CLV tracker at start of prediction loop
2. Log bets immediately when placed (with your locked-in odds)
3. Update closing lines after games start (backfill closing odds)
4. Analyze Sharp vs Soft performance

CRITICAL: This is the ultimate truth serum - proves if your edge is real or luck.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.prediction_engine import CLVTracker

def american_to_decimal(american_odds: float) -> float:
    """
    Convert American odds to Decimal format for CLV tracker.
    
    Examples:
        -110 → 1.909
        +150 → 2.50
        -200 → 1.50
    """
    if american_odds < 0:
        return (100 / abs(american_odds)) + 1
    else:
        return (american_odds / 100) + 1


def demo_clv_workflow():
    """Complete CLV tracking workflow demonstration"""
    
    print("=" * 80)
    print("🎯 CLV TRACKER INTEGRATION DEMO")
    print("=" * 80)
    
    # ==========================================
    # STEP 1: Initialize CLV Tracker
    # ==========================================
    print("\n📋 STEP 1: Initialize CLV Tracker")
    
    # Clean slate for demo
    import os
    if os.path.exists('logs/demo_clv_tracker.csv'):
        os.remove('logs/demo_clv_tracker.csv')
    
    clv_tracker = CLVTracker(log_path='logs/demo_clv_tracker.csv')
    print("   ✅ Tracker initialized at logs/demo_clv_tracker.csv")
    
    # ==========================================
    # STEP 2: Log Bets When Placing
    # ==========================================
    print("\n📋 STEP 2: Log Bets (Simulation of 3 bets)")
    print("=" * 80)
    
    # Bet 1: Celtics at -110 (you think they'll cover)
    print("\n🏀 Game 1: BOS Celtics vs MIA Heat")
    print("   Your Bet: BOS -5.5 at -110")
    clv_tracker.log_bet(
        game_id='0022400123',
        team='BOS Celtics',
        wager=100.0,
        bet_odds=american_to_decimal(-110)  # 1.909
    )
    
    # Bet 2: Lakers at +150 (underdog value play)
    print("\n🏀 Game 2: LAL Lakers vs DEN Nuggets")
    print("   Your Bet: LAL +6.5 at +150")
    clv_tracker.log_bet(
        game_id='0022400124',
        team='LAL Lakers',
        wager=75.0,
        bet_odds=american_to_decimal(+150)  # 2.50
    )
    
    # Bet 3: Warriors at -200 (heavy favorite)
    print("\n🏀 Game 3: GSW Warriors vs POR Trail Blazers")
    print("   Your Bet: GSW -10.5 at -200")
    clv_tracker.log_bet(
        game_id='0022400125',
        team='GSW Warriors',
        wager=150.0,
        bet_odds=american_to_decimal(-200)  # 1.50
    )
    
    # ==========================================
    # STEP 3: Simulate Time Passing (Games About to Start)
    # ==========================================
    print("\n" + "=" * 80)
    print("⏰ STEP 3: Games About to Start - Collect Closing Lines")
    print("=" * 80)
    print("\n(In production, you'd scrape closing odds from Kalshi/Pinnacle/DraftKings)")
    
    # ==========================================
    # STEP 4: Update with Closing Lines
    # ==========================================
    print("\n📋 STEP 4: Update CLV with Closing Lines")
    print("=" * 80)
    
    # Simulate closing lines collected from market
    closing_lines = {
        # Game 1: Line moved TO -120 (worse for you) → POSITIVE CLV
        ('22400123', 'BOS Celtics'): american_to_decimal(-120),  # 1.833
        
        # Game 2: Line moved TO +130 (better for market) → NEGATIVE CLV
        ('22400124', 'LAL Lakers'): american_to_decimal(+130),  # 2.30
        
        # Game 3: Line stayed at -200 (no movement) → ZERO CLV
        ('22400125', 'GSW Warriors'): american_to_decimal(-200),  # 1.50
    }
    
    print("\n📊 Closing Line Movement:")
    print("   BOS: You bet -110 (1.909) → Closed -120 (1.833) [Sharp money moved it worse]")
    print("   LAL: You bet +150 (2.500) → Closed +130 (2.300) [Market improved odds]")
    print("   GSW: You bet -200 (1.500) → Closed -200 (1.500) [No movement]")
    
    # Update and get report
    clv_tracker.update_closing_lines(closing_lines)
    
    # ==========================================
    # INTERPRETATION
    # ==========================================
    print("\n" + "=" * 80)
    print("📚 HOW TO INTERPRET CLV RESULTS")
    print("=" * 80)
    
    print("\n✅ BOS Celtics: +4.1% CLV (SHARP)")
    print("   → You got -110, sharp money pushed it to -120")
    print("   → You captured value before the market adjusted")
    print("   → This is GOOD even if you lose the bet")
    
    print("\n❌ LAL Lakers: -8.7% CLV (SOFT)")
    print("   → You got +150, but could've waited for +130")
    print("   → You bet BEFORE the market settled")
    print("   → This is BAD even if you win the bet")
    
    print("\n⚪ GSW Warriors: 0.0% CLV (NEUTRAL)")
    print("   → No line movement, efficient market price")
    print("   → Neither sharp nor soft bet")
    
    print("\n" + "=" * 80)
    print("💡 KEY TAKEAWAYS")
    print("=" * 80)
    print("\n1. Positive AVG CLV = You're beating the closing line (Sharp bettor)")
    print("2. Negative AVG CLV = You're chasing steam/betting too early (Soft)")
    print("3. Target: +2% to +5% average CLV for consistent profit")
    print("4. CLV > Win Rate: You can lose 48% and still profit with good CLV")
    print("5. Win Rate > CLV: 60% wins with -5% CLV = eventual ruin")
    
    print("\n" + "=" * 80)
    print("🔗 INTEGRATION INTO YOUR PREDICTION LOOP")
    print("=" * 80)
    print("\nInside your main prediction script, add:")
    print("""
# At top of script
from src.prediction.prediction_engine import CLVTracker

# Initialize once
clv_tracker = CLVTracker()

# Inside your game loop
for game in today_games:
    prediction = prediction_engine.predict_total(game, features)
    
    if should_bet(prediction):
        # Place bet via Kalshi API
        order = kalshi_client.place_order(...)
        
        # LOG THE BET IMMEDIATELY
        clv_tracker.log_bet(
            game_id=game.game_id,
            team=team_name,
            wager=kelly_stake,
            bet_odds=order.price_decimal  # Already in decimal format
        )

# After games start (in nightly script)
closing_map = fetch_closing_odds_from_provider()
clv_tracker.update_closing_lines(closing_map)
    """)
    
    print("\n" + "=" * 80)
    print("✅ Demo Complete! Check logs/demo_clv_tracker.csv")
    print("=" * 80)


if __name__ == "__main__":
    demo_clv_workflow()
