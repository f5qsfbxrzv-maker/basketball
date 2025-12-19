from datetime import datetime
print(f"Today's date: {datetime.now().strftime('%Y-%m-%d')}")
print(f"Database latest game: 2025-11-20")
print(f"Days difference: 24 days")
print("\nThis seems wrong - December 14 should have many games by now.")
print("Let me check what's actually happening...")

# The NBA 2024-25 season started October 22, 2024
# If the latest game in the database is 2025-11-20, that's in the FUTURE from a 2024 perspective
# But we're now in December 2025, so... wait

# Let me recalculate
from datetime import date
today = date(2025, 12, 14)
latest_db = date(2025, 11, 20)
diff = (today - latest_db).days
print(f"\nActual calculation: {diff} days difference")
print("This matches the '24 days old' message")
print("\nSO: The database IS stale - it hasn't been updated since Nov 20, 2025")
