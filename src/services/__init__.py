"""Services module for external data sources"""

# Don't auto-import modules - let them be imported explicitly when needed
# This prevents unrelated imports (e.g., LiveOddsFetcher shouldn't trigger InjuryScraper imports)

__all__ = [
    'InjuryScraper',
    'LiveOddsFetcher',
    'KalshiClient',
]

