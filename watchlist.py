"""
watchlist.py — Default watchlist of large-cap US stocks
Pradeep's preferred trading universe: big, liquid, well-known companies.
"""

# Default watchlist — large cap US stocks
DEFAULT_WATCHLIST = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # Nvidia
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet
    "META",   # Meta
    "TSLA",   # Tesla
    "JPM",    # JPMorgan
    "V",      # Visa
    "SPY",    # S&P 500 ETF
    "QQQ",    # Nasdaq ETF
]

# Sectors for filtering
SECTORS = {
    "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "EV / Auto": ["TSLA"],
    "E-Commerce": ["AMZN"],
    "Finance": ["JPM", "V"],
    "ETFs": ["SPY", "QQQ"],
}
