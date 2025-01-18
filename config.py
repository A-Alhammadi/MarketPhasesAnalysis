# config.py

# ====================
# POSTGRESQL SETTINGS
# ====================
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "mydatabase"
DB_USER = "myuser"
DB_PASS = "mypassword"

# ------------------
# NEW: For dynamic pool sizing
# ------------------
MINCONN = 1
MAXCONN = 20

# ------------------
# NEW: DB Connection Retry Settings
# ------------------
DB_RETRY_ATTEMPTS = 3
DB_RETRY_DELAY = 5  # seconds

# ====================
# DATA RETRIEVAL
# ====================
START_DATE = "2023-01-01"
END_DATE = None

# ------------------
# NEW: Address for the list of stocks
# ------------------
STOCK_LIST_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# ------------------
# Optionally choose data source or other logic
# (e.g., 'WIKI', 'LOCAL_FILE', etc.)
# ------------------
STOCK_LIST_SOURCE = "WIKI"


SP500_TICKER = "SPY"       # Or "SPY" or whatever ticker you use for S&P 500
ROLLING_WINDOW = 63          # e.g. 63 trading days (~3 months) for rolling correlation
PERIODS = {
    "10Y": 10,
    "5Y": 5,
    "3Y": 3,
    "1Y": 1
}

# ====================
# MOVING AVERAGES
# ====================
MA_SHORT = 50
MA_LONG = 200

# ------------------
# NEW: for compute_new_high_low default lookback
# ------------------
NEWHIGHLOW_LOOKBACK = 252

# ------------------
# NEW: for compute_percent_above_ma default window
# ------------------
PERCENTABOVE_MA_WINDOW = 200

# ====================
# PLOT SETTINGS
# ====================
PLOT_PHASES = {
    "Bullish": True,
    "Caution": False,
    "Distribution": False,
    "Bearish": False,
    "Recuperation": False,
    "Accumulation": False,
    "AllPhases": False
}

PLOT_INDICATORS = {
    "adv_decline": True,
    "adv_decline_volume": False,
    "new_high_low": False,
    "percent_above_ma": False,
    "mcclellan": False,
    "fear_greed": False,
    "trend_intensity_index": False,
    "chaikin_volatility": False,
    "chaikin_money_flow": False,
    "trin": False
}

RESULTS_DIR = "results"

# ==========================================
# NEW: Intervals for phase & indicator plots
# ==========================================
PHASE_PLOT_INTERVAL = "W"       # e.g. weekly
INDICATOR_PLOT_INTERVAL = "W"   # e.g. weekly

# ------------------
# Cache size & memory threshold
# ------------------
MAX_ROLLING_MEANS_CACHE_SIZE = 2000
MAX_MEMORY_PERCENT = 95

# ====================
# THREAD POOL SETTINGS
# ====================
MAX_WORKERS = 10
