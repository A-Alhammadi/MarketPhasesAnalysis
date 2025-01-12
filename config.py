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
# ADDED: For dynamic pool sizing
# ------------------
MINCONN = 1
MAXCONN = 20

# ====================
# DATA RETRIEVAL
# ====================
# In this version we do not fetch from Yahoo or elsewhere,
# but you can still set these if you plan to filter DB data by date, etc.
START_DATE = "2024-01-01"
END_DATE = None
DATA_FETCH_INTERVAL = "1d"  # This is no longer used here, but kept for reference

# ====================
# MOVING AVERAGES
# ====================
MA_SHORT = 50
MA_LONG = 200

# ====================
# PLOT SETTINGS
# ====================
PLOT_PHASES = {
    "Bullish": True,
    "Caution": True,
    "Distribution": True,
    "Bearish": True,
    "Recuperation": True,
    "Accumulation": True,
    "AllPhases": True
}

PLOT_INDICATORS = {
    "adv_decline": True,
    "adv_decline_volume": True,
    "new_high_low": True,
    "percent_above_ma": True,
    "mcclellan": True,
    "fear_greed": True,
    "trend_intensity_index": True,
    "chaikin_volatility": True,
    "chaikin_money_flow": True,
    "trin": True
}

RESULTS_DIR = "results"

# ==========================================
# NEW: Intervals for phase & indicator plots
# ==========================================
# Valid Pandas offset aliases include 'D', 'W', 'M', 'Q', 'Y', etc.
PHASE_PLOT_INTERVAL = "W"       # e.g. weekly
INDICATOR_PLOT_INTERVAL = "W"   # e.g. weekly

# ------------------
# Cache size & memory threshold
# ------------------
MAX_ROLLING_MEANS_CACHE_SIZE = 2000
MAX_MEMORY_PERCENT = 95  # e.g., if memory usage goes above 95%, consider skipping

# ====================
# THREAD POOL SETTINGS
# ====================
MAX_WORKERS = 10  # <--- Adjust thread pool concurrency
