# ====================
# POSTGRESQL SETTINGS
# ====================
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "mydatabase"
DB_USER = "myuser"
DB_PASS = "mypassword"

# ====================
# DATA RETRIEVAL
# ====================
START_DATE = "2020-01-01"
END_DATE = None
# e.g., "1d", "1wk", "1mo"
DATA_FETCH_INTERVAL = "1wk"

# ====================
# MOVING AVERAGE
# ====================
MA_SHORT = 8
MA_LONG = 40

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
# Valid Pandas offset aliases include 'D', 'W', 'M', 'Q', 'Y', etc.
PHASE_PLOT_INTERVAL = "W"       # e.g. monthly
INDICATOR_PLOT_INTERVAL = "W"   # e.g. monthly
