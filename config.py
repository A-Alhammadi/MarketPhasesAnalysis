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
DATA_FETCH_INTERVAL = "1d"

# ====================
# MOVING AVERAGE
# ====================
MA_SHORT = 50
MA_LONG = 200

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
    "adv_decline": False,
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
