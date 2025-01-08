# config.py

# ~~~ POSTGRESQL SETTINGS ~~~
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "mydatabase"
DB_USER = "myuser"
DB_PASS = "mypassword"

# ~~~ DATA RETRIEVAL SETTINGS ~~~
# Change these to fetch more historical data
START_DATE = "2020-01-01"
END_DATE   = None

# ~~~ MOVING AVERAGE WINDOWS ~~~
MA_SHORT = 50
MA_LONG = 200
