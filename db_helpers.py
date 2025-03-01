import logging
import psycopg2.extras
import pandas as pd
from sqlalchemy import create_engine
from typing import List, Dict
from datetime import timedelta
from concurrent.futures import as_completed, ThreadPoolExecutor

import config
from db_manager import db_pool
import psycopg2

# Initialize SQLAlchemy engine once in this module
DATABASE_URL = f"postgresql://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
engine = create_engine(DATABASE_URL)



def create_tables(conn):
    """
    Create only the tables needed for analysis, 
    excluding price_data and sector_data (which the other program handles).
    """
    cur = conn.cursor()

    # indicator_data ...
    create_indicator_table = """
    CREATE TABLE IF NOT EXISTS indicator_data (
        indicator_name VARCHAR(50),
        data_date DATE,
        value1 NUMERIC,
        value2 NUMERIC,
        value3 NUMERIC,
        value4 NUMERIC,
        value5 NUMERIC,
        PRIMARY KEY(indicator_name, data_date)
    );
    """
    cur.execute(create_indicator_table)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_indicator_data_date ON indicator_data (data_date);")

    # phase_details ...
    create_phase_details = """
    CREATE TABLE IF NOT EXISTS phase_details (
        ticker VARCHAR(20),
        data_date DATE,
        close NUMERIC,
        sma_50 NUMERIC,
        sma_200 NUMERIC,
        phase VARCHAR(50),
        PRIMARY KEY (ticker, data_date)
    );
    """
    cur.execute(create_phase_details)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_phase_details_date ON phase_details (data_date);")

    # volume_ma_data ...
    create_volume_ma_data = """
    CREATE TABLE IF NOT EXISTS volume_ma_data (
        ticker VARCHAR(20),
        trade_date DATE,
        vol_ma_10 NUMERIC,
        vol_ma_20 NUMERIC,
        PRIMARY KEY(ticker, trade_date)
    );
    """
    cur.execute(create_volume_ma_data)

    # price_ma_deviation ...
    create_price_ma_deviation = """
    CREATE TABLE IF NOT EXISTS price_ma_deviation (
        ticker VARCHAR(20),
        data_date DATE,
        dev_50 NUMERIC,
        dev_200 NUMERIC,
        PRIMARY KEY(ticker, data_date)
    );
    """
    cur.execute(create_price_ma_deviation)

    # volume_ma_deviation ...
    create_volume_ma_deviation = """
    CREATE TABLE IF NOT EXISTS volume_ma_deviation (
        ticker VARCHAR(20),
        data_date DATE,
        dev_20 NUMERIC,
        dev_63 NUMERIC,
        PRIMARY KEY(ticker, data_date)
    );
    """
    cur.execute(create_volume_ma_deviation)

    # sector_analysis (needed for your sector analysis results)
    create_sector_analysis_table(conn)

    conn.commit()
    cur.close()


def create_sector_analysis_table(conn):
    """
    Create sector_analysis if it doesn't exist. 
    This table is filled by run_sector_analysis.
    """
    cur = conn.cursor()
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS sector_analysis (
        data_date DATE NOT NULL,
        ticker VARCHAR(20) NOT NULL,
        daily_return NUMERIC,
        cumulative_return NUMERIC,
        rolling_corr_spy NUMERIC,
        relative_perf_spy NUMERIC,
        PRIMARY KEY (data_date, ticker)
    );
    """
    cur.execute(create_table_sql)
    conn.commit()
    cur.close()

def write_phase_details_to_db(df: pd.DataFrame, conn):
    """
    Writes the per-ticker daily classification to the 'phase_details' table.
    
    df is expected to have columns:
      Ticker, Close, SMA_50, SMA_200, Phase
    and the DataFrame index is the date.
    """
    if df.empty:
        return
    
    cur = conn.cursor()
    records = []
    for dt_, row in df.iterrows():
        date_ = pd.to_datetime(dt_).date()
        ticker_ = row["Ticker"]
        close_ = float(row["Close"])
        sma_50_ = float(row["SMA_50"])
        sma_200_ = float(row["SMA_200"])
        phase_ = row["Phase"]
        
        records.append((ticker_, date_, close_, sma_50_, sma_200_, phase_))
    
    insert_query = """
        INSERT INTO phase_details
          (ticker, data_date, close, sma_50, sma_200, phase)
        VALUES %s
        ON CONFLICT (ticker, data_date) DO UPDATE
          SET
            close = EXCLUDED.close,
            sma_50 = EXCLUDED.sma_50,
            sma_200 = EXCLUDED.sma_200,
            phase = EXCLUDED.phase
    """
    psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
    conn.commit()
    cur.close()


def write_indicator_data_to_db(new_df: pd.DataFrame, indicator_name: str, conn):
    """
    Write a DataFrame of indicator values into the indicator_data table.
    """
    if new_df.empty:
        logging.info(f"No new data for indicator '{indicator_name}'. Skipping DB insert.")
        return
    
    cur = conn.cursor()
    records = []
    for dt_, row in new_df.iterrows():
        dt_ = dt_.date()
        values = row.values.tolist()
        float_values = []
        for val in values:
            if pd.isna(val):
                float_values.append(None)
            else:
                float_values.append(float(val))
        # Pad or trim to exactly 5
        float_values += [None] * (5 - len(float_values))
        float_values = float_values[:5]
        records.append((indicator_name, dt_, *float_values))
    
    insert_query = """
        INSERT INTO indicator_data
          (indicator_name, data_date, value1, value2, value3, value4, value5)
        VALUES %s
        ON CONFLICT (indicator_name, data_date) DO UPDATE
          SET value1 = EXCLUDED.value1,
              value2 = EXCLUDED.value2,
              value3 = EXCLUDED.value3,
              value4 = EXCLUDED.value4,
              value5 = EXCLUDED.value5
    """
    psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
    conn.commit()
    cur.close()

def write_sector_analysis_metrics(conn, metric_records):
    """
    Insert or update rows in 'sector_analysis' table.
    'metric_records' is a list of tuples:
      (data_date, ticker, daily_return, cumulative_return, rolling_corr_spy, relative_perf_spy)
    """
    if not metric_records:
        return

    cur = conn.cursor()
    insert_query = """
        INSERT INTO sector_analysis (
            data_date, ticker, daily_return, cumulative_return,
            rolling_corr_spy, relative_perf_spy
        )
        VALUES %s
        ON CONFLICT (data_date, ticker) DO UPDATE
          SET daily_return = EXCLUDED.daily_return,
              cumulative_return = EXCLUDED.cumulative_return,
              rolling_corr_spy = EXCLUDED.rolling_corr_spy,
              relative_perf_spy = EXCLUDED.relative_perf_spy
    """
    psycopg2.extras.execute_values(cur, insert_query, metric_records, page_size=1000)
    conn.commit()
    cur.close()

def read_ticker_data_from_db(ticker: str, conn) -> pd.DataFrame:
    """
    Reads all available data for `ticker` from price_data,
    including 'sector'.
    """
    query = """
        SELECT trade_date, open, high, low, close, volume, sector
        FROM price_data
        WHERE ticker = %s
        ORDER BY trade_date
    """
    df = pd.read_sql(query, engine, params=(ticker,))
    if not df.empty:
        df.set_index("trade_date", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.columns = ["Open", "High", "Low", "Close", "Volume", "sector"]
        # Convert numeric columns as needed
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        df["Volume"] = df["Volume"].astype("int64", errors="ignore")
    return df


def batch_fetch_from_db(tickers: List[str], conn) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for the given tickers from price_data (old table),
    including the 'sector' column if available.
    Trims data by START_DATE with a lookback for MAs, etc.
    """
    data_dict = {}
    batch_size = 50
    max_workers = min(config.MAX_WORKERS, len(tickers)) if tickers else 1
    
    lookback_days = max(config.MA_SHORT, config.MA_LONG)
    start_date = pd.to_datetime(config.START_DATE) if config.START_DATE else None
    if start_date is not None:
        earliest_date = start_date - timedelta(days=lookback_days)
    else:
        earliest_date = None
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(read_ticker_data_from_db, ticker, conn): ticker
                for ticker in batch
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if not df.empty:
                        if earliest_date is not None:
                            df = df[df.index >= earliest_date]
                        if config.END_DATE:
                            end_date = pd.to_datetime(config.END_DATE)
                            df = df[df.index <= end_date]
                        if not df.empty:
                            data_dict[ticker] = df
                        else:
                            logging.info(f"No data within range for {ticker}.")
                    else:
                        logging.info(f"No data found in DB for {ticker}.")
                except Exception as e:
                    logging.error(f"Error loading data for {ticker}: {e}")
    return data_dict


# --- NEW or UPDATED ---
# --- BATCH FETCH for sector_data
def read_etf_data_from_db(ticker: str, conn) -> pd.DataFrame:
    # same as before but from sector_data
    query = """
        SELECT trade_date, open, high, low, close, volume, sma_50, sma_200
        FROM sector_data
        WHERE ticker = %s
        ORDER BY trade_date
    """
    df = pd.read_sql(query, engine, params=(ticker,))
    if not df.empty:
        df.set_index("trade_date", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.columns = ["Open", "High", "Low", "Close", "Volume", "SMA_50", "SMA_200"]
        for col in ["Open", "High", "Low", "Close", "SMA_50", "SMA_200"]:
            df[col] = df[col].astype(float)
        df["Volume"] = df["Volume"].astype("int64", errors="ignore")
    return df

def batch_fetch_sector_data(tickers: List[str], conn) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for the given tickers from the new 'sector_data' table.
    Trims data by START_DATE with a lookback for MAs, etc.
    """
    data_dict = {}
    batch_size = 50
    max_workers = min(config.MAX_WORKERS, len(tickers)) if tickers else 1
    
    lookback_days = max(config.MA_SHORT, config.MA_LONG)
    start_date = pd.to_datetime(config.START_DATE) if config.START_DATE else None
    if start_date is not None:
        earliest_date = start_date - timedelta(days=lookback_days)
    else:
        earliest_date = None

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(read_etf_data_from_db, ticker, conn): ticker
                for ticker in batch
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if not df.empty:
                        if earliest_date is not None:
                            df = df[df.index >= earliest_date]
                        if config.END_DATE:
                            end_date = pd.to_datetime(config.END_DATE)
                            df = df[df.index <= end_date]
                        if not df.empty:
                            data_dict[ticker] = df
                        else:
                            logging.info(f"No data within range for {ticker} in sector_data.")
                    else:
                        logging.info(f"No data found in sector_data for {ticker}.")
                except Exception as e:
                    logging.error(f"Error loading sector_data for {ticker}: {e}")
    return data_dict

def detect_and_log_changes(conn):
    """
    1) Detect any ticker that changes its phase (old -> new).
    2) Detect price crossing above/below SMA_50 or SMA_200.
    3) Detect golden/death crosses (SMA_50 crossing SMA_200).

    Results are written to two files, read from config:
      config.PHASE_CHANGES_FILE
      config.PRICE_SMA_CHANGES_FILE
    """
    phase_changes_file = config.PHASE_CHANGES_FILE
    price_sma_changes_file = config.PRICE_SMA_CHANGES_FILE

    query = """
        SELECT ticker, data_date, close, sma_50, sma_200, phase
        FROM phase_details
        WHERE data_date IN (
            SELECT DISTINCT data_date
            FROM phase_details
            ORDER BY data_date DESC
            LIMIT 2
        )
        ORDER BY ticker, data_date
    """
    
    df = pd.read_sql(query, conn)
    if df.empty:
        with open(phase_changes_file, "w") as f:
            f.write("No data found in phase_details, no phase changes detected.\n")
        with open(price_sma_changes_file, "w") as f:
            f.write("No data found in phase_details, no price/SMA changes detected.\n")
        return
    
    df.sort_values(["ticker", "data_date"], inplace=True)
    df["prev_phase"] = df.groupby("ticker")["phase"].shift(1)
    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["prev_sma_50"] = df.groupby("ticker")["sma_50"].shift(1)
    df["prev_sma_200"] = df.groupby("ticker")["sma_200"].shift(1)
    
    phase_changes = []
    price_sma_crosses = []
    golden_death_crosses = []
    
    for idx, row in df.iterrows():
        ticker = row["ticker"]
        date_ = row["data_date"]
        old_phase = row["prev_phase"]
        new_phase = row["phase"]

        # 1) Phase change detection
        if pd.notna(old_phase) and pd.notna(new_phase) and (old_phase != new_phase):
            phase_changes.append(
                f"[{date_}] {ticker} changed phase from {old_phase} to {new_phase}"
            )
        
        # 2) Price crossing above/below 50 or 200
        c_now = row["close"]
        c_prev = row["prev_close"]
        sma50_now = row["sma_50"]
        sma50_prev = row["prev_sma_50"]
        sma200_now = row["sma_200"]
        sma200_prev = row["prev_sma_200"]
        
        if (pd.notna(c_now) and pd.notna(c_prev) 
            and pd.notna(sma50_now) and pd.notna(sma50_prev)):
            if (c_prev < sma50_prev) and (c_now > sma50_now):
                price_sma_crosses.append(
                    f"[{date_}] {ticker} price crossed ABOVE 50-day SMA"
                )
            if (c_prev > sma50_prev) and (c_now < sma50_now):
                price_sma_crosses.append(
                    f"[{date_}] {ticker} price crossed BELOW 50-day SMA"
                )
        
        if (pd.notna(c_now) and pd.notna(c_prev) 
            and pd.notna(sma200_now) and pd.notna(sma200_prev)):
            if (c_prev < sma200_prev) and (c_now > sma200_now):
                price_sma_crosses.append(
                    f"[{date_}] {ticker} price crossed ABOVE 200-day SMA"
                )
            if (c_prev > sma200_prev) and (c_now < sma200_now):
                price_sma_crosses.append(
                    f"[{date_}] {ticker} price crossed BELOW 200-day SMA"
                )
        
        # 3) Golden / Death Cross
        if (pd.notna(sma50_now) and pd.notna(sma50_prev) 
            and pd.notna(sma200_now) and pd.notna(sma200_prev)):
            if (sma50_prev < sma200_prev) and (sma50_now > sma200_now):
                golden_death_crosses.append(
                    f"[{date_}] {ticker} GOLDEN CROSS (50-day SMA above 200-day SMA)"
                )
            if (sma50_prev > sma200_prev) and (sma50_now < sma200_now):
                golden_death_crosses.append(
                    f"[{date_}] {ticker} DEATH CROSS (50-day SMA below 200-day SMA)"
                )
    
    # Write phase changes
    with open(phase_changes_file, "w") as f:
        if phase_changes:
            f.write("=== PHASE CHANGES ===\n\n")
            for line in phase_changes:
                f.write(line + "\n")
        else:
            f.write("No phase changes detected.\n")
    
    # Write price/SMA changes
    with open(price_sma_changes_file, "w") as f:
        if price_sma_crosses or golden_death_crosses:
            f.write("=== PRICE/SMA CHANGES ===\n\n")
            if price_sma_crosses:
                f.write(">> Price / SMA Crosses:\n")
                for line in price_sma_crosses:
                    f.write(line + "\n")
                f.write("\n")
            if golden_death_crosses:
                f.write(">> Golden / Death Crosses:\n")
                for line in golden_death_crosses:
                    f.write(line + "\n")
        else:
            f.write("No price/SMA changes detected.\n")