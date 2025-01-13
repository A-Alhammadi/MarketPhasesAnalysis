import os
import sys
import logging
import pandas as pd
import matplotlib
import random
import time
import gc
import cProfile
import pstats
import atexit
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from concurrent.futures import as_completed, ThreadPoolExecutor

import config
from db_manager import DBConnectionManager, db_pool
from perf_utils import measure_time, log_memory_usage
from plotting import save_phase_plots, save_indicator_plots

# Phase classification
from phase_analysis import classify_phases, log_rolling_cache_stats

# Indicators
from analysis.indicators.indicator_template import (
    clear_caches, run_indicator
)
from analysis.indicators.breadth_indicators import (
    compute_adv_decline,
    compute_adv_decline_volume,
    compute_mcclellan
)
from analysis.indicators.high_low_indicators import (
    compute_new_high_low,
    compute_percent_above_ma
)
from analysis.indicators.money_flow_indicators import (
    compute_index_of_fear_greed,
    compute_chaikin_money_flow
)
from analysis.indicators.trend_indicators import (
    compute_trend_intensity_index
)
from analysis.indicators.volatility_indicators import (
    compute_chaikin_volatility,
    compute_trin
)

matplotlib.use("Agg")  # Headless mode

import psycopg2.extras
from sqlalchemy import create_engine

# Initialize SQLAlchemy engine once in the script
DATABASE_URL = f"postgresql://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
engine = create_engine(DATABASE_URL)

###############################################################################
# DB Helpers & Table Creation
###############################################################################
def create_tables(conn):
    """Create the price_data, indicator_data, new phase_details tables, 
       plus new tables for volume MAs and price/volume MA deviations.
    """
    cur = conn.cursor()
    
    # 1) Existing tables...
    create_price_table = """
    CREATE TABLE IF NOT EXISTS price_data (
        ticker VARCHAR(20),
        trade_date DATE,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        volume BIGINT,
        PRIMARY KEY(ticker, trade_date)
    );
    """
    cur.execute(create_price_table)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_price_trade_date ON price_data (trade_date);")

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

    # 2) NEW: volume_ma_data table
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

    # 3) NEW: price_ma_deviation table
    create_price_ma_deviation = """
    CREATE TABLE IF NOT EXISTS price_ma_deviation (
        ticker VARCHAR(20),
        data_date DATE,
        dev_50 NUMERIC,   -- % from 50-day
        dev_200 NUMERIC,  -- % from 200-day
        PRIMARY KEY(ticker, data_date)
    );
    """
    cur.execute(create_price_ma_deviation)

    # 4) NEW: volume_ma_deviation table
    create_volume_ma_deviation = """
    CREATE TABLE IF NOT EXISTS volume_ma_deviation (
        ticker VARCHAR(20),
        data_date DATE,
        dev_20 NUMERIC,  -- % from 20-day volume
        dev_63 NUMERIC,  -- % from 63-day volume
        PRIMARY KEY(ticker, data_date)
    );
    """
    cur.execute(create_volume_ma_deviation)

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


def detect_and_log_changes(conn, phase_changes_file="phase_changes.txt", price_sma_changes_file="price_sma_changes.txt"):
    """
    1) Detect any ticker that changes its phase (old -> new).
    2) Detect price crossing above/below SMA_50 or SMA_200.
    3) Detect golden/death crosses (SMA_50 crossing SMA_200).
    
    Results are written to separate files.
    """
    import pandas as pd

    # We only need the last 2 distinct dates from phase_details
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
        # No data at all? Then no changes can be detected
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

        # 1) Phase change
        if pd.notna(old_phase) and pd.notna(new_phase) and (old_phase != new_phase):
            phase_changes.append(f"[{date_}] {ticker} changed phase from {old_phase} to {new_phase}")

        # 2) Price crossing above/below 50-day or 200-day
        c_now = row["close"]
        c_prev = row["prev_close"]
        sma50_now = row["sma_50"]
        sma50_prev = row["prev_sma_50"]
        sma200_now = row["sma_200"]
        sma200_prev = row["prev_sma_200"]

        if pd.notna(c_now) and pd.notna(c_prev) and pd.notna(sma50_now) and pd.notna(sma50_prev):
            # Cross above 50
            if (c_prev < sma50_prev) and (c_now > sma50_now):
                price_sma_crosses.append(f"[{date_}] {ticker} price crossed ABOVE 50-day SMA")

            # Cross below 50
            if (c_prev > sma50_prev) and (c_now < sma50_now):
                price_sma_crosses.append(f"[{date_}] {ticker} price crossed BELOW 50-day SMA")

        if pd.notna(c_now) and pd.notna(c_prev) and pd.notna(sma200_now) and pd.notna(sma200_prev):
            # Cross above 200
            if (c_prev < sma200_prev) and (c_now > sma200_now):
                price_sma_crosses.append(f"[{date_}] {ticker} price crossed ABOVE 200-day SMA")

            # Cross below 200
            if (c_prev > sma200_prev) and (c_now < sma200_now):
                price_sma_crosses.append(f"[{date_}] {ticker} price crossed BELOW 200-day SMA")

        # 3) Golden / Death Cross
        if pd.notna(sma50_now) and pd.notna(sma50_prev) and pd.notna(sma200_now) and pd.notna(sma200_prev):
            # Golden cross: 50-day SMA crossing above 200-day SMA
            if (sma50_prev < sma200_prev) and (sma50_now > sma200_now):
                golden_death_crosses.append(f"[{date_}] {ticker} GOLDEN CROSS (50-day SMA above 200-day SMA)")

            # Death cross: 50-day SMA crossing below 200-day SMA
            if (sma50_prev > sma200_prev) and (sma50_now < sma200_now):
                golden_death_crosses.append(f"[{date_}] {ticker} DEATH CROSS (50-day SMA below 200-day SMA)")

    # Write phase changes to a separate file
    with open(phase_changes_file, "w") as f:
        if phase_changes:
            f.write("=== PHASE CHANGES ===\n\n")
            for line in phase_changes:
                f.write(line + "\n")
        else:
            f.write("No phase changes detected.\n")

    # Write price/SMA changes and golden/death crosses to another file
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


def write_indicator_data_to_db(new_df: pd.DataFrame, indicator_name: str, conn):
    """Write a DataFrame of indicator values into indicator_data table."""
    if new_df.empty:
        logging.info(f"No new data for indicator '{indicator_name}'. Skipping DB insert.")
        return
    cur = conn.cursor()
    records = []
    for dt_, row in new_df.iterrows():
        dt_ = dt_.date()
        values = row.values.tolist()
        # Up to 5 columns' worth of data stored as value1..value5
        # Convert everything to float or None if needed
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


def read_ticker_data_from_db(ticker: str, conn) -> pd.DataFrame:
    """
    Reads all available data for `ticker` from the price_data table in the DB.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    indexed by the trade_date.
    """
    query = """
        SELECT trade_date, open, high, low, close, volume
        FROM price_data
        WHERE ticker = %s
        ORDER BY trade_date
    """
    df = pd.read_sql(query, engine, params=(ticker,))
    if not df.empty:
        df.set_index("trade_date", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype("float32", errors="ignore")
        df["Volume"] = df["Volume"].astype("int64", errors="ignore")
    return df


def batch_fetch_from_db(tickers: List[str], conn) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for the given tickers, starting from an extended earliest date
    to allow for proper MA calculations. Trims based on START_DATE.
    """
    data_dict = {}
    batch_size = 50
    max_workers = min(config.MAX_WORKERS, len(tickers)) if tickers else 1

    lookback_days = max(config.MA_SHORT, config.MA_LONG)
    start_date = pd.to_datetime(config.START_DATE)
    earliest_date = start_date - timedelta(days=lookback_days)

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


def compute_and_store_volume_mas(data_dict: Dict[str, pd.DataFrame], conn):
    """
    For each ticker, compute the 10-day and 20-day volume MAs, store in volume_ma_data table.
    Fix: Convert any np.float to plain float.
    """
    if not data_dict:
        return
    
    cur = conn.cursor()
    records = []
    for ticker, df in data_dict.items():
        if df.empty or "Volume" not in df.columns:
            continue
        
        df.sort_index(inplace=True)
        df["Volume"] = df["Volume"].astype(float)

        df["vol_ma_10"] = df["Volume"].rolling(window=10, min_periods=1).mean()
        df["vol_ma_20"] = df["Volume"].rolling(window=20, min_periods=1).mean()

        for dt_, row in df.iterrows():
            trade_date = dt_.date()
            vol_ma10 = row["vol_ma_10"]
            vol_ma20 = row["vol_ma_20"]

            # Convert to Python float or None
            vol_ma10 = float(vol_ma10) if pd.notna(vol_ma10) else None
            vol_ma20 = float(vol_ma20) if pd.notna(vol_ma20) else None

            records.append((ticker, trade_date, vol_ma10, vol_ma20))

    insert_query = """
        INSERT INTO volume_ma_data
          (ticker, trade_date, vol_ma_10, vol_ma_20)
        VALUES %s
        ON CONFLICT (ticker, trade_date) DO UPDATE
          SET vol_ma_10 = EXCLUDED.vol_ma_10,
              vol_ma_20 = EXCLUDED.vol_ma_20
    """
    psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
    conn.commit()
    cur.close()


def compute_and_store_price_ma_deviation(data_dict: Dict[str, pd.DataFrame], conn):
    """
    Calculate how far Close is in % from its 50-day & 200-day MAs, store in price_ma_deviation.
    """
    if not data_dict:
        return
    
    cur = conn.cursor()
    records = []
    for ticker, df in data_dict.items():
        if df.empty or "Close" not in df.columns:
            continue
        
        df.sort_index(inplace=True)
        df["Close"] = df["Close"].astype(float)

        df["ma_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
        df["ma_200"] = df["Close"].rolling(window=200, min_periods=1).mean()

        df["dev_50"] = ((df["Close"] - df["ma_50"]) / df["ma_50"]) * 100
        df["dev_200"] = ((df["Close"] - df["ma_200"]) / df["ma_200"]) * 100

        for dt_, row in df.iterrows():
            data_date = dt_.date()
            dev50 = row["dev_50"]
            dev200 = row["dev_200"]

            # Convert to float or None
            dev50 = float(dev50) if pd.notna(dev50) else None
            dev200 = float(dev200) if pd.notna(dev200) else None

            # Skip earliest NaNs if desired
            if dev50 is None or dev200 is None:
                continue

            records.append((ticker, data_date, dev50, dev200))

    insert_query = """
        INSERT INTO price_ma_deviation
          (ticker, data_date, dev_50, dev_200)
        VALUES %s
        ON CONFLICT (ticker, data_date) DO UPDATE
          SET dev_50 = EXCLUDED.dev_50,
              dev_200 = EXCLUDED.dev_200
    """
    psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
    conn.commit()
    cur.close()


def compute_and_store_volume_ma_deviation(data_dict: Dict[str, pd.DataFrame], conn):
    """
    Calculate how far Volume is from its 20-day & 63-day MAs, store in volume_ma_deviation.
    """
    if not data_dict:
        return

    cur = conn.cursor()
    records = []

    for ticker, df in data_dict.items():
        if df.empty or "Volume" not in df.columns:
            continue
        
        df.sort_index(inplace=True)
        df["Volume"] = df["Volume"].astype(float)

        df["vol_ma_20"] = df["Volume"].rolling(window=20, min_periods=1).mean()
        df["vol_ma_63"] = df["Volume"].rolling(window=63, min_periods=1).mean()

        df["dev_20"] = ((df["Volume"] - df["vol_ma_20"]) / df["vol_ma_20"]) * 100
        df["dev_63"] = ((df["Volume"] - df["vol_ma_63"]) / df["vol_ma_63"]) * 100

        for dt_, row in df.iterrows():
            data_date = dt_.date()
            d20 = row["dev_20"]
            d63 = row["dev_63"]

            # Convert to float or None
            d20 = float(d20) if pd.notna(d20) else None
            d63 = float(d63) if pd.notna(d63) else None

            if d20 is None or d63 is None:
                continue
            records.append((ticker, data_date, d20, d63))

    insert_query = """
        INSERT INTO volume_ma_deviation
          (ticker, data_date, dev_20, dev_63)
        VALUES %s
        ON CONFLICT (ticker, data_date) DO UPDATE
          SET dev_20 = EXCLUDED.dev_20,
              dev_63 = EXCLUDED.dev_63
    """
    psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
    conn.commit()
    cur.close()


def export_extreme_volumes(conn, z_threshold=2.0):
    """
    Exports stocks with extreme volume deviations based on z-scores.
    """
    cur = conn.cursor()

    # 1) Find the last date in volume_ma_deviation
    cur.execute("SELECT MAX(data_date) FROM volume_ma_deviation;")
    row = cur.fetchone()
    if not row or not row[0]:
        logging.warning("No data in volume_ma_deviation table.")
        cur.close()
        return
    last_date = row[0]  # date object

    # 2) Query the data for that date
    query = """
        SELECT ticker, dev_20, dev_63
        FROM volume_ma_deviation
        WHERE data_date = %s
    """
    cur.execute(query, (last_date,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        logging.info(f"No volume deviation data on {last_date}.")
        return

    # 3) Create a DataFrame
    import pandas as pd
    df = pd.DataFrame(rows, columns=["Ticker", "Dev_20", "Dev_63"])

    # Convert Decimal to float
    df["Dev_20"] = df["Dev_20"].astype(float)
    df["Dev_63"] = df["Dev_63"].astype(float)

    # Compute z-scores
    df["Dev_20_Z"] = (df["Dev_20"] - df["Dev_20"].mean()) / df["Dev_20"].std()
    df["Dev_63_Z"] = (df["Dev_63"] - df["Dev_63"].mean()) / df["Dev_63"].std()

    # Filter stocks based on z_threshold
    extremes = df[
        (df["Dev_20_Z"].abs() >= z_threshold) | (df["Dev_63_Z"].abs() >= z_threshold)
    ]

    # 4) Write results to a file
    if not extremes.empty:
        file_name = os.path.join(config.RESULTS_DIR, f"extreme_volume_stocks_{last_date}.txt")
        with open(file_name, "w") as f:
            f.write(f"Extreme Volume Stocks for {last_date} (Z-Threshold: {z_threshold}):\n\n")
            for _, row in extremes.iterrows():
                f.write(
                    f"{row['Ticker']}: Dev_20_Z = {row['Dev_20_Z']:.2f}, "
                    f"Dev_63_Z = {row['Dev_63_Z']:.2f}\n"
                )
        logging.info(f"Extreme volume file created: {file_name}")
    else:
        logging.info(f"No stocks exceeded z-threshold {z_threshold} on {last_date}.")


def get_sp500_tickers() -> List[str]:
    """
    Example: read the list of S&P 500 tickers from Wikipedia.
    If you prefer, you can replace this with a hard-coded list or your own logic.
    """
    import requests
    from bs4 import BeautifulSoup
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            logging.error("Failed to fetch S&P 500 tickers from Wikipedia.")
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if not table:
            logging.error("No S&P 500 table found on Wikipedia.")
            return []
        rows = table.find_all("tr")[1:]
        tickers = []
        for row in rows:
            cols = row.find_all("td")
            if cols:
                ticker = cols[0].text.strip()
                ticker = ticker.replace(".", "-")
                tickers.append(ticker)
        return tickers
    except Exception as e:
        logging.error(f"Error fetching S&P 500 tickers: {e}")
        return []


def save_phases_breakdown(
    df_phases_daily: pd.DataFrame,
    df_phases_resampled: pd.DataFrame,
    output_file=os.path.join(config.RESULTS_DIR, "phases_breakdown.txt")
):
    """
    Save a detailed breakdown of phases to a text file.
    Includes both daily and resampled data.
    """
    with open(output_file, "w") as f:
        f.write("=== Phases Breakdown ===\n\n")

        # Daily Phases Breakdown
        f.write(">> Daily Phases Breakdown:\n")
        if not df_phases_daily.empty:
            latest_daily = df_phases_daily.iloc[-1]
            latest_date = df_phases_daily.index[-1].strftime("%Y-%m-%d")
            f.write(f"Date: {latest_date}\n")
            for phase, value in latest_daily.items():
                f.write(f"{phase}: {value:.2f}%\n")
        else:
            f.write("No daily phase data available.\n")

        f.write("\n")

        # Resampled Phases Breakdown
        f.write(">> Resampled Phases Breakdown:\n")
        if not df_phases_resampled.empty:
            for idx, row in df_phases_resampled.iterrows():
                date_str = idx.strftime("%Y-%m-%d")
                f.write(f"Date: {date_str}\n")
                for phase, value in row.items():
                    f.write(f"  {phase}: {value:.2f}%\n")
                f.write("\n")
        else:
            f.write("No resampled phase data available.\n")

    logging.info(f"Phases breakdown saved to {output_file}")


###############################################################################
# A helper to record the latest breadth & phases data + z-scores
###############################################################################
def save_latest_breadth_values(
    df_phases_daily: pd.DataFrame,
    all_indicators: Dict[str, pd.DataFrame],
    output_file="breadth_values.txt"
):
    """
    Writes the most recent phases breakdown and the most recent indicator
    values (including z-score & percentile) to a text file.
    """
    with open(output_file, "w") as f:
        f.write("=== Latest Phase Breakdown ===\n")
        if not df_phases_daily.empty:
            latest_phases = df_phases_daily.iloc[-1]
            date_str = df_phases_daily.index[-1].strftime("%Y-%m-%d")
            f.write(f"Date: {date_str}\n")
            for phase_col in latest_phases.index:
                f.write(f"{phase_col}: {latest_phases[phase_col]:.2f}%\n")
        else:
            f.write("No phase data available.\n")

        for indicator_name, df_data in all_indicators.items():
            if df_data.empty:
                continue
            f.write("\n")
            f.write(f"=== Latest Values for {indicator_name} ===\n")
            last_row = df_data.iloc[-1]
            date_str = df_data.index[-1].strftime("%Y-%m-%d")
            f.write(f"Date: {date_str}\n")

            for col in df_data.columns:
                col_series = df_data[col].dropna()
                if col_series.empty:
                    continue

                latest_val = last_row[col]
                col_mean = col_series.mean()
                col_std = col_series.std()
                if col_std != 0:
                    z_score = (latest_val - col_mean) / col_std
                else:
                    z_score = 0.0

                percentile = (col_series <= latest_val).mean() * 100

                f.write(f"{col} = {latest_val:.4f}, "
                        f"Z-score = {z_score:.4f}, "
                        f"Percentile = {percentile:.2f}%\n")


###############################################################################
# ADDED FOR PROFILING
###############################################################################
profiler = cProfile.Profile()
profiler.enable()

@atexit.register
def stop_profiler():
    profiler.disable()
    profile_stats_path = os.path.join(config.RESULTS_DIR, "profile_stats.txt")
    with open(profile_stats_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        stats.print_stats()


###############################################################################
# MAIN ENTRY POINT
###############################################################################
@measure_time
def main():
    """
    Main execution flow:
      1) Create tables if not found.
      2) Get a list of tickers (e.g. from S&P500).
      3) Fetch all existing data from DB for those tickers.
      4) Compute & store volume MAs (10, 20) => volume_ma_data
      5) Compute & store price % dev (50/200) => price_ma_deviation
      6) Compute & store volume % dev (20/63) => volume_ma_deviation
      7) Classify phases & store in DB => also do plots
      8) Compute other indicators & store => also do plots
      9) Export "extreme volume" file
      10) Save "latest breadth" text
      11) Detect/log changes (phase, SMA crosses)
    """

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    # Log setup
    log_path = os.path.join(config.RESULTS_DIR, "logs.txt")
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Additional event logger
    events_logger = logging.getLogger("events_logger")
    events_logger.setLevel(logging.INFO)
    events_log_path = os.path.join(config.RESULTS_DIR, "events_log.txt")
    fh = logging.FileHandler(events_log_path, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    events_logger.addHandler(fh)

    events_logger.info("START of main program")

    with DBConnectionManager() as conn:
        if conn is None:
            logging.error("Could not establish a database connection. Exiting.")
            events_logger.error("Could not establish DB connection. Exiting.")
            sys.exit(1)

        db_pool.monitor_pool()
        create_tables(conn)
        events_logger.info("Tables ensured in DB")

        tickers = get_sp500_tickers()
        if not tickers:
            logging.error("No tickers found. Exiting.")
            events_logger.error("No tickers found. Exiting.")
            sys.exit(1)

        log_memory_usage("Before batch_fetch_from_db")
        data_dict = batch_fetch_from_db(tickers, conn)
        db_pool.monitor_pool()
        log_memory_usage("After batch_fetch_from_db")

        if not data_dict:
            logging.warning("No data found in DB for these tickers.")
            events_logger.warning("No data found. Exiting.")
            sys.exit(0)

        # 4) Volume MAs (10, 20)
        compute_and_store_volume_mas(data_dict, conn)

        # 5) Price dev (50, 200)
        compute_and_store_price_ma_deviation(data_dict, conn)

        # 6) Volume dev (20, 63)
        compute_and_store_volume_ma_deviation(data_dict, conn)

        # 7) Phases classification
        phases = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
        df_phases_detail, df_phases_daily = classify_phases(
            data_dict,
            ma_short=config.MA_SHORT,
            ma_long=config.MA_LONG
        )
        log_rolling_cache_stats()

        if df_phases_daily.empty:
            logging.warning("Phase classification is empty. Skipping phase plots.")
            events_logger.warning("No phases data - skipping.")
        else:
            write_indicator_data_to_db(df_phases_daily, "phase_classification", conn)
            write_phase_details_to_db(df_phases_detail, conn)

            df_phases_daily.index = pd.to_datetime(df_phases_daily.index, errors="coerce")
            df_phases_resampled = df_phases_daily.resample(config.PHASE_PLOT_INTERVAL).last()

            save_phase_plots(phases, df_phases_resampled)
            save_phases_breakdown(df_phases_daily, df_phases_resampled)
            events_logger.info("Phase plots saved")

        # 8) Other indicators
        indicator_tasks = [
            ("adv_decline", compute_adv_decline),
            ("adv_decline_volume", compute_adv_decline_volume),
            ("new_high_low", compute_new_high_low),
            ("percent_above_ma", compute_percent_above_ma),
            ("mcclellan", compute_mcclellan),
            ("fear_greed", compute_index_of_fear_greed),
            ("trend_intensity_index", compute_trend_intensity_index),
            ("chaikin_volatility", compute_chaikin_volatility),
            ("chaikin_money_flow", compute_chaikin_money_flow),
            ("trin", compute_trin),
        ]

        computed_indicators = {}
        for indicator_name, func_ in indicator_tasks:
            events_logger.info(f"Starting computation for {indicator_name}")
            result_df_daily = run_indicator(
                indicator_name=indicator_name,
                data_dict=data_dict,
                compute_func=lambda cdf, vdf, hdf, ldf: func_(cdf, vdf, hdf, ldf)
            )

            if result_df_daily.empty:
                logging.warning(f"{indicator_name} returned empty. Skipping.")
                events_logger.warning(f"{indicator_name} is empty.")
                continue

            write_indicator_data_to_db(result_df_daily, indicator_name, conn)
            events_logger.info(f"Inserted {indicator_name} to DB")

            result_df_daily.index = pd.to_datetime(result_df_daily.index, errors="coerce")
            if not isinstance(result_df_daily.index, pd.DatetimeIndex) or result_df_daily.index.hasnans:
                logging.warning(f"{indicator_name} index invalid. Skipping plot.")
                continue

            result_df_resampled = result_df_daily.resample(config.INDICATOR_PLOT_INTERVAL).mean()
            save_indicator_plots(indicator_name, result_df_resampled)
            events_logger.info(f"Saved {indicator_name} plots")

            computed_indicators[indicator_name] = result_df_daily

        # 9) Extreme volume
        export_extreme_volumes(conn, z_threshold=2.0)

        # 10) Save latest breadth
        save_latest_breadth_values(
            df_phases_daily,
            computed_indicators,
            output_file=os.path.join(config.RESULTS_DIR, "breadth_values.txt")
        )
        events_logger.info("Saved latest breadth & phase data")

        # 11) Detect changes
        detect_and_log_changes(
            conn,
            phase_changes_file=os.path.join(config.RESULTS_DIR, "phase_changes.txt"),
            price_sma_changes_file=os.path.join(config.RESULTS_DIR, "price_sma_changes.txt")
        )
        events_logger.info("Logged phase/indicator changes")

        logging.info("All tasks completed successfully.")
        events_logger.info("END of main program")

if __name__ == "__main__":
    main()
