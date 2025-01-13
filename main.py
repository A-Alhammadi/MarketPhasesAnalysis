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
    """Create the price_data, indicator_data, and new phase_details tables if they do not exist."""
    cur = conn.cursor()
    
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

    # NEW: phase_details table
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
    import psycopg2.extras
    psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
    conn.commit()
    cur.close()

def detect_and_log_changes(conn, phase_changes_file="phase_changes.txt", price_sma_changes_file="price_sma_changes.txt"):
    """
    1) Detect any ticker that changes its phase (old -> new).
    2) Detect price crossing above/below SMA_50 or SMA_200.
    3) Detect golden/death crosses (SMA_50 crossing SMA_200).
    
    Results are written to separate files:
      - Phase changes: phase_changes_file
      - Price/SMA changes: price_sma_changes_file
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
        values += [None] * (5 - len(values))  # pad if fewer than 5
        values = values[:5]  # trim if more than 5
        records.append((indicator_name, dt_, *values))

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
    Fetch data for the given tickers from the database, starting from an extended earliest date
    to allow for proper moving average calculations. Trims data for plotting based on START_DATE.
    """
    data_dict = {}
    batch_size = 50
    max_workers = min(config.MAX_WORKERS, len(tickers)) if tickers else 1

    # Calculate earliest date based on the longest moving average window (e.g., 200 days)
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
                        # Fetch data starting from the earliest date
                        df = df[df.index >= earliest_date]
                        # Trim data for plotting (optional: after classification)
                        if config.END_DATE:
                            end_date = pd.to_datetime(config.END_DATE)
                            df = df[df.index <= end_date]

                        if not df.empty:
                            data_dict[ticker] = df
                        else:
                            logging.info(f"No data within the date range for {ticker}.")
                    else:
                        logging.info(f"No data found in DB for {ticker}.")
                except Exception as e:
                    logging.error(f"Error loading data for {ticker}: {e}")
    return data_dict

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
                # Some S&P500 tickers have periods - we often replace them with dash
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
            latest_daily = df_phases_daily.iloc[-1]  # Get the most recent daily row
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
# NEW: A helper to record the latest breadth & phases data + z-scores
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
    # Open in write mode
    with open(output_file, "w") as f:
        f.write("=== Latest Phase Breakdown ===\n")
        if not df_phases_daily.empty:
            latest_phases = df_phases_daily.iloc[-1]  # last row
            date_str = df_phases_daily.index[-1].strftime("%Y-%m-%d")
            f.write(f"Date: {date_str}\n")
            for phase_col in latest_phases.index:
                f.write(f"{phase_col}: {latest_phases[phase_col]:.2f}%\n")
        else:
            f.write("No phase data available.\n")

        # Now go through each indicator DF
        for indicator_name, df_data in all_indicators.items():
            if df_data.empty:
                continue
            f.write("\n")
            f.write(f"=== Latest Values for {indicator_name} ===\n")
            last_row = df_data.iloc[-1]  # last row
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

                # percentile: fraction of points <= latest_val
                percentile = (col_series <= latest_val).mean() * 100

                f.write(f"{col} = {latest_val:.4f}, "
                        f"Z-score = {z_score:.4f}, "
                        f"Percentile = {percentile:.2f}%\n")


###############################################################################
# ADDED FOR PROFILING
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
      4) Classify phases and save them in DB (indicator_data and phase_details).
      5) Compute indicators, save them in DB.
      6) Plot phases & indicators from whatever data is found.
      7) Detect and log phase changes, price crossings, and golden/death crosses.
      8) Save the "latest" breadth indicator values & phases breakdown
         (plus z-scores & percentiles) to a text file.
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

        # Ensure tables exist
        db_pool.monitor_pool()
        create_tables(conn)
        events_logger.info("Tables ensured in DB")

        # Get list of tickers (modify `get_sp500_tickers` as needed for your tickers)
        tickers = get_sp500_tickers()
        if not tickers:
            logging.error("No tickers found. Exiting.")
            events_logger.error("No tickers found. Exiting.")
            sys.exit(1)

        log_memory_usage("Before batch_fetch_from_db")
        data_dict = batch_fetch_from_db(tickers, conn)
        df_phases_detail = classify_phases(data_dict, ma_short=config.MA_SHORT, ma_long=config.MA_LONG)
        db_pool.monitor_pool()
        log_memory_usage("After batch_fetch_from_db")

        if not data_dict:
            logging.warning("No data was found in the DB for these tickers.")
            events_logger.warning("No data found for any tickers in DB.")
            sys.exit(0)

        # 1. Classification into phases (returns detail + summary DataFrames)
        phases = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
        df_phases_detail, df_phases_daily = classify_phases(
            data_dict,
            ma_short=config.MA_SHORT,
            ma_long=config.MA_LONG
        )
        log_rolling_cache_stats()

        if df_phases_daily.empty:
            logging.warning("Phase classification returned empty. Skipping phase plots.")
            events_logger.warning("No phases data - skipping plots.")
        else:
            # 2. Save daily aggregated phase data to indicator_data (as before)
            write_indicator_data_to_db(df_phases_daily, "phase_classification", conn)

            # 3. Save per-ticker phase details to phase_details (NEW)
            write_phase_details_to_db(df_phases_detail, conn)

            # 4. Resample for plotting (e.g., weekly)
            df_phases_daily.index = pd.to_datetime(df_phases_daily.index, errors="coerce")
            df_phases_resampled = df_phases_daily.resample(config.PHASE_PLOT_INTERVAL).last()

            # 5. Plot phases
            save_phase_plots(phases, df_phases_resampled)
            save_phases_breakdown(df_phases_daily, df_phases_resampled)
            events_logger.info("Phase plots saved")

        # 6. Indicator computations (as in your original code)
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

        # Keep references to each indicator's output
        computed_indicators = {}

        for indicator_name, compute_func in indicator_tasks:
            events_logger.info(f"Starting computation for {indicator_name}")
            result_df_daily = run_indicator(
                indicator_name=indicator_name,
                data_dict=data_dict,
                compute_func=lambda cdf, vdf, hdf, ldf: compute_func(cdf, vdf, hdf, ldf)
            )

            if result_df_daily.empty:
                logging.warning(f"{indicator_name} returned an empty DataFrame, skipping.")
                events_logger.warning(f"{indicator_name} is empty, skipping.")
                continue

            # Store indicator result in DB
            write_indicator_data_to_db(result_df_daily, indicator_name, conn)
            events_logger.info(f"Inserted {indicator_name} results into DB")

            # Plot indicator
            result_df_daily.index = pd.to_datetime(result_df_daily.index, errors="coerce")
            if not isinstance(result_df_daily.index, pd.DatetimeIndex):
                logging.warning(f"{indicator_name} index could not be converted to DatetimeIndex, skipping.")
                continue
            if result_df_daily.index.hasnans:
                logging.warning(f"{indicator_name} index has NaT values, skipping.")
                continue

            result_df_resampled = result_df_daily.resample(config.INDICATOR_PLOT_INTERVAL).mean()
            save_indicator_plots(indicator_name, result_df_resampled)
            events_logger.info(f"Saved plots for {indicator_name}")

            # Keep the daily DataFrame for the "latest values" report
            computed_indicators[indicator_name] = result_df_daily

        # 7. Save the latest breadth values & phases breakdown (as before)
        save_latest_breadth_values(
            df_phases_daily,
            computed_indicators,
            output_file=os.path.join(config.RESULTS_DIR, "breadth_values.txt")
        )
        events_logger.info("Saved latest breadth & phase values to 'breadth_values.txt'")

        # 8. Detect and log changes (NEW)
        detect_and_log_changes(
            conn,
            phase_changes_file=os.path.join(config.RESULTS_DIR, "phase_changes.txt"),
            price_sma_changes_file=os.path.join(config.RESULTS_DIR, "price_sma_changes.txt")
        )

        events_logger.info("Logged phase/indicator changes to 'changes_detected.txt'")

        logging.info("All tasks completed successfully.")
        events_logger.info("END of main program")

if __name__ == "__main__":
    main()
