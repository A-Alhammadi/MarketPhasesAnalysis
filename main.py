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
    """Create the price_data and indicator_data tables if they do not exist."""
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

    conn.commit()
    cur.close()


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
    Fetch data for the given tickers from the database and filter by date range.
    """
    data_dict = {}
    batch_size = 50
    max_workers = min(config.MAX_WORKERS, len(tickers)) if tickers else 1

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
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
                        # Apply date filtering here
                        if config.START_DATE:
                            start_date = pd.to_datetime(config.START_DATE)
                            df = df[df.index >= start_date]
                        if config.END_DATE:
                            end_date = pd.to_datetime(config.END_DATE)
                            df = df[df.index <= end_date]

                        if not df.empty:
                            data_dict[ticker] = df
                        else:
                            logging.info(f"No data within the specified date range for {ticker}.")
                    else:
                        logging.info(f"DB has no data for {ticker}. Skipping.")
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


###############################################################################
# MAIN ENTRY POINT
###############################################################################
# ADDED FOR PROFILING
profiler = cProfile.Profile()
profiler.enable()

@atexit.register
def stop_profiler():
    profiler.disable()
    with open("profile_stats.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        stats.print_stats()


@measure_time
def main():
    """
    Main execution flow:
      1) Create tables if not found.
      2) Get a list of tickers (e.g. from S&P500).
      3) Fetch all existing data from DB for those tickers.
      4) Classify phases and save them in DB (indicator_data).
      5) Compute indicators, save them in DB.
      6) Plot phases & indicators from whatever data is found.
    """

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    log_path = os.path.join(config.RESULTS_DIR, "logs.txt")
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    with DBConnectionManager() as conn:
        if conn is None:
            logging.error("Could not establish a database connection. Exiting.")
            sys.exit(1)

        # Make sure the tables exist
        db_pool.monitor_pool()
        create_tables(conn)

        # You can change this to your own fixed list or logic
        tickers = get_sp500_tickers()
        if not tickers:
            logging.error("No tickers found. Exiting.")
            sys.exit(1)

        log_memory_usage("Before batch_fetch_from_db")
        data_dict = batch_fetch_from_db(tickers, conn)
        db_pool.monitor_pool()
        log_memory_usage("After batch_fetch_from_db")

        if not data_dict:
            logging.warning("No data was found in the DB for these tickers.")
            logging.warning("Will not perform any classification or indicator plots.")
            sys.exit(0)

        # Classification into phases
        phases = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
        df_phases_daily = classify_phases(
            data_dict,
            ma_short=config.MA_SHORT,
            ma_long=config.MA_LONG
        )
        log_rolling_cache_stats()

        if df_phases_daily.empty:
            logging.warning("Phase classification returned empty. Skipping phase plots.")
        else:
            # Save daily phase classification in DB
            write_indicator_data_to_db(df_phases_daily, "phase_classification", conn)

            # Resample for plotting (e.g. weekly)
            df_phases_daily.index = pd.to_datetime(df_phases_daily.index, errors="coerce")
            df_phases_resampled = df_phases_daily.resample(config.PHASE_PLOT_INTERVAL).last()

            save_phase_plots(phases, df_phases_resampled)

        # Indicator computations
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

        for indicator_name, compute_func in indicator_tasks:
            result_df_daily = run_indicator(
                indicator_name=indicator_name,
                data_dict=data_dict,
                compute_func=lambda cdf, vdf, hdf, ldf: compute_func(cdf, vdf, hdf, ldf)
            )

            if result_df_daily.empty:
                logging.warning(f"{indicator_name} returned an empty DataFrame, skipping.")
                continue

            # Store indicator result in DB
            write_indicator_data_to_db(result_df_daily, indicator_name, conn)

            # Plot
            result_df_daily.index = pd.to_datetime(result_df_daily.index, errors="coerce")
            if not isinstance(result_df_daily.index, pd.DatetimeIndex):
                logging.warning(
                    f"{indicator_name} index could not be converted to a DatetimeIndex, skipping."
                )
                continue
            if result_df_daily.index.hasnans:
                logging.warning(
                    f"{indicator_name} index has NaT (missing) values, skipping."
                )
                continue

            result_df_resampled = result_df_daily.resample(config.INDICATOR_PLOT_INTERVAL).mean()
            save_indicator_plots(indicator_name, result_df_resampled)

        logging.info("All tasks completed successfully.")


if __name__ == "__main__":
    main()
