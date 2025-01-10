# main.py

import os
import sys
import logging
import requests
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

# Third-party libs
import psycopg2
import psycopg2.extras
import yfinance as yf
from sqlalchemy import create_engine

# Initialize SQLAlchemy engine once in the script
DATABASE_URL = f"postgresql://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
engine = create_engine(DATABASE_URL)

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

# Circuit Breaker for external fetches
class CircuitBreaker:
    def __init__(self, max_consecutive_errors=3, reset_timeout=300):  # Reset after 300 seconds
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0
        self.open = False
        self.last_opened = None
        self.reset_timeout = reset_timeout

    def call(self, func, *args, **kwargs):
        if self.open:
            if self.last_opened and (datetime.now() - self.last_opened).total_seconds() > self.reset_timeout:
                logging.info("Circuit breaker auto-resetting.")
                self.open = False
                self.consecutive_errors = 0
            else:
                logging.warning("Circuit breaker is open; skipping call.")
                return None
        try:
            result = func(*args, **kwargs)
            self.consecutive_errors = 0  # Reset on success
            return result
        except Exception as e:
            self.consecutive_errors += 1
            logging.error(f"CircuitBreaker: Error during call: {e}")
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.open = True
                self.last_opened = datetime.now()
                logging.error(f"CircuitBreaker: Opened after {self.consecutive_errors} errors.")
            raise e

circuit_breaker = CircuitBreaker(max_consecutive_errors=3)


def check_memory_usage():
    import psutil
    usage = psutil.virtual_memory()
    if usage.percent > config.MAX_MEMORY_PERCENT:
        logging.warning(
            f"Memory usage {usage.percent:.1f}% exceeds threshold {config.MAX_MEMORY_PERCENT}%. "
            "Consider releasing memory or skipping further processing."
        )


def exponential_backoff_fetch(func, *args, max_attempts=3, base_delay=1.0, factor=2.0, jitter=0.3, **kwargs):
    attempts = 0
    delay = base_delay
    while attempts < max_attempts:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempts += 1
            if attempts >= max_attempts:
                raise
            sleep_time = delay + (random.random() * jitter)
            logging.warning(
                f"Retrying after error: {e}. "
                f"Attempt {attempts}/{max_attempts}. "
                f"Sleeping {sleep_time:.2f}s."
            )
            time.sleep(sleep_time)
            delay *= factor


def create_tables(conn):
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


def write_ticker_data_to_db(ticker: str, df: pd.DataFrame, conn):
    if df.empty:
        return
    cur = conn.cursor()
    records = []
    for date_, row in df.iterrows():
        records.append(
            (
                ticker,
                date_.date(),
                float(row["Open"]) if not pd.isna(row["Open"]) else None,
                float(row["High"]) if not pd.isna(row["High"]) else None,
                float(row["Low"]) if not pd.isna(row["Low"]) else None,
                float(row["Close"]) if not pd.isna(row["Close"]) else None,
                int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
            )
        )

    insert_query = """
        INSERT INTO price_data
            (ticker, trade_date, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (ticker, trade_date) DO UPDATE
        SET open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
    """
    psycopg2.extras.execute_values(cur, insert_query, records, page_size=1000)
    conn.commit()
    cur.close()


def read_ticker_data_from_db(ticker: str, conn) -> pd.DataFrame:
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
        df.index.name = None
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype("float32")
        df["Volume"] = df["Volume"].astype("int64", errors="ignore")
    return df

def get_db_min_max_date(conn) -> Tuple[datetime, datetime]:
    min_date, max_date = None, None
    try:
        cur = conn.cursor()
        cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM price_data;")
        row = cur.fetchone()
        if row:
            min_date, max_date = row
        cur.close()
    except Exception as e:
        logging.error(f"Error getting min/max date from DB: {e}")
    return min_date, max_date


def get_last_business_day(ref_date: datetime) -> datetime:
    while ref_date.weekday() >= 5:
        ref_date -= timedelta(days=1)
    return ref_date


def get_last_completed_trading_day() -> datetime:
    now = datetime.now()
    while now.weekday() >= 5:
        now -= timedelta(days=1)
    # If it's before 16:00, consider the previous business day
    if now.hour < 16:
        now -= timedelta(days=1)
        while now.weekday() >= 5:
            now -= timedelta(days=1)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def batch_fetch_from_db(tickers: List[str], conn, batch_size=50) -> Dict[str, pd.DataFrame]:
    data_dict = {}
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
                    data = future.result()
                    if not data.empty:
                        data_dict[ticker] = data
                except Exception as e:
                    logging.error(f"Error loading data for {ticker}: {e}")
    return data_dict


def fetch_ticker_data(ticker, start, end):
    if start is None or end is None:
        return None
    check_memory_usage()
    data = exponential_backoff_fetch(
        circuit_breaker.call,
        yf.download,
        ticker,
        start=start,
        end=end + timedelta(days=1),
        interval=config.DATA_FETCH_INTERVAL,
        progress=False,
        max_attempts=3,
    )
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(level=1)
    if "Adj Close" in data.columns:
        data.drop(columns=["Adj Close"], inplace=True)
    needed_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed_cols:
        if col not in data.columns:
            logging.warning(f"Ticker {ticker} missing {col} after fetch. Skipping chunk.")
            return None
    for col in ["Open", "High", "Low", "Close"]:
        data[col] = data[col].astype("float32", errors="ignore")
    data["Volume"] = data["Volume"].astype("int64", errors="ignore")
    return data[needed_cols]


def fetch_and_write_all_tickers(tickers: List[str], start_date, end_date, conn) -> Tuple[Dict[str, pd.DataFrame], bool]:
    BATCH_SIZE = 50
    db_min, db_max = get_db_min_max_date(conn)
    cfg_start = pd.to_datetime(start_date) if start_date else None
    if end_date is None:
        cfg_end = get_last_completed_trading_day()
    else:
        cfg_end = pd.to_datetime(end_date)
        cfg_end = get_last_business_day(cfg_end)

    if db_min and db_max and cfg_start and cfg_end:
        # If DB covers entire date range already
        if pd.Timestamp(cfg_start) >= pd.Timestamp(db_min) and pd.Timestamp(cfg_end) <= pd.Timestamp(db_max):
            logging.info("Database covers the entire date range; using existing data only.")
            return batch_fetch_from_db(tickers, conn), False

    needed_intervals = []
    if db_min is None or db_max is None:
        needed_intervals.append((cfg_start, cfg_end))
    else:
        if cfg_start < pd.Timestamp(db_min):
            needed_intervals.append((cfg_start, pd.Timestamp(db_min) - pd.Timedelta(days=1)))
        if cfg_end > pd.Timestamp(db_max):
            needed_intervals.append((pd.Timestamp(db_max) + pd.Timedelta(days=1), cfg_end))

    data_dict = {}
    new_data_found = False

    def process_ticker_batch(ticker_batch):
        batch_data = {}
        with DBConnectionManager() as conn_local:
            for tk in ticker_batch:
                try:
                    existing_data = read_ticker_data_from_db(tk, conn_local)
                    for (mstart, mend) in needed_intervals:
                        if mstart is not None and mend is not None and mstart <= mend:
                            fetched_part = fetch_ticker_data(tk, mstart, mend)
                            # Filter out empty DataFrames before concatenating
                            if fetched_part is not None and not fetched_part.empty:
                                existing_data = pd.concat([existing_data, fetched_part])
                                nonlocal new_data_found
                                new_data_found = True
                                del fetched_part
                                gc.collect()
                    if not existing_data.empty and "Close" in existing_data.columns:
                        existing_data.sort_index(inplace=True)
                        write_ticker_data_to_db(tk, existing_data, conn_local)
                        batch_data[tk] = existing_data
                    del existing_data
                    gc.collect()
                except Exception as e:
                    logging.error(f"Error processing ticker {tk}: {e}")
                    continue
        return batch_data

    max_workers = min(config.MAX_WORKERS, len(tickers)) if tickers else 1
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i: i + BATCH_SIZE]
        logging.info(
            f"Processing batch of {len(batch)} tickers "
            f"({i+1} to {i+len(batch)} of {len(tickers)})"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_size = 5
            chunks = [batch[j: j + chunk_size] for j in range(0, len(batch), chunk_size)]
            future_to_chunk = {
                executor.submit(process_ticker_batch, chunk): chunk
                for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                sub_chunk = future_to_chunk[future]
                try:
                    batch_results = future.result()
                    data_dict.update(batch_results)
                except Exception as e:
                    logging.error(f"Error processing chunk {sub_chunk}: {e}")
                    continue

    return data_dict, new_data_found


def merge_new_indicator_data(new_df: pd.DataFrame, indicator_name: str, conn):
    if new_df.empty:
        logging.info(f"No new data for indicator {indicator_name}")
        return
    cur = conn.cursor()
    records = []
    for dt_, row in new_df.iterrows():
        dt_ = dt_.date()
        values = row.values.tolist()
        values += [None] * (5 - len(values))
        values = values[:5]
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


def get_sp500_tickers() -> List[str]:
    from bs4 import BeautifulSoup
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = exponential_backoff_fetch(
            circuit_breaker.call,
            requests.get,
            url,
            max_attempts=3,
        )
        if resp is None:
            return []
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

        db_pool.monitor_pool()
        create_tables(conn)

        tickers = get_sp500_tickers()
        if not tickers:
            logging.error("No tickers found. Exiting.")
            sys.exit(1)

        log_memory_usage("Before fetch_and_write_all_tickers")
        data_dict, new_data_found = fetch_and_write_all_tickers(
            tickers=tickers,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            conn=conn
        )
        db_pool.monitor_pool()
        log_memory_usage("After fetch_and_write_all_tickers")

        # Classification into phases
        phases = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
        df_phases_daily = classify_phases(
            data_dict,
            ma_short=config.MA_SHORT,
            ma_long=config.MA_LONG
        )
        log_rolling_cache_stats()

        # Ensure the index is a DatetimeIndex before resampling
        df_phases_daily.index = pd.to_datetime(df_phases_daily.index, errors="coerce")
        if not isinstance(df_phases_daily.index, pd.DatetimeIndex):
            raise TypeError("Index is not a DatetimeIndex after conversion!")

        df_phases_resampled = df_phases_daily.resample(config.PHASE_PLOT_INTERVAL).last()

        if not new_data_found:
            logging.info("No new data was fetched. Creating missing plots and exiting.")
            # Generate missing phase plots
            save_phase_plots(phases, df_phases_resampled)

            # Generate missing indicator plots
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
                result_df_daily = compute_func(
                    data_dict.get("Close", pd.DataFrame()), 
                    data_dict.get("Volume", pd.DataFrame()),
                    data_dict.get("High", pd.DataFrame()),
                    data_dict.get("Low", pd.DataFrame())
                )

                # Ensure the index is a DatetimeIndex before resampling
                if result_df_daily.empty:
                    logging.warning(f"{indicator_name} returned an empty DataFrame, skipping.")
                    continue

                result_df_daily.index = pd.to_datetime(result_df_daily.index, errors="coerce")
                if not isinstance(result_df_daily.index, pd.DatetimeIndex):
                    logging.warning(f"{indicator_name} index could not be converted to a DatetimeIndex, skipping.")
                    continue
                if result_df_daily.index.hasnans:
                    logging.warning(f"{indicator_name} index has NaT (missing) values, skipping.")
                    continue

                result_df_resampled = result_df_daily.resample(config.INDICATOR_PLOT_INTERVAL).mean()
                save_indicator_plots(indicator_name, result_df_resampled)
            return

        # Store new phase data
        merge_new_indicator_data(df_phases_daily, "phase_classification", conn)
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

            # Ensure the index is a DatetimeIndex before resampling
            if result_df_daily.empty:
                logging.warning(f"{indicator_name} returned an empty DataFrame, skipping.")
                continue

            result_df_daily.index = pd.to_datetime(result_df_daily.index, errors="coerce")
            if not isinstance(result_df_daily.index, pd.DatetimeIndex):
                logging.warning(f"{indicator_name} index could not be converted to a DatetimeIndex, skipping.")
                continue
            if result_df_daily.index.hasnans:
                logging.warning(f"{indicator_name} index has NaT (missing) values, skipping.")
                continue

            result_df_resampled = result_df_daily.resample(config.INDICATOR_PLOT_INTERVAL).mean()
            save_indicator_plots(indicator_name, result_df_resampled)

        logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
