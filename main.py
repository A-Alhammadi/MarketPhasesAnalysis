import os
import sys
import logging
import time
import requests
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import psycopg2
import psycopg2.extras

import config
from phase_analysis import classify_phases
from indicators.adv_decline import compute_adv_decline
from indicators.adv_decline_volume import compute_adv_decline_volume
from indicators.new_high_low import compute_new_high_low
from indicators.percent_above_ma import compute_percent_above_ma

# Additional indicators
from indicators.extra_indicators import (
    compute_mcclellan,
    compute_index_of_fear_greed,
    compute_trend_intensity_index,
    compute_chaikin_volatility,
    compute_chaikin_money_flow
)
from indicators.trin import compute_trin

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='a',  # append instead of overwriting
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_db_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASS
    )

def create_tables():
    """
    Creates the necessary tables in PostgreSQL if they do not already exist.
    We'll store all historical price data in 'price_data'.
    We'll store example indicator data in 'indicator_data'.
    """
    conn = None
    cur = None
    try:
        conn = get_db_connection()
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

        conn.commit()
    except Exception as e:
        logging.error(f"Error creating tables: {e}")
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

def read_ticker_data_from_db(ticker):
    """
    Reads any existing data for a ticker from the 'price_data' table.
    Returns a DataFrame with Date as index, or empty if none.
    """
    query = """
        SELECT trade_date, open, high, low, close, volume
        FROM price_data
        WHERE ticker = %s
        ORDER BY trade_date
    """
    conn = None
    df = pd.DataFrame()
    try:
        conn = get_db_connection()
        df = pd.read_sql(query, conn, params=(ticker,))
        if df.empty:
            return df

        df.set_index("trade_date", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index.name = None
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
    except Exception as e:
        logging.error(f"Error reading data for {ticker} from DB: {e}")
    finally:
        if conn is not None:
            conn.close()
    return df

def write_ticker_data_to_db(ticker, df):
    """
    Writes the given DataFrame into 'price_data' table in PostgreSQL, upserting rows.
    """
    if df.empty:
        return

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        records = []
        for date_, row in df.iterrows():
            records.append((
                ticker,
                date_.date(),
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                int(row["Volume"]) if not pd.isna(row["Volume"]) else None
            ))

        insert_query = """
            INSERT INTO price_data (ticker, trade_date, open, high, low, close, volume)
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
    except Exception as e:
        logging.error(f"Error writing data for {ticker} to DB: {e}")
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

def get_db_min_max_date():
    """
    Returns (min_date, max_date) across ALL tickers from the price_data table.
    If table is empty, returns (None, None).
    """
    conn = None
    min_date, max_date = None, None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM price_data;")
        row = cur.fetchone()
        if row is not None:
            min_date, max_date = row
    except Exception as e:
        logging.error(f"Error getting global min/max date from DB: {e}")
    finally:
        if conn:
            conn.close()
    return min_date, max_date

def save_indicator_plots(indicator_name, df, output_dir="results"):
    """
    Save plots for each column in the indicator DataFrame (e.g. 'AdvanceDeclineLine',
    'NHNL_Diff', etc.), automatically removing leading constant segments.
    """
    if df.empty:
        return
    for col in df.columns:
        # Drop all-NaN
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        # Remove leading constant portion so that the initial flat line is cut
        first_val = col_data.iloc[0]
        changed_mask = col_data.ne(first_val)
        if changed_mask.any():
            first_change_idx = changed_mask.idxmax()
            col_data = col_data.loc[first_change_idx:]

        plt.figure(figsize=(10, 6))
        plt.plot(col_data.index, col_data.values, label=col)
        plt.title(f"{indicator_name} - {col}")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        filename = os.path.join(output_dir, f"{indicator_name}_{col}.png")
        plt.savefig(filename, dpi=100)
        plt.close()

def save_plot(phase, df_phases_merged, filename):
    """
    Save a plot for a specific phase, overwriting the file each run.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df_phases_merged.index, df_phases_merged[phase], label=phase)
    plt.title(f"{phase} Phase % Over Time")
    plt.xlabel("Date")
    plt.ylabel("Percentage of Total Stocks")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=100)
    plt.close()

def get_sp500_tickers():
    """
    Fetches the current S&P 500 constituents from Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url)
    if resp.status_code != 200:
        logging.error("Failed to fetch S&P 500 tickers from Wikipedia.")
        return []
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    if not table:
        logging.error("No S&P 500 table found on Wikipedia.")
        return []
    rows = table.find_all('tr')[1:]
    tickers = []
    for row in rows:
        cols = row.find_all('td')
        if cols:
            ticker = cols[0].text.strip()
            ticker = ticker.replace('.', '-')  # Adjust for Yahoo format
            tickers.append(ticker)
    return tickers

def filter_trading_days(start, end):
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start, end_date=end)
    trading_days = schedule.index.normalize()
    return trading_days

def get_last_business_day(ref_date):
    while ref_date.weekday() >= 5:  # Saturday=5, Sunday=6
        ref_date -= timedelta(days=1)
    return ref_date

def get_last_completed_trading_day():
    now = datetime.now()
    while now.weekday() >= 5:
        now -= timedelta(days=1)
    if now.hour < 16:
        now -= timedelta(days=1)
        while now.weekday() >= 5:
            now -= timedelta(days=1)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)

def merge_new_indicator_data(new_df, indicator_name):
    """
    Store indicator data in 'indicator_data' table, up to 5 numeric columns.
    """
    if new_df.empty:
        logging.info(f"No new data for indicator {indicator_name}.")
        return
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        columns = list(new_df.columns)
        columns = columns[:5]  # up to 5 columns

        for date_, row in new_df.iterrows():
            date_ = date_.date()
            values = row.values.tolist()
            values = values + [None]*(5 - len(values))  # pad if needed
            values = values[:5]

            insert_query = """
                INSERT INTO indicator_data
                    (indicator_name, data_date, value1, value2, value3, value4, value5)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (indicator_name, data_date) DO UPDATE
                SET value1 = EXCLUDED.value1,
                    value2 = EXCLUDED.value2,
                    value3 = EXCLUDED.value3,
                    value4 = EXCLUDED.value4,
                    value5 = EXCLUDED.value5
            """
            try:
                cur.execute(insert_query, (indicator_name, date_, *values))
            except Exception as e:
                logging.error(f"Error upserting indicator {indicator_name} for date {date_}: {e}")
        conn.commit()
    except Exception as e:
        logging.error(f"Error merging indicator data for {indicator_name}: {e}")
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

def fetch_and_write_all_tickers(tickers, start_date, end_date):
    """
    Checks the database for existing data and fetches missing data only if needed.
    If the data range in the database already covers the configured range, skips fetching.
    Returns:
        data_dict: A dictionary of DataFrames for each ticker.
        new_data_found: Boolean indicating whether new data was fetched or updated.
    """
    db_min, db_max = get_db_min_max_date()

    cfg_start = pd.to_datetime(start_date) if start_date else None
    if end_date is None:
        cfg_end = get_last_completed_trading_day()
    else:
        cfg_end = pd.to_datetime(end_date)
        cfg_end = get_last_business_day(cfg_end)

    # Quick check: If the database already has the full range, skip fetching
    if db_min and db_max:
        if pd.Timestamp(cfg_start) >= pd.Timestamp(db_min) and pd.Timestamp(cfg_end) <= pd.Timestamp(db_max):
            logging.info("Database already contains data for the entire range. Skipping data fetch.")
            data_dict = {}
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(read_ticker_data_from_db, ticker): ticker
                    for ticker in tickers
                }
                for future in futures:
                    tck = futures[future]
                    try:
                        data_dict[tck] = future.result()
                    except Exception as e:
                        logging.error(f"Error loading data for {tck}: {e}")
            return data_dict, False  # No new data was fetched

    # Determine missing intervals
    logging.info("Fetching missing data from Yahoo Finance...")
    needed_intervals = []
    if db_min is None or db_max is None:
        needed_intervals.append((cfg_start, cfg_end))
    else:
        if cfg_start < pd.Timestamp(db_min):
            needed_intervals.append((cfg_start, pd.Timestamp(db_min) - pd.Timedelta(days=1)))
        if cfg_end > pd.Timestamp(db_max):
            needed_intervals.append((pd.Timestamp(db_max) + pd.Timedelta(days=1), cfg_end))

    # Fetch missing data
    data_dict = {}
    new_data_found = False
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(read_ticker_data_from_db, ticker): ticker
            for ticker in tickers
        }
        for future in futures:
            tck = futures[future]
            try:
                existing_data = future.result()
                for mstart, mend in needed_intervals:
                    if mstart <= mend:
                        fetched_part = yf.download(
                            tck,
                            start=mstart,
                            end=mend + timedelta(days=1),
                            interval="1d",
                            progress=False
                        )
                        if not fetched_part.empty:
                            # Normalize columns and drop unused ones
                            if isinstance(fetched_part.columns, pd.MultiIndex):
                                fetched_part.columns = fetched_part.columns.droplevel(level=1)
                            if "Adj Close" in fetched_part.columns:
                                fetched_part.drop(columns=["Adj Close"], inplace=True)
                            fetched_part = fetched_part[["Open", "High", "Low", "Close", "Volume"]]
                            existing_data = pd.concat([existing_data, fetched_part])
                            new_data_found = True  # New data was fetched
                if "Close" in existing_data.columns:
                    existing_data.sort_index(inplace=True)
                    write_ticker_data_to_db(tck, existing_data)
                    data_dict[tck] = existing_data
                else:
                    logging.warning(f"Ticker {tck} is missing 'Close' column. Skipping.")
            except Exception as e:
                logging.error(f"Error processing ticker {tck}: {e}")

    return data_dict, new_data_found


def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    log_path = os.path.join("results", "logs.txt")
    setup_logging(log_path)

    create_tables()

    tickers = get_sp500_tickers()
    if not tickers:
        logging.error("No tickers found. Exiting.")
        sys.exit(1)

    # Fetch or fill missing data based on DB's global min/max + config
    data_dict, new_data_found = fetch_and_write_all_tickers(
        tickers=tickers,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )

    # Exit early if no new data was fetched
    if not new_data_found:
        logging.info("No new data was fetched. Exiting without updating plots.")
        return  # Exit the program early

    # Phase classification
    from phase_analysis import classify_phases
    df_phases = classify_phases(data_dict, ma_short=config.MA_SHORT, ma_long=config.MA_LONG)
    merge_new_indicator_data(df_phases, "phase_classification")

    # Phase plots
    if not df_phases.empty:
        phases = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
        for phase in phases:
            filename = os.path.join("results", f"phase_{phase.lower()}.png")
            save_plot(phase, df_phases, filename)
    else:
        logging.error("No valid phase data to plot.")

    # Indicators
    from indicators.adv_decline import compute_adv_decline
    from indicators.adv_decline_volume import compute_adv_decline_volume
    from indicators.new_high_low import compute_new_high_low
    from indicators.percent_above_ma import compute_percent_above_ma

    # Additional indicators
    from indicators.extra_indicators import (
        compute_mcclellan,
        compute_index_of_fear_greed,
        compute_trend_intensity_index,
        compute_chaikin_volatility,
        compute_chaikin_money_flow
    )
    from indicators.trin import compute_trin

    indicator_tasks = [
        (compute_adv_decline, "adv_decline"),
        (compute_adv_decline_volume, "adv_decline_volume"),
        (compute_new_high_low, "new_high_low"),
        (compute_percent_above_ma, "percent_above_ma"),
        (compute_mcclellan, "mcclellan"),
        (compute_index_of_fear_greed, "fear_greed"),
        (compute_trend_intensity_index, "trend_intensity_index"),
        (compute_chaikin_volatility, "chaikin_volatility"),
        (compute_chaikin_money_flow, "chaikin_money_flow"),
        (compute_trin, "trin"),
    ]

    with ThreadPoolExecutor() as executor:
        for compute_func, indicator_name in indicator_tasks:
            result_df = compute_func(data_dict)
            executor.submit(merge_new_indicator_data, result_df, indicator_name)
            executor.submit(save_indicator_plots, indicator_name, result_df, "results")

    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
