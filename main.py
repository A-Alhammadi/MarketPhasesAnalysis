# main.py

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


def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='a',  # append instead of overwriting
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
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
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASS
        )
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

def get_db_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASS
    )

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

def fetch_missing_data_for_ticker(ticker, existing_df, config_start_date, config_end_date=None, retries=3):
    """
    Fetch only missing data for a ticker from Yahoo, flatten columns, drop 'Adj Close', etc.
    """
    start_dt = pd.to_datetime(config_start_date)
    if config_end_date is None:
        end_dt = get_last_completed_trading_day()
    else:
        end_dt = pd.to_datetime(config_end_date)
        end_dt = get_last_business_day(end_dt)

    if existing_df.empty:
        logging.info(f"{ticker}: No DB data. Will fetch from {start_dt.date()} to {end_dt.date()}.")
        missing_intervals = [(start_dt, end_dt)]
    else:
        earliest_local = existing_df.index.min()
        latest_local = existing_df.index.max()
        missing_intervals = []

        if start_dt < earliest_local:
            fetch_end = earliest_local - timedelta(days=1)
            if start_dt <= fetch_end:
                trading_days = filter_trading_days(start_dt, fetch_end)
                if not trading_days.empty:
                    missing_intervals.append((trading_days.min(), trading_days.max()))

        if end_dt > latest_local:
            fetch_start = latest_local + timedelta(days=1)
            if fetch_start <= end_dt:
                trading_days = filter_trading_days(fetch_start, end_dt)
                if not trading_days.empty:
                    missing_intervals.append((trading_days.min(), trading_days.max()))

    if not missing_intervals:
        logging.info(f"{ticker}: No missing intervals. Skipping Yahoo download.")
        return existing_df

    logging.info(f"{ticker}: Missing intervals {missing_intervals}")
    all_new_parts = []

    for (m_start, m_end) in missing_intervals:
        if m_start > m_end:
            continue
        success = False
        new_data_part = pd.DataFrame()

        for attempt in range(retries):
            try:
                data = yf.download(
                    ticker,
                    start=m_start,
                    end=m_end + timedelta(days=1),
                    interval="1d",
                    progress=False
                )
                if not data.empty:
                    data.index = pd.to_datetime(data.index).tz_localize(None)

                    # Flatten multi-level columns
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(level=1)

                    # Drop "Adj Close" if present
                    if "Adj Close" in data.columns:
                        data.drop(columns=["Adj Close"], inplace=True)

                    # Keep only O/H/L/C/Volume
                    new_data_part = data[["Open", "High", "Low", "Close", "Volume"]].copy()

                success = True
                break
            except Exception as exc:
                logging.error(f"{ticker}: Error fetching data ({m_start.date()} - {m_end.date()}): {exc} (attempt {attempt+1}/{retries})")
                time.sleep(1)

        if not success:
            logging.warning(f"{ticker}: Failed to fetch data for {m_start.date()} - {m_end.date()}")
        else:
            all_new_parts.append(new_data_part)

    if all_new_parts:
        combined_new_data = pd.concat(all_new_parts)
        combined = pd.concat([existing_df, combined_new_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        return combined
    else:
        return existing_df

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

def fetch_and_write_ticker_data(ticker, start_date, end_date):
    """
    Fetch missing data for a ticker, then write it to the DB.
    """
    existing_data = read_ticker_data_from_db(ticker)
    updated_data = fetch_missing_data_for_ticker(ticker, existing_data, start_date, end_date)
    if updated_data is not None and not updated_data.empty:
        write_ticker_data_to_db(ticker, updated_data)
        return updated_data
    return None

def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    log_path = os.path.join("results", "logs.txt")
    setup_logging(log_path)

    # 1) Create tables in PostgreSQL (safe to call every time)
    create_tables()

    # 2) Get S&P 500 tickers
    tickers = get_sp500_tickers()
    if not tickers:
        logging.error("No tickers found. Exiting.")
        sys.exit(1)

    # 3) Read existing data from DB for each ticker, fetch missing from Yahoo, store updated
    data_dict = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(fetch_and_write_ticker_data, ticker, config.START_DATE, config.END_DATE): ticker
            for ticker in tickers
        }
        for future in futures:
            ticker = futures[future]
            try:
                updated_data = future.result()
                if updated_data is not None and not updated_data.empty:
                    data_dict[ticker] = updated_data
            except Exception as e:
                logging.error(f"Error processing ticker {ticker}: {e}")

    if not data_dict:
        logging.error("No valid data for any tickers. Exiting.")
        sys.exit(1)

    # 4) Compute / merge Phase Classification
    df_phases = classify_phases(data_dict, ma_short=config.MA_SHORT, ma_long=config.MA_LONG)
    merge_new_indicator_data(df_phases, "phase_classification")

    # Plot phase distribution
    if df_phases.empty or df_phases.isnull().all().all():
        logging.error("No valid data for phase distribution. Skipping plots.")
    else:
        phases = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
        for phase in phases:
            filename = os.path.join("results", f"phase_{phase.lower()}.png")
            save_plot(phase, df_phases, filename)

    # 5) Compute / merge Breadth Indicators
    indicator_tasks = [
        (compute_adv_decline, "adv_decline"),
        (compute_adv_decline_volume, "adv_decline_volume"),
        (compute_new_high_low, "new_high_low"),
        (compute_percent_above_ma, "percent_above_ma"),
    ]

    with ThreadPoolExecutor() as executor:
        for compute_func, indicator_name in indicator_tasks:
            result_df = compute_func(data_dict)
            executor.submit(merge_new_indicator_data, result_df, indicator_name)

    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
