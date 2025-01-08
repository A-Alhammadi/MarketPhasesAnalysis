import os
import sys
import logging
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import psycopg2
import psycopg2.extras

import config
from phase_analysis import classify_phases

# Indicators
from indicators.adv_decline import compute_adv_decline
from indicators.adv_decline_volume import compute_adv_decline_volume
from indicators.new_high_low import compute_new_high_low
from indicators.percent_above_ma import compute_percent_above_ma
from indicators.extra_indicators import (
    compute_mcclellan,
    compute_index_of_fear_greed,
    compute_trend_intensity_index,
    compute_chaikin_volatility,
    compute_chaikin_money_flow
)
from indicators.trin import compute_trin


def setup_logging(log_path: str):
    """
    Set up logging to a file with DEBUG level.
    """
    logging.basicConfig(
        filename=log_path,
        filemode="a",  # Append instead of overwrite
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_db_connection():
    """
    Create and return a new PostgreSQL connection using config credentials.
    """
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASS
    )


def create_tables():
    """
    Creates the necessary tables in PostgreSQL if they do not already exist:
      - price_data
      - indicator_data
    """
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # price_data table
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

        # indicator_data table
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
        if cur:
            cur.close()
        if conn:
            conn.close()


def read_ticker_data_from_db(ticker: str) -> pd.DataFrame:
    """
    Reads existing data for a single ticker from the 'price_data' table.
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
        if not df.empty:
            df.set_index("trade_date", inplace=True)
            df.index = pd.to_datetime(df.index)
            df.index.name = None
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
    except Exception as e:
        logging.error(f"Error reading data for {ticker} from DB: {e}")
    finally:
        if conn:
            conn.close()
    return df


def write_ticker_data_to_db(ticker: str, df: pd.DataFrame):
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
            records.append(
                (
                    ticker,
                    date_.date(),
                    row["Open"],
                    row["High"],
                    row["Low"],
                    row["Close"],
                    int(row["Volume"]) if not pd.isna(row["Volume"]) else None
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
    except Exception as e:
        logging.error(f"Error writing data for {ticker} to DB: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def get_db_min_max_date():
    """
    Returns (min_date, max_date) across ALL tickers from the price_data table.
    If the table is empty, returns (None, None).
    """
    conn = None
    min_date, max_date = None, None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM price_data;")
        row = cur.fetchone()
        if row:
            min_date, max_date = row
    except Exception as e:
        logging.error(f"Error getting min/max date from DB: {e}")
    finally:
        if conn:
            conn.close()
    return min_date, max_date


def fetch_and_write_all_tickers(tickers, start_date, end_date):
    """
    Checks the database for existing data and fetches missing data only if needed.
    If the data range in the database already covers [start_date, end_date], skip fetching.
    Returns:
        data_dict: A dict of DataFrames keyed by ticker
        new_data_found: Boolean indicating whether new data was fetched
    """
    from datetime import timedelta
    import yfinance as yf

    db_min, db_max = get_db_min_max_date()

    cfg_start = pd.to_datetime(start_date) if start_date else None
    if end_date is None:
        cfg_end = get_last_completed_trading_day()
    else:
        cfg_end = pd.to_datetime(end_date)
        cfg_end = get_last_business_day(cfg_end)

    # Quick check: If the DB already covers the entire range, skip fetching
    if db_min and db_max:
        if pd.Timestamp(cfg_start) >= pd.Timestamp(db_min) and pd.Timestamp(cfg_end) <= pd.Timestamp(db_max):
            logging.info("Database covers the entire date range; skipping Yahoo Finance fetch.")
            data_dict = {}
            # Load data for each ticker from DB
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(read_ticker_data_from_db, t): t
                    for t in tickers
                }
                for fut in futures:
                    tck = futures[fut]
                    try:
                        data_dict[tck] = fut.result()
                    except Exception as e:
                        logging.error(f"Error loading data for {tck}: {e}")
            return data_dict, False

    logging.info("Checking for missing data in DB, may fetch from Yahoo Finance...")
    needed_intervals = []

    # If the DB is empty
    if db_min is None or db_max is None:
        needed_intervals.append((cfg_start, cfg_end))
    else:
        if cfg_start < pd.Timestamp(db_min):
            needed_intervals.append(
                (cfg_start, pd.Timestamp(db_min) - pd.Timedelta(days=1))
            )
        if cfg_end > pd.Timestamp(db_max):
            needed_intervals.append(
                (pd.Timestamp(db_max) + pd.Timedelta(days=1), cfg_end)
            )

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
                for (mstart, mend) in needed_intervals:
                    if mstart <= mend:
                        fetched_part = yf.download(
                            tck,
                            start=mstart,
                            end=mend + timedelta(days=1),
                            interval=config.DATA_FETCH_INTERVAL,
                            progress=False
                        )
                        if not fetched_part.empty:
                            if isinstance(fetched_part.columns, pd.MultiIndex):
                                fetched_part.columns = fetched_part.columns.droplevel(level=1)
                            if "Adj Close" in fetched_part.columns:
                                fetched_part.drop(columns=["Adj Close"], inplace=True)

                            needed_cols = ["Open", "High", "Low", "Close", "Volume"]
                            for col in needed_cols:
                                if col not in fetched_part.columns:
                                    logging.warning(f"Ticker {tck} missing {col} after fetch. Skipping chunk.")
                                    continue

                            fetched_part = fetched_part[needed_cols]
                            existing_data = pd.concat([existing_data, fetched_part])
                            new_data_found = True

                if "Close" in existing_data.columns:
                    existing_data.sort_index(inplace=True)
                    write_ticker_data_to_db(tck, existing_data)
                    data_dict[tck] = existing_data
                else:
                    logging.warning(f"Ticker {tck} is missing 'Close' column. Skipping.")
            except Exception as e:
                logging.error(f"Error processing ticker {tck}: {e}")

    return data_dict, new_data_found


def get_sp500_tickers():
    """
    Fetch the current S&P 500 constituents from Wikipedia, adjusting '.' to '-' for Yahoo format.
    """
    import requests
    from bs4 import BeautifulSoup

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
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
            ticker = ticker.replace(".", "-")  # Adjust for Yahoo format
            tickers.append(ticker)
    return tickers


def get_last_business_day(ref_date):
    """
    Returns the last business day on or before ref_date.
    """
    while ref_date.weekday() >= 5:  # Sat=5, Sun=6
        ref_date -= timedelta(days=1)
    return ref_date


def get_last_completed_trading_day():
    """
    Returns the last completed trading day (i.e., if today's market hasn't closed,
    go to the previous day).
    """
    now = datetime.now()
    while now.weekday() >= 5:  # If weekend
        now -= timedelta(days=1)
    if now.hour < 16:
        now -= timedelta(days=1)
        while now.weekday() >= 5:
            now -= timedelta(days=1)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def merge_new_indicator_data(new_df: pd.DataFrame, indicator_name: str):
    """
    Store indicator data in 'indicator_data' table, up to 5 numeric columns.
    """
    if new_df.empty:
        logging.info(f"No new data for indicator {indicator_name}")
        return

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Up to 5 columns
        columns = list(new_df.columns)[:5]

        for dt_, row in new_df.iterrows():
            dt_ = dt_.date()
            values = row.values.tolist()
            # Pad to length 5 with None
            values += [None] * (5 - len(values))
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
                cur.execute(insert_query, (indicator_name, dt_, *values))
            except Exception as e:
                logging.error(f"Error merging {indicator_name} data for {dt_}: {e}")

        conn.commit()
    except Exception as e:
        logging.error(f"Error in merge_new_indicator_data({indicator_name}): {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def check_and_save_missing_phase_plots(phases, df_phases):
    """
    If we are NOT making new plots (due to no new data),
    still create a plot if:
      - The config for that phase is True
      - The file does not already exist
    """
    # If we want a single "all phases" plot and it doesn't exist
    if config.PLOT_PHASES.get("AllPhases", False):
        all_phases_file = os.path.join(config.RESULTS_DIR, "all_phases.png")
        if not os.path.exists(all_phases_file):
            plt.figure(figsize=(12, 8))
            for phase in phases:
                plt.plot(df_phases.index, df_phases[phase], label=phase)
            plt.title("All Phases Over Time")
            plt.xlabel("Date")
            plt.ylabel("Percentage of Total Stocks")
            plt.legend()
            plt.grid(True)
            plt.savefig(all_phases_file, dpi=100)
            plt.close()

    # Individual phases
    for phase in phases:
        if config.PLOT_PHASES.get(phase, False):
            phase_file = os.path.join(config.RESULTS_DIR, f"phase_{phase.lower()}.png")
            if not os.path.exists(phase_file):
                plt.figure(figsize=(10, 6))
                plt.plot(df_phases.index, df_phases[phase], label=phase)
                plt.title(f"{phase} Phase % Over Time")
                plt.xlabel("Date")
                plt.ylabel("Percentage of Total Stocks")
                plt.legend()
                plt.grid(True)
                plt.savefig(phase_file, dpi=100)
                plt.close()


def check_and_save_missing_indicator_plots(indicator_name, df):
    """
    If we are NOT making new plots (due to no new data),
    still create a plot if:
      - The config for that indicator is True
      - The file does not already exist
    """
    if not config.PLOT_INDICATORS.get(indicator_name, False):
        return  # Plotting disabled

    if df.empty:
        return

    for col in df.columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        # Remove leading constant portion
        first_val = col_data.iloc[0]
        changed_mask = col_data.ne(first_val)
        if changed_mask.any():
            first_change_idx = changed_mask.idxmax()
            col_data = col_data.loc[first_change_idx:]

        plot_file = os.path.join(config.RESULTS_DIR, f"{indicator_name}_{col}.png")
        if not os.path.exists(plot_file):
            plt.figure(figsize=(10, 6))
            plt.plot(col_data.index, col_data.values, label=col)
            plt.title(f"{indicator_name} - {col}")
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_file, dpi=100)
            plt.close()


def save_phase_plots(phases, df_phases):
    """
    Normal function that always generates the plots (regardless of existing files).
    """

    if config.PLOT_PHASES.get("AllPhases", False):
        plt.figure(figsize=(12, 8))
        for phase in phases:
            plt.plot(df_phases.index, df_phases[phase], label=phase)
        plt.title("All Phases Over Time")
        plt.xlabel("Date")
        plt.ylabel("Percentage of Total Stocks")
        plt.legend()
        plt.grid(True)
        filename = os.path.join(config.RESULTS_DIR, "all_phases.png")
        plt.savefig(filename, dpi=100)
        plt.close()

    for phase in phases:
        if config.PLOT_PHASES.get(phase, False):
            plt.figure(figsize=(10, 6))
            plt.plot(df_phases.index, df_phases[phase], label=phase)
            plt.title(f"{phase} Phase % Over Time")
            plt.xlabel("Date")
            plt.ylabel("Percentage of Total Stocks")
            plt.legend()
            plt.grid(True)
            filename = os.path.join(config.RESULTS_DIR, f"phase_{phase.lower()}.png")
            plt.savefig(filename, dpi=100)
            plt.close()

def save_plot(phase, df_phases_merged, filename):
    """
    Save a plot for a specific phase, overwriting the file each run.
    """
    # Check if the phase exists in the DataFrame
    if phase not in df_phases_merged.columns:
        logging.warning(f"Phase '{phase}' not found in the DataFrame. Skipping plot.")
        return

    # Drop any rows with NaN values for this phase
    phase_data = df_phases_merged[phase].dropna()
    if phase_data.empty:
        logging.warning(f"No data available for phase '{phase}'. Skipping plot.")
        return

    # Create and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(phase_data.index, phase_data.values, label=phase)
    plt.title(f"{phase} Phase % Over Time")
    plt.xlabel("Date")
    plt.ylabel("Percentage of Total Stocks")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=100)
    plt.close()
    logging.info(f"Saved plot for phase: {phase} at {filename}")

def save_indicator_plots(indicator_name, df):
    """
    Normal function that always generates the plots (regardless of existing files).
    """
    if not config.PLOT_INDICATORS.get(indicator_name, False):
        return

    if df.empty:
        return

    for col in df.columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        # Remove leading constant portion
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
        filename = os.path.join(config.RESULTS_DIR, f"{indicator_name}_{col}.png")
        plt.savefig(filename, dpi=100)
        plt.close()

def main():
    """
    Main workflow:
      1. Create tables if not exist.
      2. Get tickers.
      3. Fetch missing data => DB => data_dict.
      4. If no new data => only create missing plots for phases/indicators.
      5. Else, do normal logic: compute phases => DB => create plots, and indicators => DB => create plots.
    """
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    log_path = os.path.join(config.RESULTS_DIR, "logs.txt")
    setup_logging(log_path)

    create_tables()

    tickers = get_sp500_tickers()
    if not tickers:
        logging.error("No tickers found. Exiting.")
        sys.exit(1)

    data_dict, new_data_found = fetch_and_write_all_tickers(
        tickers=tickers,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )

    # Phase classification
    phases = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
    df_phases = classify_phases(data_dict, ma_short=config.MA_SHORT, ma_long=config.MA_LONG)

    # If NO new data => ONLY create missing plots if they do not exist
    if not new_data_found:
        logging.info("No new data was fetched. Creating missing plots (if any) then exiting.")
        
        # Check and create missing phase plots
        for phase in phases:
            plot_path = os.path.join(config.RESULTS_DIR, f"phase_{phase.lower()}.png")
            if not os.path.exists(plot_path) and config.PLOT_PHASES.get(phase, False):
                logging.info(f"Creating missing plot for phase: {phase}")
                save_plot(phase, df_phases, plot_path)

        # Check and create missing indicator plots
        indicator_info = [
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
        for compute_func, indicator_name in indicator_info:
            plot_dir = config.RESULTS_DIR
            plot_path = os.path.join(plot_dir, f"{indicator_name}.png")
            if not os.path.exists(plot_path) and config.PLOT_INDICATORS.get(indicator_name, False):
                logging.info(f"Creating missing plot for indicator: {indicator_name}")
                result_df = compute_func(data_dict)
                save_indicator_plots(indicator_name, result_df)

        logging.info("Missing plots created. Exiting.")
        return

    # If we HAVE new data => do normal logic: update DB, produce all plots
    merge_new_indicator_data(df_phases, "phase_classification")
    save_phase_plots(phases, df_phases)

    # Indicators
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
        futures = []
        for compute_func, indicator_name in indicator_tasks:
            result_df = compute_func(data_dict)
            futures.append(executor.submit(merge_new_indicator_data, result_df, indicator_name))
            futures.append(executor.submit(save_indicator_plots, indicator_name, result_df))
        # Ensure all tasks are completed before exiting
        for future in futures:
            future.result()

    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
