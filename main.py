import os
import sys
import logging
import time
import requests
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

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


def save_plot(phase, df_phases_merged, filename):
    """
    Save a plot for a specific phase, only if the data has changed.
    """
    if not os.path.exists(filename):
        logging.info(f"Generating plot for {phase}...")
        plt.figure(figsize=(10, 6))
        plt.plot(df_phases_merged.index, df_phases_merged[phase], label=phase)
        plt.title(f"{phase} Phase % Over Time")
        plt.xlabel("Date")
        plt.ylabel("Percentage of Total Stocks")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename, dpi=100)  # Save with reduced DPI for faster saving
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
    rows = table.find_all('tr')
    tickers = []
    for row in rows[1:]:
        cols = row.find_all('td')
        if cols:
            ticker = cols[0].text.strip()
            ticker = ticker.replace('.', '-')  # Adjust for Yahoo Finance format
            tickers.append(ticker)
    return tickers

import pandas_market_calendars as mcal

def filter_trading_days(start, end):
    """
    Filters a date range to include only valid trading days (weekdays and market-open days).
    Uses the NYSE market calendar by default.
    """
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start, end_date=end)
    trading_days = schedule.index.normalize()  # Normalize to keep only the date part
    return trading_days


def read_local_data_for_ticker(ticker):
    """
    Reads existing CSV for the ticker if it exists, returning a DataFrame.
    If no file exists or file is empty, returns an empty DataFrame.
    """
    data_dir = os.path.join("results", "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{ticker}.csv")

    if not os.path.isfile(file_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        # Ensure standard columns exist
        expected_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not expected_cols.issubset(set(df.columns)):
            logging.warning(f"{ticker}.csv missing required columns. Returning empty DataFrame.")
            return pd.DataFrame()
        return df
    except Exception as e:
        logging.error(f"Error reading existing CSV for {ticker}: {e}")
        return pd.DataFrame()


def write_local_data_for_ticker(ticker, df):
    """
    Writes the updated DataFrame for a single ticker into results/data/{ticker}.csv
    only if the data has changed.
    """
    data_dir = os.path.join("results", "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{ticker}.csv")

    # Check if the file exists and is identical to the new data
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        if existing_df.equals(df):  # If data is identical, skip writing
            logging.info(f"{ticker}: No changes detected. Skipping file write.")
            return

    # Write to file if new or changed
    logging.info(f"{ticker}: Writing updated data to file.")
    df.to_csv(file_path)

def get_last_business_day(ref_date):
    """
    Returns the last business weekday (Mon-Fri) on or before ref_date.
    """
    while ref_date.weekday() >= 5:  # Saturday=5, Sunday=6
        ref_date -= timedelta(days=1)
    return ref_date

def get_last_completed_trading_day():
    """
    Returns the last trading day for which we expect full data.
    If today is Saturday/Sunday, return last Friday.
    If today is a weekday but before 16:00 local time, return yesterday (or last Friday if yesterday is weekend).
    Otherwise, return today.
    """
    now = datetime.now()
    
    # If it's Saturday (5) or Sunday (6), move back to last Friday
    while now.weekday() >= 5:  # Saturday=5, Sunday=6
        now -= timedelta(days=1)

    # If it's a weekday (Mon-Fri) but before 16:00 local time, go back one more day
    if now.hour < 16:
        now = now - timedelta(days=1)
        # If that day is Saturday/Sunday, keep going back to Friday
        while now.weekday() >= 5:
            now -= timedelta(days=1)

    return now.replace(hour=0, minute=0, second=0, microsecond=0)

def get_ticker_listing_date(ticker):
    """
    Fetches the earliest available trading date for a ticker using Yahoo Finance.
    If no data is available, returns None.
    """
    try:
        # Fetch minimal data to determine earliest available date
        data = yf.download(ticker, period="max", interval="1d", progress=False)
        if not data.empty:
            return data.index.min()
        else:
            return None
    except Exception as e:
        logging.warning(f"Failed to fetch listing date for {ticker}: {e}")
        return None

def fetch_missing_data_for_ticker(ticker, existing_df, config_start_date, config_end_date=None, retries=3):
    """
    For a given ticker and existing DataFrame:
      1) Determine missing date ranges based on config_start_date and config_end_date.
      2) Only if needed, connect to Yahoo to fetch *just* the missing intervals.
      3) Combine everything and return the final DataFrame.
    """

    # -------------------------------------------
    # 1. Parse the user-specified start/end dates
    # -------------------------------------------
    start_dt = pd.to_datetime(config_start_date)
    if config_end_date is None:
        # Use last COMPLETED trading day so we don't try to fetch partial data for today
        end_dt = get_last_completed_trading_day()
    else:
        end_dt = pd.to_datetime(config_end_date)
        # If the user picks a weekend day or a weekday morning, push it back to last business day or completed day
        # (same logic as before, or you can apply get_last_completed_trading_day() conditionally)
        end_dt = get_last_business_day(end_dt)


    # If there's no local data, we fetch everything in one shot.
    if existing_df.empty:
        logging.info(f"{ticker}: No local data. Will fetch from {start_dt.date()} to {end_dt.date()}.")
        missing_intervals = [(start_dt, end_dt)]
    else:
        # Convert the index to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(existing_df.index):
            existing_df.index = pd.to_datetime(existing_df.index)

        earliest_local = existing_df.index.min()
        latest_local = existing_df.index.max()

        missing_intervals = []

        # 2a. Check if there's data missing at the beginning
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


    # If no missing intervals, just return existing data
    if not missing_intervals:
        logging.info(f"{ticker}: No missing intervals. Skipping Yahoo download.")
        return existing_df

    logging.info(f"{ticker}: Missing intervals {missing_intervals}")

    # -------------------------------------------------
    # 3. Download data for each missing interval if any
    # -------------------------------------------------
    all_new_parts = []
    for (m_start, m_end) in missing_intervals:
        # If m_start > m_end, skip
        if m_start > m_end:
            continue

        # Also skip purely weekend intervals if they fall exactly on weekend 
        # (rare, but let's be consistent).
        # We'll rely on yfinance to return empty if no trading days in that range.
        success = False
        new_data_part = pd.DataFrame()
        for attempt in range(retries):
            try:
                # yfinance's `end` is exclusive, so add one day
                data = yf.download(
                    ticker,
                    start=m_start,
                    end=m_end + timedelta(days=1),
                    progress=False
                )
                if not data.empty:
                    # Clean up the data
                    data.index = pd.to_datetime(data.index).tz_localize(None)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] for col in data.columns]
                    new_data_part = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                success = True
                break
            except Exception as exc:
                logging.error(f"{ticker}: Error fetching data ({m_start.date()} - {m_end.date()}): {exc}. Attempt {attempt+1} of {retries}")
                time.sleep(1)

        if not success:
            logging.warning(f"{ticker}: Failed to fetch new data for interval {m_start.date()} - {m_end.date()}")
        else:
            all_new_parts.append(new_data_part)

    # ---------------------------------------
    # 4. Combine old + new data, drop duplicates
    # ---------------------------------------
    if all_new_parts:
        combined_new_data = pd.concat(all_new_parts)
        combined = pd.concat([existing_df, combined_new_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        return combined
    else:
        return existing_df

def merge_new_indicator_data(new_df, csv_path):
    """
    Efficiently merges `new_df` with existing data in `csv_path` by checking only the relevant date ranges.
    Writes back to the file only if there are new rows to add.
    """
    if os.path.exists(csv_path):
        try:
            # Load existing data
            old_df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
            
            # Check if the new data overlaps with the existing data
            min_new_date = new_df.index.min()
            max_new_date = new_df.index.max()
            min_old_date = old_df.index.min()
            max_old_date = old_df.index.max()

            # If new data is entirely within the existing range, skip writing
            if min_new_date >= min_old_date and max_new_date <= max_old_date:
                logging.info(f"No new dates in {csv_path}. Skipping file write.")
                return

            # Combine old and new data
            merged_df = pd.concat([old_df, new_df]).drop_duplicates().sort_index()

            # Write only if there are new rows
            if not old_df.equals(merged_df):
                logging.info(f"Writing updated indicator data to {csv_path}.")
                merged_df.to_csv(csv_path)
            else:
                logging.info(f"No changes detected in {csv_path}. Skipping file write.")
        except Exception as e:
            logging.error(f"Error merging indicator data for {csv_path}: {e}")
            # Fallback: write new data
            logging.info(f"Fallback: Writing new data to {csv_path}.")
            new_df.to_csv(csv_path)
    else:
        # If file doesn't exist, write new data
        logging.info(f"Writing new indicator data to {csv_path}.")
        new_df.to_csv(csv_path)


def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    log_path = os.path.join("results", "logs.txt")
    setup_logging(log_path)

    # 1. Get the S&P 500 tickers
    tickers = get_sp500_tickers()
    if not tickers:
        logging.error("No tickers found. Exiting.")
        sys.exit(1)

    # 2. For each ticker, read existing data, fetch only missing days, then write updated data.
    data_dict = {}
    for ticker in tickers:
        existing_data = read_local_data_for_ticker(ticker)
        updated_data = fetch_missing_data_for_ticker(
            ticker,
            existing_data,
            config.START_DATE,   # from config
            config.END_DATE      # from config (or None)
        )
        if updated_data is not None and not updated_data.empty:
            write_local_data_for_ticker(ticker, updated_data)
            data_dict[ticker] = updated_data
        else:
            logging.warning(f"No valid data found (old or new) for {ticker}. Skipping.")

    # If we have no data at all, exit
    if not data_dict:
        logging.error("No valid data for any tickers. Exiting.")
        sys.exit(1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Compute / merge Phase Classification
    df_phases = classify_phases(data_dict, ma_short=config.MA_SHORT, ma_long=config.MA_LONG)
    phase_csv = os.path.join("results", "phase_classification.csv")
    merge_new_indicator_data(df_phases, phase_csv)

    # Re-read the merged file for plotting
    df_phases_merged = pd.read_csv(phase_csv, parse_dates=True, index_col=0)

    # Parallelize plot generation
    phases = ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]
    with ThreadPoolExecutor() as executor:
        for phase in phases:
            filename = os.path.join("results", f"phase_{phase.lower()}_timeseries.png")
            executor.submit(save_plot, phase, df_phases_merged, filename)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Compute / merge Breadth Indicators
    
    # Advance-Decline
    ad_line = compute_adv_decline(data_dict)
    ad_csv = os.path.join("results", "adv_decline.csv")
    merge_new_indicator_data(ad_line, ad_csv)

    # Advance-Decline Volume
    ad_volume_line = compute_adv_decline_volume(data_dict)
    ad_volume_csv = os.path.join("results", "adv_decline_volume.csv")
    merge_new_indicator_data(ad_volume_line, ad_volume_csv)

    # New High / New Low
    nhnl_df = compute_new_high_low(data_dict, lookback=252)
    nhnl_csv = os.path.join("results", "new_high_low.csv")
    merge_new_indicator_data(nhnl_df, nhnl_csv)

    # Percent Above MA
    pct_ma_df = compute_percent_above_ma(data_dict, ma_window=config.MA_LONG)
    pct_ma_csv = os.path.join("results", "percent_above_ma.csv")
    merge_new_indicator_data(pct_ma_df, pct_ma_csv)

    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
