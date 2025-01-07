#main.py

import os
import sys
import logging
import time
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

import config
from phase_analysis import classify_phases
from indicators.adv_decline import compute_adv_decline
from indicators.adv_decline_volume import compute_adv_decline_volume
from indicators.new_high_low import compute_new_high_low
from indicators.percent_above_ma import compute_percent_above_ma


def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='a',  # <- Use 'a' so it appends instead of overwriting
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


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
            # Adjust for Yahoo Finance format
            ticker = ticker.replace('.', '-')
            tickers.append(ticker)
    return tickers


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
    """
    data_dir = os.path.join("results", "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    df.to_csv(file_path)


def fetch_missing_data_for_ticker(ticker, existing_df, start_date, end_date=None, retries=3):
    """
    Given the existing DataFrame for the ticker, fetch only the missing data
    from the start_date up to the end_date (default = today). Fills gaps in the existing data.
    """
    # Determine the start and end dates for fetching
    fetch_start = pd.to_datetime(start_date)
    if existing_df.empty:
        earliest_date = None
    else:
        earliest_date = existing_df.index.min()

    # If existing_df has data but there's a gap before the earliest date
    if earliest_date is not None and fetch_start < earliest_date:
        fetch_start = pd.to_datetime(start_date)  # Ensure we fetch from the start_date

    fetch_end = pd.Timestamp.today().normalize() if end_date is None else pd.to_datetime(end_date)

    # If there's no new data to fetch, return the existing data
    if fetch_start > fetch_end:
        logging.info(f"No missing data for {ticker} from {fetch_start.date()} to {fetch_end.date()}.")
        return existing_df

    # Attempt to fetch missing data
    new_data = None
    for attempt in range(retries):
        try:
            data = yf.download(
                ticker,
                start=fetch_start,
                end=fetch_end + timedelta(days=1),  # end is exclusive
                progress=False
            )
            if not data.empty:
                # Remove timezone info
                data.index = pd.to_datetime(data.index).tz_localize(None)
                # Flatten MultiIndex columns if necessary
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]
                new_data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                break
            else:
                logging.warning(f"No data returned for {ticker}. Attempt {attempt+1} of {retries}.")
        except Exception as exc:
            logging.error(f"Error fetching data for {ticker}: {exc}. Attempt {attempt+1} of {retries}.")
            time.sleep(1)

    if new_data is None or new_data.empty:
        logging.warning(f"Failed to fetch new data for {ticker} after {retries} attempts.")
        return existing_df

    # Combine old + new, drop duplicates if any, sort
    combined = pd.concat([existing_df, new_data])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)
    return combined


def merge_new_indicator_data(new_df, csv_path):
    """
    Reads an existing CSV of the same structure, merges with `new_df`, 
    and writes back so old data is retained (and updated if needed) 
    while new data is appended.
    """
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
            # Outer join on the index
            merged_df = old_df.combine_first(new_df)
            # Overwrite old data with new where there's overlap
            # (combine_first only fills NaNs, so to truly "update" we can do an update)
            merged_df.update(new_df)
            merged_df.sort_index(inplace=True)
            merged_df.to_csv(csv_path)
        except Exception as e:
            logging.error(f"Error merging indicator data from {csv_path}: {e}")
            # Fallback: just write new data
            new_df.to_csv(csv_path)
    else:
        # If file doesn't exist, just write new data
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

    # 2. For each ticker, read existing data, fetch only missing days, write updated.
    data_dict = {}
    for ticker in tickers:
        existing_data = read_local_data_for_ticker(ticker)
        updated_data = fetch_missing_data_for_ticker(
            ticker,
            existing_data,
            config.START_DATE,   # from config
            config.END_DATE      # from config (or you can default to today's date)
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

    # Re-read the merged file to ensure we have the complete version for plotting
    df_phases_merged = pd.read_csv(phase_csv, parse_dates=True, index_col=0)

    # Plot each phase with the merged data
    for phase in ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]:
        plt.figure(figsize=(10, 6))
        plt.plot(df_phases_merged.index, df_phases_merged[phase], label=phase)
        plt.title(f"{phase} Phase % Over Time")
        plt.xlabel("Date")
        plt.ylabel("Percentage of Total Stocks")
        plt.legend()
        plt.grid(True)
        filename = f"phase_{phase.lower()}_timeseries.png"
        plt.savefig(os.path.join("results", filename))
        plt.close()

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
