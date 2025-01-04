# main.py

import os
import sys
import logging
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import config
from phase_analysis import classify_phases
from indicators.adv_decline import compute_adv_decline
from indicators.adv_decline_volume import compute_adv_decline_volume
from indicators.new_high_low import compute_new_high_low
from indicators.percent_above_ma import compute_percent_above_ma

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_sp500_tickers():
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

def fetch_data_for_ticker(ticker, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                logging.warning(f"No data returned for {ticker}. Attempt {attempt + 1} of {retries}.")
                continue

            # Remove timezone information
            data.index = pd.to_datetime(data.index).tz_localize(None)

            # Flatten MultiIndex columns if necessary
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            # Debugging: Check what columns are present
            print(f"DEBUG: Raw data for {ticker}:\n{data.head()}")

            # Ensure required columns are present
            if not all(col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
                logging.warning(f"Ticker {ticker} is missing columns: {data.columns}. Skipping.")
                return None

            return data
        except Exception as exc:
            logging.error(f"Error fetching data for {ticker}: {exc}. Attempt {attempt + 1} of {retries}.")
            time.sleep(1)
    logging.warning(f"Failed to fetch data for {ticker} after {retries} attempts.")
    return None


def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    log_path = os.path.join("results", "logs.txt")
    setup_logging(log_path)

    tickers = get_sp500_tickers()
    if not tickers:
        logging.error("No tickers found. Exiting.")
        sys.exit(1)

    data_dict = {}
    for ticker in tickers:
        df = fetch_data_for_ticker(ticker, config.START_DATE, config.END_DATE)
        if df is not None and not df.empty:
            data_dict[ticker] = df
        else:
            logging.warning(f"Ticker {ticker} has no valid data or required columns. Skipping.")


    if not data_dict:
        logging.error("No valid data for any tickers. Exiting.")
        sys.exit(1)

    # Classify phases
    df_phases = classify_phases(data_dict, ma_short=config.MA_SHORT, ma_long=config.MA_LONG)
    phase_csv = os.path.join("results", "phase_classification.csv")
    df_phases.to_csv(phase_csv)

    # Plot each phase
    for phase in ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]:
        plt.figure(figsize=(10, 6))
        plt.plot(df_phases.index, df_phases[phase], label=phase)
        plt.title(f"{phase} Phase % Over Time")
        plt.xlabel("Date")
        plt.ylabel("Percentage of Total Stocks")
        plt.legend()
        plt.grid(True)
        filename = f"phase_{phase.lower()}_timeseries.png"
        plt.savefig(os.path.join("results", filename))
        plt.close()

    # Breadth indicators
    ad_line = compute_adv_decline(data_dict)
    ad_line.to_csv(os.path.join("results", "adv_decline.csv"))
    ad_volume_line = compute_adv_decline_volume(data_dict)
    ad_volume_line.to_csv(os.path.join("results", "adv_decline_volume.csv"))
    nhnl_df = compute_new_high_low(data_dict, lookback=252)
    nhnl_df.to_csv(os.path.join("results", "new_high_low.csv"))
    pct_ma_df = compute_percent_above_ma(data_dict, config.MA_LONG)
    pct_ma_df.to_csv(os.path.join("results", "percent_above_ma.csv"))

    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
