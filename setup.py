###############################################################################
# setup.py
###############################################################################

###############################################################################
# config.py
###############################################################################
config_py = r"""# config.py

# Basic settings for data retrieval
START_DATE = "2021-01-01"
END_DATE   = "2023-12-31"

# Moving average windows
MA_SHORT = 50
MA_LONG = 200
"""

###############################################################################
# indicators/adv_decline.py
###############################################################################
adv_decline_py = r"""# indicators/adv_decline.py

import pandas as pd

def compute_adv_decline(data_dict):
    # Compute the Advance-Decline Line for all tickers combined.
    # data_dict: {ticker: DataFrame}
    # Returns: DataFrame [Date, AdvanceDeclineLine]
    all_dates = sorted(
        list(
            set(
                date
                for df in data_dict.values()
                for date in df.index
            )
        )
    )
    ad_line = []
    cumulative = 0
    for date in all_dates:
        advancers = 0
        decliners = 0
        for ticker, df in data_dict.items():
            if date in df.index:
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_close = df.iloc[idx - 1]["Close"]
                    curr_close = df.iloc[idx]["Close"]
                    if curr_close > prev_close:
                        advancers += 1
                    elif curr_close < prev_close:
                        decliners += 1
        net = advancers - decliners
        cumulative += net
        ad_line.append([date, cumulative])
    ad_df = pd.DataFrame(ad_line, columns=["Date", "AdvanceDeclineLine"])
    ad_df.set_index("Date", inplace=True)
    return ad_df
"""

###############################################################################
# indicators/adv_decline_volume.py
###############################################################################
adv_decline_volume_py = r"""# indicators/adv_decline_volume.py

import pandas as pd

def compute_adv_decline_volume(data_dict):
    # Compute the Advance-Decline Volume Line (Up/Down Volume Line).
    # If a stock closes up vs previous day, add its volume; if down, subtract.
    # Returns DataFrame [Date, AdvDeclVolumeLine]
    all_dates = sorted(
        list(
            set(
                date
                for df in data_dict.values()
                for date in df.index
            )
        )
    )
    volume_line = []
    cumulative = 0
    for date in all_dates:
        up_volume = 0
        down_volume = 0
        for ticker, df in data_dict.items():
            if date in df.index:
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_close = df.iloc[idx - 1]["Close"]
                    curr_close = df.iloc[idx]["Close"]
                    volume = df.iloc[idx]["Volume"]
                    if curr_close > prev_close:
                        up_volume += volume
                    elif curr_close < prev_close:
                        down_volume += volume
        net = up_volume - down_volume
        cumulative += net
        volume_line.append([date, cumulative])
    volume_df = pd.DataFrame(volume_line, columns=["Date", "AdvDeclVolumeLine"])
    volume_df.set_index("Date", inplace=True)
    return volume_df
"""

###############################################################################
# indicators/new_high_low.py
###############################################################################
new_high_low_py = r"""# indicators/new_high_low.py

import pandas as pd
import numpy as np

def compute_new_high_low(data_dict, lookback=252):
    # Compute how many tickers made a new ~52-week high or low each day.
    # Returns DataFrame [Date, NewHighCount, NewLowCount, NHNL_Diff, NHNL_Ratio]
    all_dates = sorted(
        list(
            set(
                date
                for df in data_dict.values()
                for date in df.index
            )
        )
    )
    rolling_info = {}
    for ticker, df in data_dict.items():
        df_sorted = df.sort_index()
        df_sorted["RollingMax"] = df_sorted["Close"].rolling(
            window=lookback, min_periods=1
        ).max()
        df_sorted["RollingMin"] = df_sorted["Close"].rolling(
            window=lookback, min_periods=1
        ).min()
        rolling_info[ticker] = df_sorted

    output = []
    for date in all_dates:
        new_high_count = 0
        new_low_count = 0
        for ticker, df_ in rolling_info.items():
            if date in df_.index:
                idx = df_.index.get_loc(date)
                row = df_.iloc[idx]
                if np.isclose(row["Close"], row["RollingMax"]):
                    new_high_count += 1
                if np.isclose(row["Close"], row["RollingMin"]):
                    new_low_count += 1
        nhnl_diff = new_high_count - new_low_count
        nhnl_ratio = (new_high_count / new_low_count) if new_low_count else np.nan
        output.append([date, new_high_count, new_low_count, nhnl_diff, nhnl_ratio])
    nhnl_df = pd.DataFrame(
        output,
        columns=["Date", "NewHighCount", "NewLowCount", "NHNL_Diff", "NHNL_Ratio"]
    )
    nhnl_df.set_index("Date", inplace=True)
    return nhnl_df
"""

###############################################################################
# indicators/percent_above_ma.py
###############################################################################
percent_above_ma_py = r"""# indicators/percent_above_ma.py

import pandas as pd

def compute_percent_above_ma(data_dict, ma_window=200):
    # Compute % of tickers above the given moving average window.
    # Returns DataFrame [Date, PercentAboveMA]
    all_dates = sorted(
        list(
            set(
                date
                for df in data_dict.values()
                for date in df.index
            )
        )
    )
    for ticker, df in data_dict.items():
        df.sort_index(inplace=True)
        df[f"SMA_{ma_window}"] = df["Close"].rolling(window=ma_window).mean()

    output = []
    total_tickers = len(data_dict)
    for date in all_dates:
        above_count = 0
        for ticker, df in data_dict.items():
            if date in df.index:
                row = df.loc[date]
                ma_val = row.get(f"SMA_{ma_window}", None)
                close_val = row.get("Close", None)
                if ma_val is not None and close_val is not None and close_val > ma_val:
                    above_count += 1
        pct = (above_count / total_tickers) * 100 if total_tickers else 0
        output.append([date, pct])
    pct_df = pd.DataFrame(output, columns=["Date", "PercentAboveMA"])
    pct_df.set_index("Date", inplace=True)
    return pct_df
"""

###############################################################################
# phase_analysis.py
###############################################################################
phase_analysis_py = r"""# phase_analysis.py

import pandas as pd

def classify_phases(data_dict, ma_short=50, ma_long=200):
    # Classify daily phases for each stock and return a DataFrame of percentages.
    for ticker, df in data_dict.items():
        df.sort_index(inplace=True)
        df[f"SMA_{ma_short}"] = df["Close"].rolling(window=ma_short).mean()
        df[f"SMA_{ma_long}"] = df["Close"].rolling(window=ma_long).mean()

    all_dates = sorted(
        list(
            set(
                date
                for df in data_dict.values()
                for date in df.index
            )
        )
    )
    classification_counts = {
        date: {
            "Bullish": 0,
            "Caution": 0,
            "Distribution": 0,
            "Bearish": 0,
            "Recuperation": 0,
            "Accumulation": 0
        }
        for date in all_dates
    }
    for ticker, df in data_dict.items():
        for date in df.index:
            sma50 = df.loc[date, f"SMA_{ma_short}"]
            sma200 = df.loc[date, f"SMA_{ma_long}"]
            price = df.loc[date, "Close"]
            if pd.isna(sma50) or pd.isna(sma200):
                continue
            if (sma50 > sma200) and (price > sma50) and (price > sma200):
                classification_counts[date]["Bullish"] += 1
            elif (sma50 > sma200) and (price < sma50):
                if price < sma200:
                    classification_counts[date]["Distribution"] += 1
                else:
                    classification_counts[date]["Caution"] += 1
            elif (sma50 < sma200) and (price < sma50) and (price < sma200):
                classification_counts[date]["Bearish"] += 1
            elif (sma50 < sma200) and (price > sma50):
                if price > sma200:
                    classification_counts[date]["Accumulation"] += 1
                else:
                    classification_counts[date]["Recuperation"] += 1

    total_tickers = len(data_dict)
    rows = []
    for date in all_dates:
        row = [date]
        for phase in [
            "Bullish",
            "Caution",
            "Distribution",
            "Bearish",
            "Recuperation",
            "Accumulation"
        ]:
            c = classification_counts[date][phase]
            pct = (c / total_tickers) * 100 if total_tickers else 0
            row.append(pct)
        rows.append(row)

    col_names = [
        "Date", "Bullish", "Caution", "Distribution",
        "Bearish", "Recuperation", "Accumulation"
    ]
    df_phases = pd.DataFrame(rows, columns=col_names)
    df_phases.set_index("Date", inplace=True)
    return df_phases
"""

###############################################################################
# main.py
###############################################################################
main_py = r"""# main.py

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
    # Configure file-based logging
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_sp500_tickers():
    # Scrape Wikipedia for S&P 500 tickers
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
            ticker = ticker.replace('.', '-')
            tickers.append(ticker)
    return tickers

def fetch_data_for_ticker(ticker, start_date, end_date):
    # Fetch daily price data for a ticker using yfinance
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logging.warning(f"No data returned for {ticker}.")
            return None
        return data[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as exc:
        logging.error(f"Error fetching data for {ticker}: {exc}")
        return None

def main():
    # Create results folder
    if not os.path.exists("results"):
        os.makedirs("results")

    # Logging
    log_path = os.path.join("results", "logs.txt")
    setup_logging(log_path)

    # Tickers
    tickers = get_sp500_tickers()
    if not tickers:
        logging.error("No tickers found. Exiting.")
        sys.exit(1)

    # Download data
    data_dict = {}
    for ticker in tickers:
        df = fetch_data_for_ticker(ticker, config.START_DATE, config.END_DATE)
        if df is not None and not df.empty:
            data_dict[ticker] = df

    if not data_dict:
        logging.error("No valid data for any tickers. Exiting.")
        sys.exit(1)

    # Classify phases
    df_phases = classify_phases(data_dict, ma_short=config.MA_SHORT, ma_long=config.MA_LONG)
    phase_csv = os.path.join("results", "phase_classification.csv")
    df_phases.to_csv(phase_csv)

    # Plot each phase
    for phase in ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]:
        plt.figure(figsize=(10,6))
        plt.plot(df_phases.index, df_phases[phase], label=phase)
        plt.title(f"{phase} Phase % Over Time")
        plt.xlabel("Date")
        plt.ylabel("Percentage of Total Stocks")
        plt.legend()
        plt.grid(True)
        filename = f"phase_{phase.lower()}_timeseries.png"
        plt.savefig(os.path.join("results", filename))
        plt.close()

    # Indicators
    # A) Advance-Decline
    ad_line = compute_adv_decline(data_dict)
    ad_line_csv = os.path.join("results", "adv_decline.csv")
    ad_line.to_csv(ad_line_csv)
    plt.figure(figsize=(10,6))
    plt.plot(ad_line.index, ad_line["AdvanceDeclineLine"], label="Advance-Decline")
    plt.title("Advance-Decline Line")
    plt.xlabel("Date")
    plt.ylabel("A/D Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("results", "adv_decline_line.png"))
    plt.close()

    # B) Advance-Decline Volume
    ad_volume_line = compute_adv_decline_volume(data_dict)
    ad_volume_csv = os.path.join("results", "adv_decline_volume.csv")
    ad_volume_line.to_csv(ad_volume_csv)
    plt.figure(figsize=(10,6))
    plt.plot(ad_volume_line.index, ad_volume_line["AdvDeclVolumeLine"], label="A/D Volume")
    plt.title("Advance-Decline Volume Line")
    plt.xlabel("Date")
    plt.ylabel("Volume Diff")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("results", "adv_decline_volume_line.png"))
    plt.close()

    # C) New High - New Low
    nhnl_df = compute_new_high_low(data_dict, lookback=252)
    nhnl_csv = os.path.join("results", "new_high_low.csv")
    nhnl_df.to_csv(nhnl_csv)
    plt.figure(figsize=(10,6))
        # NHNL_Diff is new highs - new lows
    plt.plot(nhnl_df.index, nhnl_df["NHNL_Diff"], label="NH-NL Diff")
    plt.title("New High - New Low Difference")
    plt.xlabel("Date")
    plt.ylabel("NH - NL")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("results", "new_high_low_diff.png"))
    plt.close()

    # D) Percent above 200-day MA
    pct_ma_df = compute_percent_above_ma(data_dict, config.MA_LONG)
    pct_ma_csv = os.path.join("results", "percent_above_ma.csv")
    pct_ma_df.to_csv(pct_ma_csv)
    plt.figure(figsize=(10,6))
    plt.plot(pct_ma_df.index, pct_ma_df["PercentAboveMA"], label=f"% Above {config.MA_LONG} MA")
    plt.title(f"Percent Above {config.MA_LONG}-day MA")
    plt.xlabel("Date")
    plt.ylabel("Percent")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("results", "percent_above_ma.png"))
    plt.close()

    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
"""

###############################################################################
# Putting it all together in one script
###############################################################################
def write_all_files():
    import os

    # Ensure 'indicators' folder exists
    if not os.path.exists("indicators"):
        os.makedirs("indicators")

    # Write config.py
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(config_py)

    # Write main.py
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(main_py)

    # Write phase_analysis.py
    with open("phase_analysis.py", "w", encoding="utf-8") as f:
        f.write(phase_analysis_py)

    # Write the indicator scripts
    with open(os.path.join("indicators", "adv_decline.py"), "w", encoding="utf-8") as f:
        f.write(adv_decline_py)
    with open(os.path.join("indicators", "adv_decline_volume.py"), "w", encoding="utf-8") as f:
        f.write(adv_decline_volume_py)
    with open(os.path.join("indicators", "new_high_low.py"), "w", encoding="utf-8") as f:
        f.write(new_high_low_py)
    with open(os.path.join("indicators", "percent_above_ma.py"), "w", encoding="utf-8") as f:
        f.write(percent_above_ma_py)

    print("All files created successfully. Next, run `python main.py`.")

if __name__ == "__main__":
    write_all_files()
