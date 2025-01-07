#phase_analysis.py

import logging
import pandas as pd

def classify_phases(data_dict, ma_short=50, ma_long=200):
    for ticker, df in data_dict.items():
        df.sort_index(inplace=True)
        if "Close" not in df.columns:
            logging.warning(f"Ticker {ticker} is missing 'Close' column. Skipping SMA calculation.")
            continue

        df[f"SMA_{ma_short}"] = df["Close"].rolling(window=ma_short).mean()
        df[f"SMA_{ma_long}"] = df["Close"].rolling(window=ma_long).mean()
        df = df[~df.index.duplicated(keep="first")]

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
            try:
                sma50 = df.at[date, f"SMA_{ma_short}"]
                sma200 = df.at[date, f"SMA_{ma_long}"]
                price = df.at[date, "Close"]
            except KeyError as e:
                print(f"DEBUG: Missing column for {ticker} on {date}: {e}")
                continue

            if pd.isna(sma50) or pd.isna(sma200) or pd.isna(price):
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
        for phase in ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]:
            count = classification_counts[date][phase]
            pct = (count / total_tickers) * 100 if total_tickers else 0
            row.append(pct)
        rows.append(row)

    col_names = [
        "Date", "Bullish", "Caution", "Distribution",
        "Bearish", "Recuperation", "Accumulation"
    ]
    df_phases = pd.DataFrame(rows, columns=col_names)
    df_phases.set_index("Date", inplace=True)
    return df_phases
