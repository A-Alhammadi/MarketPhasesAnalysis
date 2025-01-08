# phase_analysis.py

import logging
import pandas as pd

def classify_phases(data_dict, ma_short=50, ma_long=200):
    """
    Classify each ticker's daily state into phases, then compute a percentage
    distribution of each phase across all tickers on each date.
    """
    for ticker, df in data_dict.items():
        df.sort_index(inplace=True)

        # Debug check: make sure "Close" is truly a single column Series
        if "Close" not in df.columns:
            logging.warning(f"Ticker {ticker} is missing 'Close' column. Skipping.")
            continue
        # Debug: see if df["Close"] is actually a Series or DataFrame
        if isinstance(df["Close"], pd.DataFrame):
            logging.error(f"DEBUG: {ticker} has 'Close' as a DataFrame with shape: {df['Close'].shape}")
        else:
            logging.debug(f"DEBUG: {ticker} 'Close' is Series of shape {df['Close'].shape}")

        # Calculate rolling means for short & long windows
        df[f"SMA_{ma_short}"] = df["Close"].rolling(window=ma_short).mean()
        df[f"SMA_{ma_long}"]  = df["Close"].rolling(window=ma_long).mean()

        df = df[~df.index.duplicated(keep="first")]

    # Gather all dates present in any DataFrame
    all_dates = sorted(
        set(
            date
            for df in data_dict.values()
            for date in df.index
        )
    )

    # Initialize classification counts
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

    # Tally each ticker's phase on each date
    for ticker, df in data_dict.items():
        for date in df.index:
            try:
                sma_short_val = df.at[date, f"SMA_{ma_short}"]
                sma_long_val  = df.at[date, f"SMA_{ma_long}"]
                price         = df.at[date, "Close"]
            except KeyError:
                continue  # Missing data for this date

            if pd.isna(sma_short_val) or pd.isna(sma_long_val) or pd.isna(price):
                continue

            # Basic logic for phases
            if (sma_short_val > sma_long_val) and (price > sma_short_val) and (price > sma_long_val):
                classification_counts[date]["Bullish"] += 1
            elif (sma_short_val > sma_long_val) and (price < sma_short_val):
                if price < sma_long_val:
                    classification_counts[date]["Distribution"] += 1
                else:
                    classification_counts[date]["Caution"] += 1
            elif (sma_short_val < sma_long_val) and (price < sma_short_val) and (price < sma_long_val):
                classification_counts[date]["Bearish"] += 1
            elif (sma_short_val < sma_long_val) and (price > sma_short_val):
                if price > sma_long_val:
                    classification_counts[date]["Accumulation"] += 1
                else:
                    classification_counts[date]["Recuperation"] += 1

    # Convert tally counts to percentage
    total_tickers = len(data_dict)
    rows = []
    for date in sorted(all_dates):
        row = [date]
        for phase in ["Bullish", "Caution", "Distribution", "Bearish", "Recuperation", "Accumulation"]:
            count = classification_counts[date][phase]
            pct   = (count / total_tickers) * 100 if total_tickers > 0 else 0
            row.append(pct)
        rows.append(row)

    col_names = [
        "Date",
        "Bullish", "Caution", "Distribution",
        "Bearish", "Recuperation", "Accumulation"
    ]
    df_phases = pd.DataFrame(rows, columns=col_names)
    df_phases.set_index("Date", inplace=True)

    return df_phases
