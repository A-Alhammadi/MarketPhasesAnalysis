# indicators/new_high_low.py

import pandas as pd
import numpy as np

def compute_new_high_low(data_dict, lookback=252):
    """
    Compute how many tickers made a new ~52-week high or low each day.
    Returns DataFrame [Date, NewHighCount, NewLowCount, NHNL_Diff, NHNL_Ratio].
    """
    all_dates = sorted(
        set(
            date
            for df in data_dict.values()
            for date in df.index
        )
    )

    rolling_info = {}
    for ticker, df in data_dict.items():
        if "Close" not in df.columns:
            logging.warning(f"Ticker {ticker} is missing 'Close' column. Skipping.")
            continue
        df_sorted = df.sort_index()
        df_sorted["RollingMax"] = df_sorted["Close"].rolling(window=lookback, min_periods=1).max()
        df_sorted["RollingMin"] = df_sorted["Close"].rolling(window=lookback, min_periods=1).min()
        rolling_info[ticker] = df_sorted

    output = []
    for date in all_dates:
        new_high_count = 0
        new_low_count = 0
        for ticker, df_ in rolling_info.items():
            if date in df_.index:
                idx = df_.index.get_loc(date)
                row = df_.iloc[idx]
                if pd.isna(row["Close"]) or pd.isna(row["RollingMax"]) or pd.isna(row["RollingMin"]):
                    continue
                if row["Close"] == row["RollingMax"]:
                    new_high_count += 1
                if row["Close"] == row["RollingMin"]:
                    new_low_count += 1
        nhnl_diff = new_high_count - new_low_count
        nhnl_ratio = (new_high_count / new_low_count) if new_low_count else None
        output.append([date, new_high_count, new_low_count, nhnl_diff, nhnl_ratio])

    nhnl_df = pd.DataFrame(
        output,
        columns=["Date", "NewHighCount", "NewLowCount", "NHNL_Diff", "NHNL_Ratio"]
    )
    nhnl_df.set_index("Date", inplace=True)
    return nhnl_df

